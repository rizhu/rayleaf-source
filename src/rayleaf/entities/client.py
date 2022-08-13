from pathlib import Path


import torch

from thop import profile
from torch import Tensor
from torch.utils.data import DataLoader


import rayleaf.entities.constants as constants
import rayleaf.models.utils as model_utils
import rayleaf.stats as stats


class Client:
    
    def __init__(
            self,
            client_num: int,
            client_id: str,
            dataset_dir: Path,
            train_data: dict,
            eval_data: dict,
            model: type,
            model_settings: dict,
            group: list = None,
            device: str = "cpu"
        ) -> None:

        self.client_num = client_num

        self.device = device
        self.active_device = "cpu"
        self.model = model(**model_settings)
        self.layer_shapes = [layer.shape for layer in self.model_params]

        self.id = client_id
        self.group = group

        self.train_data = self.model.generate_dataset(train_data, dataset_dir)
        self.eval_data = self.model.generate_dataset(eval_data, dataset_dir)

        self._num_train_samples = len(self.train_data) if self.train_data is not None else 0
        self._num_eval_samples = len(self.eval_data) if self.eval_data is not None else 0
        self._num_samples = self.num_train_samples + self.num_eval_samples

        self.metrics = {}
        self.flops = 0
        self.flops_counted = False

        self.init()

    
    def init(self) -> None:
        pass


    def _train(self, model_params: list, num_epochs: int = 1, batch_size: int = 10) -> tuple:
        self.model_params = model_params
        self.server_model_params = model_params

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.metrics = {}

        update = self.train()

        training_result = {
            constants.CLIENT_ID_KEY: self.id,
            constants.NUM_SAMPLES_KEY: self.num_train_samples,
            constants.METRICS_KEY: self.metrics
        }

        if type(update) == dict:
            training_result.update(update)
        else:
            training_result[constants.UPDATE_KEY] = update

        return training_result


    def train(self):
        self.train_model()

        return self.model_params


    def train_model(self, compute_grads: bool = False):
        if compute_grads:
            grads = []
            for layer in self.model_params:
                grads.append(layer.detach().clone())

        self.model.to(self.device)
        self.active_device = self.device

        pin_memory = self.device == "cuda"
        train_dataloader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=pin_memory,
            collate_fn=self.model.collate_fn
        )

        self.model.train()
        for _ in range(self.num_epochs):
            self._run_epoch(train_dataloader)

        self.model.to("cpu")
        self.active_device = "cpu"

        self.collect_metric(self.flops, constants.FLOPS_KEY)

        if compute_grads:
            for i, layer in enumerate(self.model_params):
                grads[i] = layer - grads[i]
            
            return grads


    def _run_epoch(self, dataloader: DataLoader) -> None:

        flops_counted = self.flops_counted

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            if not flops_counted:
                macs, _ = profile(self.model, inputs=(X, ), verbose=False)
                self.flops += macs * 2

            probs = self.model.forward(X).squeeze(dim=1)

            self.model.optimizer.zero_grad()
            loss = self.compute_loss(probs, y)
            loss.backward()
            self.model.optimizer.step()
        
        self.flops_counted = True


    def compute_loss(self, probs: Tensor, targets: Tensor) -> Tensor:
        return self.model.loss_fn(probs, targets)


    def _eval(self, set_to_use: str ="test", batch_size: int = 10) -> dict:
        assert set_to_use in ["train", "test", "val"]

        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.eval_data

        self.model.to(self.device)
        self.active_device = self.device

        eval_metrics = self.eval_model(data, batch_size)

        self.model.to("cpu")
        self.active_device = "cpu"

        eval_metrics[stats.CLIENT_ID_KEY] = self.id
        return eval_metrics


    @torch.no_grad()
    def eval_model(self, data, batch_size: int = 10) -> dict:
        pin_memory = self.device == "cuda"
        test_dataloader = DataLoader(
            data, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=pin_memory,
            collate_fn=self.model.collate_fn
        )

        size = len(data)
        num_batches = len(test_dataloader)

        self.model.eval()
        test_loss, correct = 0, 0

        for X, y in test_dataloader:
            X, y = X.to(self.device), y.to(self.device)
            
            probs = self.model.forward(X).squeeze(dim=1)
            test_loss += self.model.loss_fn(probs, y).item()

            preds = model_utils.get_predicted_labels(probs)
            correct += model_utils.number_of_correct(preds, y)

        test_loss /= num_batches

        return {
            stats.NUM_CORRECT_KEY: correct,
            stats.NUM_SAMPLES_KEY: size,
            stats.LOSS_KEY: test_loss
        }


    @property
    def bn_running_param_indices(self) -> set:
        return self.model.bn_running_param_indices


    @property
    def model_params(self) -> list:
        return self.model.params


    @model_params.setter
    def model_params(self, params: list) -> None:
        self.model.params = params

    
    @property
    def server_model_params(self) -> list:
        res = [p.to(self.active_device) for p in self._server_model_params]
        return res


    @server_model_params.setter
    def server_model_params(self, params: list) -> None:
        self._server_model_params = params


    @property
    def num_train_samples(self) -> int:
        return self._num_train_samples


    @property
    def num_eval_samples(self) -> int:
        return self._num_eval_samples

    
    @property
    def num_samples(self) -> int:
        return self._num_samples


    def param_like_zeros(self, dtype = torch.float32) -> list:
        res = []
        for shape in self.layer_shapes:
            res.append(torch.zeros(shape))
        
        return res.to(dtype)
    

    def param_like_const(self, value: float = 1, dtype = torch.float32) -> list:
        res = []

        for shape in self.layer_shapes:
            res.append(torch.ones(shape) * value)
        
        return res.to(dtype)


    def param_like_ones(self, dtype = torch.float32) -> list:
        return self.param_like_const(dtype=dtype)
    

    def param_like_normal(self, mean: float = 0, std: float = 1, dtype = torch.float32) -> list:
        res = []

        for shape in self.layer_shapes:
            res.append(torch.normal(mean=torch.ones(shape) * mean, std=torch.ones(shape) * std))
        
        return res.to(dtype)


    def collect_metric(self, metric, metric_name):
        self.metrics[metric_name] = metric
