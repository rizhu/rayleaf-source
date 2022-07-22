from pathlib import Path
import torch


import rayleaf.entities.constants as constants
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

        self.init()

    
    def init(self) -> None:
        pass


    def train(self):
        self.train_model()

        return self.model_params


    def train_model(self, compute_grads: bool = False):
        if compute_grads:
            self.grads = []
            for layer in self.model_params:
                self.grads.append(layer.detach().clone())

        self.model = self.model.to(self.device)
        flops = self.model.train_model(self.train_data, self.num_epochs, self.batch_size, self.device)
        self.model = self.model.to("cpu")

        self.collect_metric(flops, constants.FLOPS_KEY)

        if compute_grads:
            for i, layer in enumerate(self.model_params):
                self.grads[i] = layer - self.grads[i]


    def _train(self, num_epochs: int = 1, batch_size: int = 10) -> tuple:
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


    def _eval(self, set_to_use: str ="test", batch_size: int = 10) -> dict:
        assert set_to_use in ["train", "test", "val"]

        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.eval_data

        self.model = self.model.to(self.device)
        eval_metrics = self.model.eval_model(data, batch_size, self.device)
        self.model = self.model.to("cpu")

        eval_metrics[stats.CLIENT_ID_KEY] = self.id
        return eval_metrics


    @property
    def model_params(self) -> list:
        return self.model.get_params()


    @model_params.setter
    def model_params(self, params: list) -> None:
        self.model.set_params(params)


    @property
    def grads(self) -> list:
        return self._grads

    
    @grads.setter
    def grads(self, grads: list) -> None:
        self._grads = grads


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
