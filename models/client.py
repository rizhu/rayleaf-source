from collections import (
    OrderedDict
)

class Client:
    
    def __init__(
            self,
            client_num: int,
            client_id: str,
            train_data: dict,
            eval_data: dict,
            model: type,
            model_settings: tuple,
            group: list = None,
            device: str = "cpu"
        ) -> None:

        self.client_num = client_num

        self.device = device
        self.model = model(*model_settings).to(self.device)

        self.id = client_id
        self.group = group

        self.train_data = self.model.generate_dataset(train_data)
        self.eval_data = self.model.generate_dataset(eval_data)

        self._num_train_samples = len(self.train_data) if self.train_data is not None else 0
        self._num_eval_samples = len(self.eval_data) if self.eval_data is not None else 0
        self._num_samples = self.num_train_samples + self.num_eval_samples


    def train(self):
        self.train_model()

        return self.num_train_samples, self.model_params


    def train_model(self):
        self.model.train_model(self.train_data, self.num_epochs, self.batch_size, self.device)


    def _train(self, num_epochs: int = 1, batch_size: int = 10) -> tuple:
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        return self.train()


    def _eval(self, set_to_use: str ="test", batch_size: int = 10) -> dict:
        assert set_to_use in ["train", "test", "val"]

        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.eval_data

        eval_metrics = self.model.eval_model(data, batch_size, self.device)
        return eval_metrics


    @property
    def model_params(self) -> OrderedDict:
        return self.model.state_dict()


    @model_params.setter
    def model_params(self, params: OrderedDict) -> None:
        self.model.load_state_dict(params)


    @property
    def num_train_samples(self) -> int:
        return self._num_train_samples


    @property
    def num_eval_samples(self) -> int:
        return self._num_eval_samples

    
    @property
    def num_samples(self) -> int:
        return self._num_samples
