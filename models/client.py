from collections import (
    OrderedDict
)

import ray

@ray.remote
class Client:
    
    def __init__(self, client_id: str, train_data: dict, eval_data: dict, model: type, model_settings: tuple, group: list = None, device: str = "cpu") -> None:
        self._model = model(*model_settings).to(device)
        self.device = device

        self._id = client_id
        self._group = group

        self.train_data = self._model.generate_dataset(train_data)
        self.eval_data = self._model.generate_dataset(eval_data)

    def train(self, num_epochs: int = 1, batch_size: int = 10) -> tuple:
        """
        Trains on self._model using the Client's train_data.

        Args:
            num_epochs: int - Number of epochs to train.
            batch_size: int - Size of training batches.
        Return:
            int - Number of samples used in training
            OrderedDict - This Client's weights after training.
        """
        update = self._model.train_model(self.train_data, num_epochs, batch_size, self.device)
        num_train_samples = len(self.train_data)

        return num_train_samples, update

    def test(self, set_to_use: str ="test", batch_size: int = 10) -> dict:
        """
        Tests self._model on self.test_data.
        
        Args:
            set_to_use: str - Set to test on. Should be in ["train", "test"].
        Return:
            str - This Client's id.
            dict - Evaluation results.
        """
        assert set_to_use in ["train", "test", "val"]

        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.eval_data

        return self.id(), self._model.test(data, batch_size, self.device)

    def set_params(self, params: OrderedDict) -> None:
        """
        Setter for this Client's model parameters.

        Args:
            params: OrderedDict - New parameters to assign to this Client.
        """
        self._model.load_state_dict(params)

    def get_params(self) -> OrderedDict:
        """
        Getter for this Client's model parameters.

        Return:
            OrderedDict - This Client's model parameters.
        """
        return self._model.state_dict()

    def num_test_samples(self) -> int:
        """
        Number of test samples for this Client.

        Return:
            int - Number of test samples for this Client.
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data)

    def num_train_samples(self) -> int:
        """
        Number of train samples for this Client.

        Return:
            int - Number of train samples for this Client.
        """
        if self.train_data is None:
            return 0
        return len(self.train_data)

    def num_samples(self) -> int:
        """
        Number samples for this Client.

        Return:
            int - Number of samples for this Client.
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data)

        test_size = 0 
        if self.eval_data is not None:
            test_size = len(self.eval_data)
        return train_size + test_size

    def id(self) -> str:
        """
        This Client's id.

        Return:
            str - This Client's id.
        """
        return self._id

    def group(self) -> list:
        """
        This Client's group.

        Return:
            list - This Client's group.
        """
        return self._group
