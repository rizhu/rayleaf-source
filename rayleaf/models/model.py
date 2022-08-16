from pathlib import Path
from typing import Union


import torch

from torch import nn, optim
from torch.utils.data import Dataset, default_collate


from rayleaf.tensorarray.tensorarray import TensorArray


class Model(nn.Module):

    def __init__(self, seed: float, lr: float, optimizer=optim.SGD) -> None:
        super(Model, self).__init__()

        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer

        self.flops = 0

        self.collate_fn = default_collate
        self.bn_running_param_indices = set()

        self._shapes = None


    @property
    def shapes(self) -> tuple:
        if self._shapes is None:
            self._shapes = self.get_params().shapes

        return self._shapes


    def generate_dataset(self, data: dict, dataset_dir: Path) -> Dataset:
        return None


    def get_params(self) -> TensorArray:
        return TensorArray(self.parameters())
    

    def set_params(self, new_params: Union[list, TensorArray]) -> None:
        with torch.no_grad():
            if isinstance(new_params, TensorArray):
                new_params = new_params.tensors
            for i, param_tensor in enumerate(self.parameters()):
                param_tensor.copy_(new_params[i])
