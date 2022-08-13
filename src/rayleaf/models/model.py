from pathlib import Path


import torch

from torch import nn, optim
from torch.utils.data import Dataset, default_collate


class Model(nn.Module):

    def __init__(self, seed: float, lr: float, optimizer=optim.SGD) -> None:
        super(Model, self).__init__()

        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer

        self.flops = 0

        self.collate_fn = default_collate
        self.bn_running_param_indices = set()


    def generate_dataset(self, data: dict, dataset_dir: Path) -> Dataset:
        return None


    def set_params(self, params: list) -> None:
        with torch.no_grad():
            for i, param in enumerate(self.get_params()):
                param.copy_(params[i])


    @property
    def params(self) -> list:
        return list(self.parameters())
    

    @params.setter
    def params(self, new_params: list) -> None:
        with torch.no_grad():
            for i, param_tensor in enumerate(self.params):
                param_tensor.copy_(new_params[i])
