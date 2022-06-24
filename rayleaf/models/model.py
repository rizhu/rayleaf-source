from collections import OrderedDict


import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


from rayleaf.metrics.metrics_constants import ACCURACY_KEY, LOSS_KEY


class Model(nn.Module):

    def __init__(self, seed: float, lr: float, optimizer=optim.SGD) -> None:
        super(Model, self).__init__()

        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer


    def generate_dataset(self, data: dict) -> Dataset:
        return None


    def train_model(self, data: Dataset, num_epochs: int = 1, batch_size: int = 10, device: str = "cpu") -> None:
        train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.train()
        for _ in range(num_epochs):
            self.run_epoch(train_dataloader, device)


    def run_epoch(self, dataloader: DataLoader, device: str = "cpu") -> None:

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = self.forward(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    @torch.no_grad()
    def eval_model(self, data: Dataset, batch_size: int = 10, device: str = "cpu") -> dict:
        test_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        size = len(data)
        num_batches = len(test_dataloader)
        self.eval()
        test_loss, correct = 0, 0

        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred = self.forward(X)

            test_loss += self.loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        return {ACCURACY_KEY: correct, LOSS_KEY: test_loss}
