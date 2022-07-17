import torch

from thop import profile
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


import rayleaf.models.utils as model_utils
import rayleaf.stats as stats


class Model(nn.Module):

    def __init__(self, seed: float, lr: float, optimizer=optim.SGD) -> None:
        super(Model, self).__init__()

        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer

        self.flops = 0


    def generate_dataset(self, data: dict) -> Dataset:
        return None


    def train_model(self, data: Dataset, num_epochs: int = 1, batch_size: int = 10, device: str = "cpu") -> None:
        train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.train()
        for _ in range(num_epochs):
            self._run_epoch(train_dataloader, device)

        return self.flops


    def _run_epoch(self, dataloader: DataLoader, device: str = "cpu") -> None:

        flops_counted = self.flops > 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            if not flops_counted:
                macs, _ = profile(self, inputs=(X, ), verbose=False)
                self.flops += macs * 2

            probs = self.forward(X)
            loss = self.loss_fn(probs, y)

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
            
            probs = self.forward(X)
            test_loss += self.loss_fn(probs, y).item()

            preds = model_utils.get_predicted_labels(probs)
            correct += model_utils.number_of_correct(preds, y)

        test_loss /= num_batches

        return {
            stats.NUM_CORRECT_KEY: correct,
            stats.NUM_SAMPLES_KEY: size,
            stats.LOSS_KEY: test_loss
        }


    def get_params(self) -> list:
        return list(self.parameters())


    def set_params(self, params: list) -> None:
        with torch.no_grad():
            for i, param in enumerate(self.get_params()):
                param.copy_(params[i])
