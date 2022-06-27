import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


import rayleaf.models.utils as model_utils

from rayleaf.metrics.metrics_constants import ACCURACY_KEY, LOSS_KEY
from rayleaf.models.model import Model
from rayleaf.models.speech_commands import utils as sc_utils


ORIGINAL_FREQ = 16000
RESAMPLE_FREQ = 8000


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        super(ClientModel, self).__init__(seed, lr)

        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=2 * 32, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(num_features=2 * 32)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=2 * 32, out_channels=2 * 32, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(num_features=2 * 32)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(2 * 32, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        return x


    def train_model(self, data: Dataset, num_epochs: int = 1, batch_size: int = 256, device: str = "cpu") -> None:
        pin_memory = device == "cuda"

        train_dataloader = DataLoader(data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=False,
            collate_fn=sc_utils.collate_fn,
            pin_memory=pin_memory
        )

        self.train()
        for _ in range(num_epochs):
            self.run_epoch(train_dataloader, device)


    def run_epoch(self, dataloader: DataLoader, device: str = "cpu") -> None:
        resample = torchaudio.transforms.Resample(orig_freq=ORIGINAL_FREQ, new_freq=RESAMPLE_FREQ).to(device)

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = resample(X)

            pred = self.forward(X)
            loss = self.loss_fn(pred.squeeze(), y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    
    @torch.no_grad()
    def eval_model(self, data: Dataset, batch_size: int = 10, device: str = "cpu") -> dict:
        pin_memory = device == "cuda"

        test_dataloader = DataLoader(data, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False,
            collate_fn=sc_utils.collate_fn,
            pin_memory=pin_memory
        )

        size = len(data)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0

        resample = torchaudio.transforms.Resample(orig_freq=ORIGINAL_FREQ, new_freq=RESAMPLE_FREQ).to(device)

        self.eval()
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            X = resample(X)

            probs = self.forward(X)
            probs = probs.squeeze(dim=1)
            try:
                test_loss += self.loss_fn(probs, y).item()
            except Exception as e:
                print(probs.squeeze())
                print(y)
                raise e

            preds = model_utils.get_predicted_labels(probs)
            correct += model_utils.number_of_correct(preds, y)

        test_loss /= num_batches
        correct /= size

        return {ACCURACY_KEY: correct, LOSS_KEY: test_loss}


    def generate_dataset(self, data: Dataset) -> Dataset:
        return data