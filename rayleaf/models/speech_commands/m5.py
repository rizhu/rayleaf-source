from pathlib import Path


import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


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

        self.collate_fn = sc_utils.make_collate_fn(frequency=ORIGINAL_FREQ)
        self.resample = torchaudio.transforms.Resample(orig_freq=ORIGINAL_FREQ, new_freq=RESAMPLE_FREQ)


    def forward(self, x):
        x = self.resample.to(x.device)(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        return x


    def generate_dataset(self, data: Dataset, dataset_dir: Path) -> Dataset:
        return data
