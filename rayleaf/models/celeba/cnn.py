from pathlib import Path


import numpy as np
import torch

from PIL import Image
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset


from rayleaf.models.model import Model


IMAGE_SIZE = 84


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        super(ClientModel, self).__init__(seed, lr)

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = nn.Linear(32 * 5 * 5, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)
        logits = self.fc1(x)

        return logits
        

    def generate_dataset(self, data: dict, dataset_dir: Path):
        img_dir = Path(dataset_dir, "data", "raw", "img_align_celeba")
        assert img_dir.is_dir()

        X = []
        for file_name in data["x"]:
            X.append(self._load_image(Path(img_dir, file_name)))

        X = torch.stack(X)
        y = Tensor(data["y"]).long()

        return TensorDataset(X, y)


    def _load_image(self, img_path: Path):
        img = Image.open(img_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')

        img = Tensor(np.array(img)).permute((2, 0, 1))

        return img
