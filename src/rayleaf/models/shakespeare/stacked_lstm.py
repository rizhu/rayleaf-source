from pathlib import Path


from torch import nn, Tensor
from torch.utils.data import TensorDataset


from rayleaf.models.model import Model
from rayleaf.utils.language_utils import letter_to_index, word_to_indices


class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, num_hidden):
        super(ClientModel, self).__init__(seed, lr)

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_hidden = num_hidden

        self.embedding = nn.Embedding(self.num_classes, 8)
        self.lstm = nn.LSTM(input_size=8, hidden_size=self.num_hidden, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(self.num_hidden * 2, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)


    def generate_dataset(self, data: dict, dataset_dir: Path) -> TensorDataset:
        return TensorDataset(
            Tensor([word_to_indices(x) for x in data["x"]]).long(),
            Tensor([letter_to_index(y) for y in data["y"]]).long()
        )


    def forward(self, x):
        embedding = self.embedding(x)
        _, (hn, _) = self.lstm(embedding)
        hn = hn.transpose(0, 1).reshape(-1, 2 * self.num_hidden)
        logits = self.fc1(hn)

        return logits
