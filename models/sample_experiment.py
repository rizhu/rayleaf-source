from main import run_experiment

from server import Server
from client import Client


class FlippingClient(Client):
    """
    Malicious client that communicates flipped weights.
    """
    def train(self):
        self.train_model()

        for param_tensor, layer in self.model_params.items():
            self.model_params[param_tensor] *= -1

        return self.num_train_samples, self.model_params


class AmplifyingClient(Client):
    """
    Malicious client that gradually makes its own weights larger.
    """
    def init(self):
        self.amplifying_factor = 1.2

    def train(self):
        self.train_model()

        for param_tensor, layer in self.model_params.items():
            self.model_params[param_tensor] *= self.amplifying_factor
        
        self.amplifying_factor *= 1.2

        return self.num_train_samples, self.model_params


run_experiment(
    dataset = "femnist",
    model = "cnn",
    num_rounds = 80,
    eval_every = 5,
    ServerType=Server,
    client_types=[(FlippingClient, 2), (AmplifyingClient, 2), (Client, -1)],
    clients_per_round = 20,
    client_lr = 0.06,
    batch_size = 64,
    seed = 0,
    metrics_name = "metrics",
    metrics_dir = "metrics",
    use_val_set = False,
    num_epochs = 5
)
