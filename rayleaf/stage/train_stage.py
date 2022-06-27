import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils

from rayleaf.entities.server import Server
from rayleaf.utils.client_count_utils import client_counts_string, count_selected_clients, online


def train(
    server: Server,
    clients: list,
    clients_per_round: int,
    num_epochs: int,
    batch_size: int,
    round: int,
    num_rounds: int,
    log_progress: bool = True
):
    selected_clients = server.select_clients(round, online(clients), num_clients=clients_per_round)
    if log_progress:
        selected_client_counts = count_selected_clients(selected_clients, clients)
        logging_utils.log(utils.ROUND_STR.format(f"Round {round} of {num_rounds}: Training {clients_per_round} clients: {client_counts_string(selected_client_counts)}"))

    server.train_clients(num_epochs=num_epochs, batch_size=batch_size)
    server._update_model()
