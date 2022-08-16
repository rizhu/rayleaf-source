from pathlib import Path


import pandas as pd


import rayleaf.stats as stats
import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils

from rayleaf.entities.server import constants, Server
from rayleaf.utils.client_count_utils import client_counts_string, count_selected_clients, online


def train(
    server: Server,
    clients: list,
    clients_per_round: int,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    curr_round: int,
    num_rounds: int,
    log_progress: bool = True
):
    selected_clients = server.select_clients(curr_round, online(clients), num_clients=clients_per_round)
    if log_progress:
        selected_client_counts = count_selected_clients(selected_clients, clients)
        logging_utils.log(utils.ROUND_STR.format(f"Round {curr_round} of {num_rounds}: Training {clients_per_round} clients: {client_counts_string(selected_client_counts)}"))

    updates = server.train_clients(num_epochs=num_epochs, batch_size=batch_size)

    metrics_dir = stats.METRICS_DIR(output_dir)
    for update in updates:
        cid = update[constants.CLIENT_ID_KEY]
        for metric_name in update[constants.METRICS_KEY]:
            metric = pd.DataFrame([
                {
                    constants.CLIENT_ID_KEY: cid,
                    constants.ROUND_KEY: curr_round,
                    metric_name: update[constants.METRICS_KEY][metric_name]
                }
            ])
            _append_metric(metric, metric_name, metrics_dir)

    server._update_model()


def _append_metric(metric_df: pd.DataFrame, metric_name: str, metrics_dir: Path):
    csv_file = Path(metrics_dir, f"{metric_name}.csv")
    append_header = not csv_file.is_file()
    metric_df.to_csv(csv_file, mode="a+", index=False, header=append_header)
