import random

from datetime import datetime


import numpy as np
import pandas as pd
import ray
import torch


# import rayleaf.metrics.writer as metrics_writer
import rayleaf.stage
import rayleaf.utils.logging_utils as logging_utils

from rayleaf.utils.client_count_utils import client_counts_string, count_selected_clients, online


NUM_GPUS = torch.cuda.device_count()


def run_experiment(
    dataset: str,
    dataset_dir: str,
    output_dir: str,
    model: str,
    num_rounds: int,
    eval_every,
    ServerType: type,
    client_types: list,
    clients_per_round: int,
    client_lr: float,
    batch_size: int = 10,
    seed: int = 0,
    use_val_set: bool = False,
    num_epochs: int = 1,
    gpus_per_client_cluster: float = 1,
    num_client_clusters: int = NUM_GPUS,
    save_model: bool = False
) -> None:
    start_time = datetime.now()
    
    ClientModel, model_settings = rayleaf.stage.initialize_resources(
        dataset_input=dataset,
        output_dir=output_dir,
        model=model,
        client_lr=client_lr,
        seed=seed,
    )

    ray.init()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    server, clients = rayleaf.stage.create_entities(
        device=device,
        num_client_clusters=num_client_clusters,
        gpus_per_client_cluster=gpus_per_client_cluster,
        seed=seed,
        model_settings=model_settings,
        ClientModel=ClientModel,
        ServerType=ServerType,
        client_types=client_types,
        dataset=dataset,
        dataset_dir=dataset_dir,
        use_val_set=use_val_set
    )

    if clients_per_round == -1:
        clients_per_round = len(clients)

    client_counts = count_selected_clients(online(clients), clients)
    logging_utils.log(f"{len(clients)} total clients: {client_counts_string(client_counts)}")
    logging_utils.log()

    eval_set = "test" if not use_val_set else "val"
    rayleaf.stage.eval_server(
        server=server,
        num_round=0,
        eval_set=eval_set
    )

    if type(eval_every) == int:
        eval_every = set(range(0, num_rounds + 1, eval_every))
    eval_every = set(eval_every)

    # Simulate training
    for round_number in range(1, num_rounds + 1):
        rayleaf.stage.train(
            server=server,
            clients=clients,
            clients_per_round=clients_per_round,
            num_epochs=num_epochs,
            batch_size=batch_size,
            round=round_number,
            num_rounds=num_rounds,
            log_progress=True
        )

        # Test model
        if round_number in eval_every:
            rayleaf.stage.eval_server(
                server=server,
                num_round=round_number,
                eval_set=eval_set
            )

    rayleaf.stage.teardown(
        save_model=save_model,
        output_dir=output_dir,
        dataset=dataset,
        model=model,
        server=server,
        start_time=start_time
    )

    ray.shutdown()
