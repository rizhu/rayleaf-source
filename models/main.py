"""Script to run the baselines."""
from datetime import (
    datetime
)
import importlib
import logging
import os
import random
from matplotlib import dviread

import numpy as np
import ray
import torch

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_SETTINGS
from client_manager import ClientManager
from server import Server

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = "metrics/stat_metrics.csv"
SYS_METRICS_PATH = "metrics/sys_metrics.csv"

SECTION_STR = "\n############################## {} ##############################"

logger = logging.getLogger("MAIN")

def main():
    start_time = datetime.now()
    ray.init()

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_path = f"{args.dataset}/{args.model}.py"
    if not os.path.exists(model_path):
        print("Please specify a valid dataset and a valid model.")
    model_path = f"{args.dataset}.{args.model}"
    
    print(SECTION_STR.format(model_path))
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, "ClientModel")

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Create 2 models
    model_settings = MODEL_SETTINGS[model_path]
    if args.lr != -1:
        model_settings[0] = args.lr
    model_settings.insert(0, args.seed)
    model_settings = tuple(model_settings)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_client_managers = torch.cuda.device_count()
    print(f"Spawning {num_client_managers} Client Managers using {device} device")
    client_managers = setup_client_managers(num_client_managers, seed=args.seed, device=device)

    # Create client model, and share params with server model
    client_model = ClientModel(*model_settings)

    # Create server
    server = Server(model_params=client_model.state_dict(), client_managers=client_managers)

    # Create clients
    clients = setup_clients(
        client_managers=client_managers,
        dataset=args.dataset,
        model=ClientModel,
        model_settings=model_settings,
        use_val_set=args.use_val_set
    )
    client_ids, client_groups, client_num_samples = server.get_clients_info()
    print(f"Clients in Total: {len(clients)}")

    # Initial status
    print("--- Random Initialization ---")
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    # Simulate training
    for i in range(num_rounds):
        print(f"--- Round {i + 1} of {num_rounds}: Training {clients_per_round} Clients ---")

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)

        # Simulate server model training on selected clients" data
        server.train_clients(num_epochs=args.num_epochs, batch_size=args.batch_size)
        
        # Update server model
        server._update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
    
    end_time = datetime.now()

    print(SECTION_STR.format("Post-Simulation"))
    print(f"Total Simulation time: {end_time - start_time}")
    # Save server model
    ckpt_path = os.path.join("checkpoints", args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, "{}.ckpt".format(args.model)))
    print(f"Model saved in path: {save_path}")

def online(clients: list) -> list:
    """We assume all users are always online."""
    return clients

def setup_client_managers(num_client_managers: int, seed: float, device: str = "cpu"):
    if num_client_managers < 1:
        return [ClientManager.remote(id=0)]

    client_managers = []
    
    for id in range(num_client_managers):
        client_managers.append(ClientManager.remote(id=id, seed=seed, device=device))
    
    return client_managers

def create_clients(
    client_managers: list,
    users: list,
    groups: list,
    train_data: dict,
    eval_data: dict, 
    model: type,
    model_settings: tuple
) -> list:
    if len(groups) == 0:
        groups = [[] for _ in users]

    num_client_managers = len(client_managers)

    client_creation_futures = []
    for client_num, (u, g) in enumerate(zip(users, groups)):
        future = client_managers[client_num % num_client_managers].add_client.remote(
            client_num=client_num,
            client_id=u,
            train_data=train_data[u],
            eval_data=eval_data[u],
            model=model,
            model_settings=model_settings,
            group=g
        )
        client_creation_futures.append(future)
    
    clients = []
    while len(client_creation_futures) > 0:
        complete, incomplete = ray.wait(client_creation_futures)

        for client_num in ray.get(complete):
            clients.append(client_num)
        client_creation_futures = incomplete
    
    clients.sort()

    return clients


def setup_clients(
    client_managers: list,
    dataset: str,
    model: type,
    model_settings: tuple,
    use_val_set: bool = False
) -> list:
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = "test" if not use_val_set else "val"
    train_data_dir = os.path.join("..", "data", dataset, "data", "train")
    test_data_dir = os.path.join("..", "data", dataset, "data", eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(
        client_managers=client_managers,
        users=users,
        groups=groups,
        train_data=train_data,
        eval_data=test_data,
        model=model,
        model_settings=model_settings
    )

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, "{}_{}".format(args.metrics_name, "stat"))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, "train", args.metrics_dir, "{}_{}".format(args.metrics_name, "sys"))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.eval_model(set_to_use="train")
    print_metrics(train_stat_metrics, num_samples, prefix="train_")
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.eval_model(set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix="{}_".format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=""):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print("%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g" \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == "__main__":
    main()
