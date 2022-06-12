from datetime import (
    datetime
)
import importlib
from pathlib import (
    Path
)
import random

import numpy as np
import ray
import torch

import metrics.writer as metrics_writer

from baseline_constants import (
    MODEL_SETTINGS
)
from client import (
    Client
)
from client_manager import (
    ClientManager
)
from server import (
    Server
)

from utils.data_utils import (
    read_data
)

CWD = Path().resolve()
STAT_METRICS_PATH = Path("metrics", "stat_metrics.csv")
SYS_METRICS_PATH = Path("metrics", "sys_metrics.csv")

DATASETS = {"sent140", "femnist", "shakespeare", "celeba", "synthetic", "reddit"}

SECTION_STR = "\n############################## {} ##############################"


def run_experiment(
    dataset: str,
    model: str,
    num_rounds: int,
    eval_every: int,
    ServerType: type,
    client_types: list,
    clients_per_round: int,
    client_lr: float,
    batch_size: int = 10,
    seed: int = 0,
    metrics_name: str = "metrics",
    metrics_dir: str = "metrics",
    use_val_set: bool = False,
    num_epochs: int = 1
) -> None:
    start_time = datetime.now()

    assert dataset.lower() in DATASETS, f"Dataset \"{dataset}\" is not a valid dataset. Available datasets are {DATASETS}"
    dataset = dataset.lower()
    verify_server_input(ServerType=ServerType)

    ray.init()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_path = Path(dataset, f"{model}.py")
    if not model_path.is_file():
        print("Please specify a valid dataset and a valid model.")
    model_path = f"{dataset}.{model}"
    
    print(SECTION_STR.format(model_path))
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, "ClientModel")

    # Create 2 models
    model_settings = MODEL_SETTINGS[model_path]
    model_settings[0] = client_lr
    model_settings.insert(0, seed)
    model_settings = tuple(model_settings)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_client_managers = torch.cuda.device_count()
    print(f"Spawning {num_client_managers} Client Managers using {device} device")
    client_managers = setup_client_managers(num_client_managers, seed=seed, device=device)

    # Create client model, and share params with server model
    client_model = ClientModel(*model_settings)

    # Create server
    server = ServerType(model_params=client_model.state_dict(), client_managers=client_managers)

    # Create clients
    clients = setup_clients(
        clients=client_types,
        client_managers=client_managers,
        dataset=dataset,
        model=ClientModel,
        model_settings=model_settings,
        use_val_set=use_val_set
    )
    client_ids, client_groups, client_num_samples = server.get_clients_info()
    client_counts = count_selected_clients(online(clients), clients)
    print(f"{len(clients)} total clients: {client_counts_string(client_counts)}")

    # Initial status
    print("--- Random Initialization ---")
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, metrics_name, metrics_dir)
    sys_writer_fn = get_sys_writer_function(metrics_name, metrics_dir)
    print_stats(0, server, client_num_samples, stat_writer_fn, use_val_set)

    # Simulate training
    for i in range(num_rounds):
        # Select clients to train this round
        selected_clients = server.select_clients(i, online(clients), num_clients=clients_per_round)
        selected_client_counts = count_selected_clients(selected_clients, clients)
        print(f"--- Round {i + 1} of {num_rounds}: Training {clients_per_round} clients: {client_counts_string(selected_client_counts)} ---")

        # Simulate server model training on selected clients" data
        server.train_clients(num_epochs=num_epochs, batch_size=batch_size)
        
        # Update server model
        server._update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, client_num_samples, stat_writer_fn, use_val_set)
    
    end_time = datetime.now()

    print(SECTION_STR.format("Post-Simulation"))
    print(f"Total Simulation time: {end_time - start_time}")
    # Save server model
    ckpt_path = Path("checkpoints", dataset)
    if not ckpt_path.is_dir():
        ckpt_path.mkdir()
    save_path = server.save_model(Path(ckpt_path, f"{model}.ckpt"))
    print(f"Model saved in path: {save_path}")


def online(clients: list) -> list:
    """We assume all users are always online."""
    return sorted(clients.keys())


def setup_client_managers(num_client_managers: int, seed: float, device: str = "cpu"):
    if num_client_managers < 1:
        return [ClientManager.remote(id=0)]

    client_managers = []
    
    for id in range(num_client_managers):
        client_managers.append(ClientManager.remote(id=id, seed=seed, device=device))
    
    return client_managers


def verify_server_input(ServerType: type):
    assert isinstance(ServerType, type), f"ServerType {ServerType} is not a class."
    assert issubclass(ServerType, Server), f"ServerType {ServerType} is not a subclass of Server."


def verify_clients_input(clients: list, max_num_clients: int, dataset: str):
    prompt = "Each element in clients list must be a length-2 tuple of the form (Client subclass: type, count: int)."
    total = 0

    num_clients = len(clients)
    for idx, pair in enumerate(clients):

        assert len(pair) == 2, f"{prompt} Got {pair} at index {idx} of clients list."

        assert isinstance(pair[0], type), f"{prompt} First element at index {idx} of clients list has type {type(pair[0])}."
        assert issubclass(pair[0], Client), f"{prompt} First element at index {idx} of clients list is not a subclass of Client."

        assert isinstance(pair[1], int), f"{prompt} Second element at index {idx} or clients list has type {type(pair[1])}."

        if idx < num_clients - 1:
            assert pair[1] > 0, f"Each Client count except for last one must be greater than 0. Got count of {pair[1]} at index {idx} of clients list."
        else:
            assert pair[1] == -1 or pair[1] > 0, f"Last Client count must be greater than 0 or be -1. Got count of {pair[1]} at index {idx} of clients list."
            if pair[1] == -1:
                new_pair = (pair[0], max_num_clients - total)
                clients[idx] = new_pair
        
        total += pair[1]
    
    assert total <= max_num_clients, f"Max number of clients for {dataset} dataset is {max_num_clients} but clients list defines {total} clients."


def create_clients(
    clients: list,
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

    user_group_pairs = zip(users, groups)
    client_num = 0

    for ClientType, count in clients:
        for _ in range(count):
            u, g = next(user_group_pairs)
            future = client_managers[client_num % num_client_managers].add_client.remote(
                ClientType=ClientType,
                client_num=client_num,
                client_id=u,
                train_data=train_data[u],
                eval_data=eval_data[u],
                model=model,
                model_settings=model_settings,
                group=g
            )
            client_creation_futures.append(future)

            client_num += 1
    
    clients = {}
    while len(client_creation_futures) > 0:
        complete, incomplete = ray.wait(client_creation_futures)

        for client_num, ClientType in ray.get(complete):
            clients[client_num] = ClientType
        client_creation_futures = incomplete

    return clients


def setup_clients(
    clients: list,
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
    train_data_dir = Path(Path("..").resolve(), "data", dataset, "data", "train")
    test_data_dir = Path(Path("..").resolve(), "data", dataset, "data", eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    verify_clients_input(clients=clients, max_num_clients=len(users), dataset=dataset)

    clients = create_clients(
        clients=clients,
        client_managers=client_managers,
        users=users,
        groups=groups,
        train_data=train_data,
        eval_data=test_data,
        model=model,
        model_settings=model_settings
    )

    return clients


def count_selected_clients(selected_clients: list, clients: dict) -> dict:
    counts = {}
    for client_num in selected_clients:
        client_type = clients[client_num]
        counts[client_type] = counts.get(client_type, 0) + 1

    return counts


def client_counts_string(counts: dict) -> str:
    client_types = list(counts.keys())
    client_types.sort(key=lambda ClientType: ClientType.__name__)

    counts_str = ""
    count_format = "{count} {client_type_name}"

    if len(client_types) == 1:
        counts_str += count_format.format(count=counts[client_types[0]], client_type_name=client_types[0].__name__)
        if counts[client_types[0]] != 1:
            counts_str += "s"
    elif len(client_types) == 2:
        counts_str += count_format.format(count=counts[client_types[0]], client_type_name=client_types[0].__name__)
        if counts[client_types[0]] != 1:
            counts_str += "s"

        counts_str += " and "

        counts_str += count_format.format(count=counts[client_types[1]], client_type_name=client_types[1].__name__)
        if counts[client_types[1]] != 1:
            counts_str += "s"
    else:
        for client_type in client_types[:-1]:
            counts_str += count_format.format(count=counts[client_type], client_type_name=client_type.__name__)
            if counts[client_type] != 1:
                counts_str += "s"
            counts_str += ", "
        
        counts_str += "and "
        counts_str += count_format.format(count=counts[client_types[-1]], client_type_name=client_types[-1].__name__)
        if counts[client_types[-1]] != 1:
            counts_str += "s"
    
    return counts_str
        

def get_stat_writer_function(ids, groups, num_samples, metrics_name: str, metrics_dir: str):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, Path(CWD, metrics_dir), "{}_{}".format(metrics_name, "stat"))

    return writer_fn


def get_sys_writer_function(metrics_name: str, metrics_dir: str):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, "train", Path(CWD, metrics_dir), "{}_{}".format(metrics_name, "sys"))

    return writer_fn


def print_stats(
    num_round, server, num_samples, writer, use_val_set):
    
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
