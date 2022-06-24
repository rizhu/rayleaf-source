import importlib
import os
import sys

from pathlib import Path


import ray


import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils
from rayleaf.core.client_manager import make_client_manager
from rayleaf.utils.data_utils import read_data
from rayleaf.entities.client import Client
from rayleaf.entities.server import Server
from rayleaf.models.model_constants import MODEL_SETTINGS


def initialize_resources(
    dataset: str,
    output_dir: str,
    model: str,
    client_lr: float,
    seed: int = 0,
):
    experiment_log = open(Path(output_dir, "log.txt"), mode="w+")
    logging_utils.logging_file = experiment_log

    assert dataset.lower() in utils.DATASETS, f"Dataset \"{dataset}\" is not a valid dataset. Available datasets are {utils.DATASETS}"
    dataset = dataset.lower()

    model_path = f"{dataset}.{model}"
    try:
        mod = importlib.import_module(f"rayleaf.models.{model_path}")
        ClientModel = getattr(mod, "ClientModel")
        model_settings = dict(MODEL_SETTINGS[model_path])
        model_settings["seed"] = seed
        model_settings["lr"] = client_lr
    except ModuleNotFoundError:
        logging_utils.log(f"Please specify a valid dataset and model.")
        logging_utils.logging_file = experiment_log = None
        sys.exit()

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        os.makedirs(output_dir, exist_ok=True)
    
    logging_utils.log(utils.SECTION_STR.format(model_path))

    return ClientModel, model_settings


def create_entities(
    device: str,
    num_client_managers: int,
    gpus_per_client_manager: float,
    seed: float,
    model_settings: dict,
    ClientModel: type,
    ServerType: type,
    client_types: list,
    dataset: str,
    dataset_dir: str,
    use_val_set: bool
):
    logging_utils.log(f"Spawning {num_client_managers} Client Managers using {device} device")
    client_managers = setup_client_managers(
        num_client_managers=num_client_managers,
        gpus_per_client_manager=gpus_per_client_manager,
        seed=seed,
        device=device
    )

    # Create client model, and share params with server model
    client_model = ClientModel(**model_settings)

    # Create server
    verify_server_input(ServerType=ServerType)
    server = ServerType(model_params=client_model.state_dict(), client_managers=client_managers)

    # Create clients
    clients = setup_clients(
        clients=client_types,
        client_managers=client_managers,
        dataset=dataset,
        dataset_dir=dataset_dir,
        model=ClientModel,
        model_settings=model_settings,
        use_val_set=use_val_set
    )

    return server, clients


def setup_client_managers(num_client_managers: int, gpus_per_client_manager: int, seed: float, device: str = "cpu"):
    ClientManager = make_client_manager(num_gpus=gpus_per_client_manager)

    if num_client_managers < 1:
        return [ClientManager.remote(id=0)]

    client_managers = []
    
    for id in range(num_client_managers):
        client_managers.append(ClientManager.remote(id=id, seed=seed, device=device))
    
    return client_managers


def setup_clients(
    clients: list,
    client_managers: list,
    dataset: str,
    dataset_dir: str,
    model: type,
    model_settings: tuple,
    use_val_set: bool = False
) -> list:
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = "test" if not use_val_set else "val"
    train_data_dir = Path(dataset_dir, "data", "train")
    test_data_dir = Path(dataset_dir, "data", eval_set)

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
