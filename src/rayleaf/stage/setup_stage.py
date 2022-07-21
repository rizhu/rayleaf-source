import importlib
import os

from pathlib import Path


import ray

from tqdm import tqdm


import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils
import rayleaf.models.speech_commands.initialize_data as sc_initialize_data

from rayleaf.core.client_cluster import make_client_cluster
from rayleaf.utils.data_utils import read_data
from rayleaf.entities.client import Client
from rayleaf.entities.server import Server
from rayleaf.models.model_constants import MODEL_SETTINGS


def log_experiment_details(**kwargs):
    logging_utils.log(utils.SECTION_STR.format("Experiment details"))
    for k, v in kwargs.items():
        if v is not None:
            logging_utils.log(f"{str(k)}: {str(v)}")


def initialize_resources(
    dataset_input: str,
    output_dir: str,
    model: str,
    client_lr: float,
    seed: int = 0,
):
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        os.makedirs(output_dir, exist_ok=True)
        
    experiment_log = open(Path(output_dir, "log.txt"), mode="w+")
    logging_utils.logging_file = experiment_log

    dataset = verify_dataset(dataset_input)
    dataset_input = dataset_input.lower()

    model_path = f"{dataset}.{model}"
    try:
        mod = importlib.import_module(f"rayleaf.models.{model_path}")
        ClientModel = getattr(mod, "ClientModel")
        model_settings = dict(MODEL_SETTINGS[model_path])
        model_settings["seed"] = seed
        model_settings["lr"] = client_lr
    except ModuleNotFoundError as e:
        logging_utils.log(f"Dataset {dataset} and model {model} is not a valid dataset-model pair.")
        logging_utils.shutdown_log()
        raise e

    return ClientModel, model_settings


def create_entities(
    device: str,
    num_client_clusters: int,
    gpus_per_client_cluster: float,
    seed: float,
    model_settings: dict,
    ClientModel: type,
    ServerType: type,
    client_types: list,
    dataset: str,
    dataset_dir: str,
    use_val_set: bool
):
    logging_utils.log(f"Spawning {num_client_clusters} ClientClusters using {device} device (this may take a while)")
    client_clusters = setup_client_clusters(
        num_client_clusters=num_client_clusters,
        gpus_per_client_cluster=gpus_per_client_cluster,
        seed=seed,
        device=device
    )

    # Create client model, and share params with server model
    client_model = ClientModel(**model_settings)

    # Create server
    verify_server_input(ServerType=ServerType)
    server = ServerType(model_params=list(client_model.parameters()), client_clusters=client_clusters)

    # Create clients
    clients = setup_clients(
        clients=client_types,
        client_clusters=client_clusters,
        dataset=dataset,
        dataset_dir=dataset_dir,
        model=ClientModel,
        model_settings=model_settings,
        seed=seed,
        use_val_set=use_val_set
    )

    return server, clients


def setup_client_clusters(num_client_clusters: int, gpus_per_client_cluster: int, seed: float, device: str = "cpu"):
    ClientCluster = make_client_cluster(num_gpus=gpus_per_client_cluster)

    if num_client_clusters < 1:
        return [ClientCluster.remote(id=0)]

    client_clusters = []
    
    for id in range(num_client_clusters):
        client_clusters.append(ClientCluster.remote(id=id, seed=seed, device=device))
    
    return client_clusters


def setup_clients(
    clients: list,
    client_clusters: list,
    dataset: str,
    dataset_dir: str,
    model: type,
    model_settings: tuple,
    seed: float,
    use_val_set: bool = False,
) -> list:
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    dataset_dir = Path(dataset_dir)

    eval_set = "test" if not use_val_set else "val"
    train_data_dir = Path(dataset_dir, "data", "train")
    test_data_dir = Path(dataset_dir, "data", eval_set)

    if "speech_commands" in dataset:
        num_clients = verify_clients_input(clients=clients, max_num_clients=sc_initialize_data.MAX_NUMBER_OF_CLIENTS, dataset="speech_commands")
        users, groups, train_data, test_data = sc_initialize_data.federate_dataset(root=dataset_dir, num_clients=num_clients, seed=seed, iid="iid" in dataset)
    else:
        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        verify_clients_input(clients=clients, max_num_clients=len(users), dataset=dataset)

    clients = create_clients(
        clients=clients,
        client_clusters=client_clusters,
        users=users,
        groups=groups,
        dataset_dir=dataset_dir,
        train_data=train_data,
        eval_data=test_data,
        model=model,
        model_settings=model_settings
    )

    return clients


def create_clients(
    clients: list,
    client_clusters: list,
    users: list,
    groups: list,
    dataset_dir: Path,
    train_data: dict,
    eval_data: dict, 
    model: type,
    model_settings: tuple
) -> list:
    if not groups or len(groups) == 0:
        groups = [[] for _ in users]

    num_client_clusters = len(client_clusters)

    client_creation_futures = []

    user_group_pairs = zip(users, groups)
    client_num = 0

    for ClientType, count in clients:
        for _ in range(count):
            u, g = next(user_group_pairs)
            future = client_clusters[client_num % num_client_clusters].add_client.remote(
                ClientType=ClientType,
                client_num=client_num,
                client_id=u,
                dataset_dir=dataset_dir,
                train_data=train_data[u],
                eval_data=eval_data[u],
                model=model,
                model_settings=model_settings,
                group=g
            )
            client_creation_futures.append(future)

            client_num += 1
    
    num_futures = len(client_creation_futures)
    clients = {}
    with tqdm(total=num_futures, leave=False, desc="Creating clients") as pbar:
        while len(client_creation_futures) > 0:
            complete, incomplete = ray.wait(client_creation_futures)

            for client_num, ClientType in ray.get(complete):
                clients[client_num] = ClientType
                pbar.update(1)

            client_creation_futures = incomplete

    return clients


def verify_dataset(dataset: str):
    for valid_dataset in utils.DATASETS:
        if valid_dataset in dataset:
            return valid_dataset

    raise ValueError(f"Dataset {dataset} is not a valid dataset.")


def verify_server_input(ServerType: type):
    assert isinstance(ServerType, type), f"ServerType {ServerType} is not a class."
    assert issubclass(ServerType, Server), f"ServerType {ServerType} is not a subclass of Server."


def verify_clients_input(clients: list, max_num_clients: int, dataset: str):
    prompt = "Each element in clients list must be a length-2 tuple of the form (Client subclass: type, count: int)."
    total = 0

    num_client_types = len(clients)
    for idx, pair in enumerate(clients):

        assert len(pair) == 2, f"{prompt} Got {pair} at index {idx} of clients list."

        assert isinstance(pair[0], type), f"{prompt} First element at index {idx} of clients list has type {type(pair[0])}."
        assert issubclass(pair[0], Client), f"{prompt} First element at index {idx} of clients list is not a subclass of Client."

        assert isinstance(pair[1], int), f"{prompt} Second element at index {idx} or clients list has type {type(pair[1])}."

        if idx < num_client_types - 1:
            assert pair[1] > 0, f"Each Client count except for last one must be greater than 0. Got count of {pair[1]} at index {idx} of clients list."
        else:
            assert pair[1] == -1 or pair[1] > 0, f"Last Client count must be greater than 0 or be -1. Got count of {pair[1]} at index {idx} of clients list."
            if pair[1] == -1:
                new_pair = (pair[0], max_num_clients - total)
                clients[idx] = new_pair
        
        total += pair[1]
    
    assert total <= max_num_clients, f"Max number of clients for {dataset} dataset is {max_num_clients} but clients list defines {total} clients."

    return total
