import random

from collections import OrderedDict


import numpy as np
import ray
import torch


def make_client_cluster(num_gpus: float) -> type:

    @ray.remote(num_gpus=num_gpus)
    class ClientCluster:
        def __init__(self, id: int, seed: float, device: str = "cpu"):
            self.seed = seed
            self.set_seeds(seed)

            self.id = id
            self.device = device
            self.clients = {}


        def add_client(
            self,
            ClientType: type,
            client_num: int,
            client_id: str,
            train_data: dict,
            eval_data: dict,
            model: type,
            model_settings: tuple,
            group: list = None
        ) -> None:
            self.clients[client_num] = ClientType(
                client_num,
                client_id,
                train_data,
                eval_data,
                model,
                model_settings,
                group,
                device=self.device
            )

            return client_num, ClientType

        
        def train_clients(
            self,
            model_params: OrderedDict,
            selected_clients: list = None,
            num_epochs: int = 1,
            batch_size: int = 10
        ) -> list:

            clients_to_train = self.get_clients_from_client_nums(selected_clients)
            
            updates = []

            for client in clients_to_train:
                client.model_params = model_params
                num_samples, update = client._train(num_epochs, batch_size)

                updates.append((num_samples, update))
            
            return updates


        def eval_model(
            self,
            model_params: OrderedDict,
            selected_clients: list = None,
            set_to_use: str = "test",
            batch_size: int = 10
        ) -> dict:
            clients_to_eval = self.get_clients_from_client_nums(selected_clients)

            stats = []

            for client in clients_to_eval:
                client.model_params = model_params
                client_stats = client._eval(set_to_use, batch_size)
                stats.append(client_stats)
            
            return stats


        def get_clients_info(self, client_nums: list = None) -> tuple:
            clients = self.get_clients_from_client_nums(client_nums)

            ids = []
            groups = {}
            num_samples = {}
            for client in clients:
                ids.append(client.id)
                groups[client.id] = client.group
                num_samples[client.id] = client.num_samples

            return ids, groups, num_samples


        def get_clients_from_client_nums(self, client_nums: list) -> list:
            if client_nums is None:
                return list(self.clients.values())

            clients = []

            for client_num in client_nums:
                clients.append(self.clients[client_num])

            return clients


        def set_seeds(self, seed: float = 0):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
    
    return ClientCluster
