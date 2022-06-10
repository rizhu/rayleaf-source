from collections import (
    OrderedDict
)

import numpy as np
import ray
import torch

class Server:
    
    def __init__(self, model_params: OrderedDict, client_managers: list) -> None:
        self.model_params = model_params

        self.client_managers = client_managers
        self.num_client_managers = len(client_managers)

        self.updates = []
        self.selected_clients = [[] for _ in range(self.num_client_managers)]

    def select_clients(self, my_round: int, possible_clients: list, num_clients: int = 20) -> None:
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        self.selected_clients = [[] for _ in range(self.num_client_managers)]

        for client_num in selected_clients:
            self.selected_clients[client_num % self.num_client_managers].append(client_num)

    def train_model(self, num_epochs: int = 1, batch_size: int = 10) -> None:
        """Trains self.model_params on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client"s data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        training_futures = []
        for client_manager_idx, manager_clients in enumerate(self.selected_clients):
            if len(manager_clients) > 0:
                training_future = self.client_managers[client_manager_idx].train_clients.remote(
                    model_params=self.model_params,
                    selected_clients=manager_clients,
                    num_epochs=num_epochs,
                    batch_size=batch_size
                )
                training_futures.append(training_future)
        
        while len(training_futures) > 0:
            complete, incomplete = ray.wait(training_futures)

            for result in ray.get(complete):
                self.updates.extend(result)

            training_futures = incomplete


    @torch.no_grad()
    def update_model(self) -> None:
        new_model = OrderedDict()
        for param_tensor in self.model_params.keys():
            new_model[param_tensor] = 0

        total_weight = 0
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples

            for param_tensor, layer in client_model.items():
                new_model[param_tensor] += client_samples * layer

        for param_tensor in new_model.keys():
            new_model[param_tensor] /= total_weight

        self.model_params = new_model
        self.updates = []

    def eval_model(self, eval_all_clients: bool = True, set_to_use: str = "test", batch_size: int = 10) -> dict:
        """Tests self.model_params on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ["train", "test"].
        """
        metrics = {}

        eval_futures = []
        if eval_all_clients:
            for client_manager in self.client_managers:
                eval_future = client_manager.eval_model.remote(
                    model_params=self.model_params,
                    set_to_use=set_to_use
                )
                eval_futures.append(eval_future)
        else:
            for client_manager_idx, manager_clients in enumerate(self.selected_clients):
                if len(manager_clients) > 0:
                    eval_future = self.client_managers[client_manager_idx].eval_model.remote(
                        model_params=self.model_params,
                        set_to_use=set_to_use,
                        selected_clients=manager_clients,
                        batch_size=batch_size
                    )
                    eval_futures.append(eval_future)

        while len(eval_futures) > 0:
            complete, incomplete = ray.wait(eval_futures)

            for manager_metrics in ray.get(complete):
                metrics.update(manager_metrics)

            eval_futures = incomplete
        
        return metrics

    def get_clients_info(self) -> tuple:
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        info_futures = []
        for client_manager in self.client_managers:
            info_future = client_manager.get_clients_info.remote()
            info_futures.append(info_future)

        ids = []
        groups = {}
        num_samples = {}
        while len(info_futures) > 0:
            complete, incomplete = ray.wait(info_futures)

            for future_ids, future_groups, future_num_samples in ray.get(complete):
                ids.extend(future_ids)
                groups.update(future_groups)
                num_samples.update(future_num_samples)

            info_futures = incomplete
        
        return ids, groups, num_samples


    def save_model(self, path: str) -> None:
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        torch.save({"model_params": self.model_params}, path)