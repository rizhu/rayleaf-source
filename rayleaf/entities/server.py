from collections import (
    OrderedDict
)

import numpy as np
import ray
import torch

class Server:
    
    def __init__(self, model_params: OrderedDict, client_managers: list) -> None:
        self.model_params = model_params
        for param_tensor, layer in self.model_params.items():
            self.model_params[param_tensor] = layer.cuda()
        self.reset_grads()

        self.client_managers = client_managers
        self.num_client_managers = len(client_managers)

        self.updates = []
        self.selected_clients = [[] for _ in range(self.num_client_managers)]

        self.init()


    def init(self) -> None:
        pass


    def select_clients(self, my_round: int, possible_clients: list, num_clients: int = 20) -> None:
        selected_client_nums = np.random.choice(possible_clients, num_clients, replace=False)
        self.selected_clients = [[] for _ in range(self.num_client_managers)]

        for client_num in selected_client_nums:
            self.selected_clients[client_num % self.num_client_managers].append(client_num)
        
        return selected_client_nums


    def train_clients(self, num_epochs: int = 1, batch_size: int = 10) -> None:
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


    def update_model(self) -> None:
        self.reset_model()

        total_weight = 0
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples

            for param_tensor, layer in client_model.items():
                self.model_params[param_tensor] += client_samples * layer

        for param_tensor in self.model_params.keys():
            self.model_params[param_tensor] /= total_weight


    @torch.no_grad()
    def _update_model(self) -> None:
        self.update_model()

        self.updates.clear()


    def reset_model(self) -> None:
        new_model_params = OrderedDict()

        for param_tensor in self.model_params.keys():
            new_model_params[param_tensor] = 0
        
        self.model_params = new_model_params
    

    def reset_grads(self) -> None:
        new_grads = OrderedDict()

        for param_tensor, layer in self.model_params.items():
            new_grads[param_tensor] = torch.zeros(layer.shape).cuda()
        
        self.grads = new_grads


    def eval_model(self, eval_all_clients: bool = True, set_to_use: str = "test", batch_size: int = 10) -> dict:
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
        torch.save({"model_params": self.model_params}, path)


    @property
    def model_params(self) -> OrderedDict:
        return self._model_params

    
    @model_params.setter
    def model_params(self, params: OrderedDict) -> None:
        self._model_params = params


    @property
    def grads(self) -> OrderedDict:
        return self._grads

    
    @grads.setter
    def grads(self, grads: OrderedDict) -> None:
        self._grads = grads