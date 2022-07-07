from collections import OrderedDict


import numpy as np
import ray
import torch

from tqdm import tqdm


class Server:
    
    def __init__(self, model_params: OrderedDict, client_clusters: list) -> None:
        self.model_params = model_params
        self.reset_grads()

        self.client_clusters = client_clusters
        self.num_client_clusters = len(client_clusters)

        self.updates = []
        self.selected_clients = [[] for _ in range(self.num_client_clusters)]

        self.init()


    def init(self) -> None:
        pass


    def select_clients(self, my_round: int, possible_clients: list, num_clients: int = 20) -> None:
        selected_client_nums = np.random.choice(possible_clients, num_clients, replace=False)
        self.selected_clients = [[] for _ in range(self.num_client_clusters)]

        for client_num in selected_client_nums:
            self.selected_clients[client_num % self.num_client_clusters].append(client_num)
        
        return selected_client_nums


    def train_clients(self, num_epochs: int = 1, batch_size: int = 10) -> None:
        training_futures = []
        for client_cluster_idx, cluster_clients in enumerate(self.selected_clients):
            if len(cluster_clients) > 0:
                training_future = self.client_clusters[client_cluster_idx].train_clients.remote(
                    model_params=self.model_params,
                    selected_clients=cluster_clients,
                    num_epochs=num_epochs,
                    batch_size=batch_size
                )
                training_futures.append(training_future)
        
        num_futures = len(training_futures)
        with tqdm(total=num_futures, leave=False, desc="Training clients") as pbar:
            while len(training_futures) > 0:
                complete, incomplete = ray.wait(training_futures)

                for result in ray.get(complete):
                    self.updates.extend(result)
                    pbar.update(1)

                training_futures = incomplete


    def update_model(self) -> None:
        self.reset_model()

        total_weight = 0
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples

            for param_tensor, layer in client_model.items():
                self.model_params[param_tensor] += client_samples * layer

        for param_tensor in self.model_params.keys():
            self.model_params[param_tensor] = (self.model_params[param_tensor] / total_weight).to(self.model_params[param_tensor].dtype)

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
            new_grads[param_tensor] = torch.zeros(layer.shape)
        
        self.grads = new_grads


    def eval_model(self, eval_all_clients: bool = True, set_to_use: str = "test", batch_size: int = 10) -> dict:
        eval_futures = []
        if eval_all_clients:
            for client_cluster in self.client_clusters:
                eval_future = client_cluster.eval_model.remote(
                    model_params=self.model_params,
                    set_to_use=set_to_use,
                    batch_size=batch_size 
                )
                eval_futures.append(eval_future)
        else:
            for client_cluster_idx, cluster_clients in enumerate(self.selected_clients):
                if len(cluster_clients) > 0:
                    eval_future = self.client_clusters[client_cluster_idx].eval_model.remote(
                        model_params=self.model_params,
                        set_to_use=set_to_use,
                        selected_clients=cluster_clients,
                        batch_size=batch_size
                    )
                    eval_futures.append(eval_future)

        stats = []

        num_futures = len(eval_futures)
        with tqdm(total=num_futures, leave=False, desc="Evaluating model") as pbar:
            while len(eval_futures) > 0:
                complete, incomplete = ray.wait(eval_futures)

                for cluster_metrics in ray.get(complete):
                    stats.extend(cluster_metrics)
                    pbar.update(1)

                eval_futures = incomplete
        
        return stats


    def get_clients_info(self) -> tuple:
        info_futures = []
        for client_cluster in self.client_clusters:
            info_future = client_cluster.get_clients_info.remote()
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
