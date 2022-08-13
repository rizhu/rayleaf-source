import numpy as np
import ray
import torch

from tqdm import tqdm


import rayleaf.entities.constants as constants


class Server:
    
    def __init__(self, model_params: list, client_clusters: list) -> None:
        self.model_params = model_params
        self.layer_shapes = [layer.shape for layer in self.model_params]

        self.client_clusters = client_clusters
        self.num_client_clusters = len(client_clusters) 

        self.updates = []
        self.selected_clients = [[] for _ in range(self.num_client_clusters)]

        self.clients_profiled = set()
        self.client_flops = []

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
        
        return self.updates


    def update_layer(self, current_params, updates: list, client_num_samples: list, num_clients: int):
        new_layer = 0

        for i in range(num_clients):
            new_layer += updates[i] * client_num_samples[i]
        
        return new_layer / self.num_train_samples


    def update_model(self) -> None:
        num_clients = len(self.updates)

        client_num_samples = []
        for update in self.updates:
            client_num_samples.append(update[constants.NUM_SAMPLES_KEY])

        for i, layer in enumerate(self.model_params):
            layer_updates = []
            for update in self.updates:
                layer_updates.append(update[constants.UPDATE_KEY][i])
            
            self.model_params[i] = self.update_layer(
                self.model_params[i].clone().detach(),
                layer_updates,
                client_num_samples.copy(),
                num_clients
            ).to(layer.dtype)


    @torch.no_grad()
    def _update_model(self) -> None:
        self.num_train_samples = 0
        for update in self.updates:
            self.num_train_samples += update[constants.NUM_SAMPLES_KEY]

        self.update_model()
        self.updates.clear()


    def reset_model(self) -> None:
        new_model_params = [0 for _ in range(len(self.model_params))]
        
        self.model_params = new_model_params


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
    def model_params(self) -> list:
        return self._model_params

    
    @model_params.setter
    def model_params(self, params: list) -> None:
        self._model_params = params


    def param_like_zeros(self, dtype = torch.float32) -> list:
        res = []
        for shape in self.layer_shapes:
            res.append(torch.zeros(shape))
        
        return res.to(dtype)
    

    def param_like_const(self, value: float = 1, dtype = torch.float32) -> list:
        res = []

        for shape in self.layer_shapes:
            res.append(torch.ones(shape) * value)
        
        return res.to(dtype)


    def param_like_ones(self, dtype = torch.float32) -> list:
        return self.param_like_const(dtype=dtype)
    

    def param_like_normal(self, mean: float = 0, std: float = 1, dtype = torch.float32) -> list:
        res = []

        for shape in self.layer_shapes:
            res.append(torch.normal(mean=torch.ones(shape) * mean, std=torch.ones(shape) * std))
        
        return res.to(dtype)
