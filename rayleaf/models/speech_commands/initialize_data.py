import json
import os

from pathlib import Path
from tqdm import tqdm


import torch

from torchaudio.datasets import SPEECHCOMMANDS


MAX_NUMBER_OF_CLIENTS = 2000


class SpeakerSC(SPEECHCOMMANDS):
    def __init__(self, root: Path, speaker: str):
        super().__init__(root=root, download=True)
        
        self._walker = [w for w in self._walker if speaker in w]


def federate_dataset(root, seed, num_clients=-1, train_split=0.8, iid=False):
    assert -1 <= num_clients <= MAX_NUMBER_OF_CLIENTS, f"num_clients must be an integer between -1 and {MAX_NUMBER_OF_CLIENTS} for speech_commands dataset"

    if num_clients == -1:
        num_clients = MAX_NUMBER_OF_CLIENTS

    root = Path(root)
    if not root.is_dir():
        os.makedirs(root, exist_ok=True)

    if not iid:
        return federate_dataset_realistic(root, seed, num_clients, train_split)
    else:
        return federate_dataset_iid(root, seed, num_clients, train_split)


def federate_dataset_realistic(root: Path, seed: float, num_clients: int, train_split: float):
    speaker_data_size_json = Path(root, "speaker_data_size.json")
        
    if not speaker_data_size_json.is_file():
        data = SPEECHCOMMANDS(
            root=root,
            download=True
        )
        
        speakers = set()

        for d in tqdm(data, desc="Collecting speaker IDs", leave=False):
            speakers.add(d[3])
        
        speakers = list(speakers)
        
        speaker_data_size = {}
        speaker_dataset = {}
        
        for speaker in tqdm(speakers, desc="Counting speaker dataset sizes", leave=False):
            data = SpeakerSC(
                root=root,
                speaker=speaker
            )
            speaker_data_size[speaker] = len(data)
            speaker_dataset[speaker] = data
        
        with open(speaker_data_size_json, mode="w+") as fp:
            json.dump(speaker_data_size, fp)
        
        speakers.sort(key=lambda speaker: speaker_data_size[speaker], reverse=True)
        
        datasets = {}
        for speaker in tqdm(speakers[:num_clients], "Federating dataset", leave=False):
            datasets[speaker] = speaker_dataset[speaker]
    else:
        with open(speaker_data_size_json, "r") as fp:
            print("Loading speaker to dataset count json...")
            speaker_data_size = json.load(fp)
        
        speakers = sorted(list(speaker_data_size.keys()), key=lambda speaker: speaker_data_size[speaker], reverse=True)
            
        datasets = {}
        for speaker in tqdm(speakers[:num_clients], desc="Federating dataset", leave=False):
            datasets[speaker] = SpeakerSC(
                                    root=root,
                                    speaker=speaker
                                )
        
    users = list(datasets.keys())
    groups = None
    train_data, test_data = {}, {}

    for user in tqdm(users, "Generating training and testing splits", leave=False):
        user_dataset = datasets[user]

        total_size = len(user_dataset)
        train_size = int(total_size * train_split)
        test_size = total_size - train_size

        train_data[user], test_data[user] = torch.utils.data.random_split(user_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    
    return users, groups, train_data, test_data


def federate_dataset_iid(root: Path, seed: float, num_clients: int, train_split: float):
    data = SPEECHCOMMANDS(
        root=root,
        download=True
    )
    
    users = []
    groups = None
    train_data, test_data = {}, {}

    n = len(data)
    data_per_client = len(data) // num_clients

    split = [data_per_client + 1  if i < n % num_clients else data_per_client for i in range(num_clients)]

    federated = torch.utils.data.random_split(data, split, generator=torch.Generator().manual_seed(seed))

    for client in range(num_clients):
        user = str(client)

        users.append(user)

        train_size = int(split[client]  * train_split)
        test_size = split[client] - train_size
        train_data[user], test_data[user] = torch.utils.data.random_split(federated[client], [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    return users, groups, train_data, test_data
