"""
Based on https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html#formatting-the-data
"""


import torch
import torch.nn.functional as F


LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero"
]


def label_to_index(word):
    return torch.tensor(LABELS.index(word))


def index_to_label(index):
    return LABELS[index]


def pad_waveform(waveform, frequency: int):
    return F.pad(waveform, (0, frequency - waveform.shape[-1]), mode="constant", value=0.0)


def make_collate_fn(frequency: int):
    def collate_fn(batch):
        tensors, targets = [], []

        for waveform, _, label, *_ in batch:
            tensors.append(pad_waveform(waveform, frequency))
            targets.append(label_to_index(label))

        tensors = torch.stack(tensors)
        targets = torch.stack(targets)

        return tensors, targets
    
    return collate_fn
