import torch
import numpy as np

from torch import Tensor


import rayleaf

from rayleaf.models.femnist.cnn import ClientModel as CNN
from rayleaf.models.speech_commands.m5 import ClientModel as M5


def test_simple_utils():
    shape = (3, 2, 3)

    arr = torch.arange(np.prod(shape)).reshape(shape)

    blocked = rayleaf.tensorarray.utils._tensor_to_block_diag(arr)
    expected = Tensor([
        [0, 1, 2, 0, 0, 0, 0, 0, 0],
        [3, 4, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 7, 8, 0, 0, 0],
        [0, 0, 0, 9, 10, 11, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 12, 13, 14],
        [0, 0, 0, 0, 0, 0, 15, 16, 17]
    ])
    assert torch.all(blocked == expected)

    reconstructed = rayleaf.tensorarray.utils._block_diag_to_tensor(blocked, shape)
    assert torch.all(reconstructed == arr)


def test_simple_ta():
    shapes = [(2, 2, 3), (3, 2), (2, 1, 2)]

    tensors = [torch.arange(np.prod(shape)).reshape(shape) for shape in shapes]
    ta = rayleaf.TensorArray(tensors)

    blocked = ta.as_block_diag()
    expected = Tensor([
        [0, 1, 2, 0, 0,  0,  0, 0, 0, 0, 0, 0],
        [3, 4, 5, 0, 0,  0,  0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 7,  8,  0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9, 10, 11, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,  0,  0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,  0,  2, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,  0,  4, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,  0,  0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 2, 3],
    ])
    assert torch.all(blocked == expected)

    reconstructed = rayleaf.TensorArray.from_block_diag(blocked, shapes)
    assert reconstructed == ta


def test_model_femnist():
    torch.manual_seed(0)
    model = CNN(0, 0.05, 62)
    ta = rayleaf.TensorArray(model)

    print(model.shapes)

    blocked = ta.as_block_diag()
    print("femnist", blocked.shape)
    reconstructed = rayleaf.TensorArray.from_block_diag(blocked, model.shapes)
    assert reconstructed == ta


def test_model_speech_commands():
    torch.manual_seed(0)
    model = M5(0, 0.05, 35)
    ta = rayleaf.TensorArray(model)

    print(model.shapes)

    blocked = ta.as_block_diag()
    print("speech_commands", blocked.shape)
    reconstructed = rayleaf.TensorArray.from_block_diag(blocked, model.shapes)
    assert reconstructed == ta
