import numpy as np
import torch


import rayleaf

from rayleaf.models.femnist.cnn import ClientModel as CNN


def test_simple_neq():
    rng = np.random.RandomState(0)
    arr1 = rng.rand(3, 1, 4, 1)
    arr2 = rng.rand(5, 9)
    arr3 = rng.rand(2, 6, 5)
    arr4 = rng.rand(3, 5, 8)
    arr = [arr1, arr2, arr3, arr4]

    ta1 = rayleaf.TensorArray(arr)
    
    rng = np.random.RandomState(1)
    arr1 = rng.rand(3, 1, 4, 1)
    arr2 = rng.rand(5, 9)
    arr3 = rng.rand(2, 6, 5)
    arr4 = rng.rand(3, 5, 8)
    arr = [arr1, arr2, arr3, arr4]

    ta2 = rayleaf.TensorArray(arr)

    assert ta1 != ta2


def test_model_neq():
    torch.manual_seed(0)
    model = CNN(0, 0.05, 62)
    ta1 = rayleaf.TensorArray(model)

    torch.manual_seed(1)
    model = CNN(0, 0.05, 62)
    ta2 = rayleaf.TensorArray(model)

    assert ta1 != ta2


def test_type_mismatch():
    torch.manual_seed(0)
    model = CNN(0, 0.05, 62)
    ta1 = rayleaf.TensorArray(model)

    ta2 = 0

    assert ta1 != ta2


def test_len_mismatch():
    torch.manual_seed(0)
    model = CNN(0, 0.05, 62)
    ta1 = rayleaf.TensorArray(model)

    ta2 = rayleaf.TensorArray([[0]])

    assert ta1 != ta2


def test_shape_mismatch():
    rng = np.random.RandomState(0)
    arr1 = rng.rand(3, 1, 4, 1)
    arr2 = rng.rand(5, 9)
    arr3 = rng.rand(2, 6, 5)
    arr4 = rng.rand(3, 5, 8)
    arr = [arr1, arr2, arr3, arr4]

    ta1 = rayleaf.TensorArray(arr)

    rng = np.random.RandomState(0)
    arr1 = rng.rand(3, 1, 4, 1)
    arr2 = rng.rand(5, 9)
    arr3 = rng.rand(2, 6, 5)
    arr4 = rng.rand(3, 5, 9)
    arr = [arr1, arr2, arr3, arr4]
    ta2 = rayleaf.TensorArray(arr)

    assert ta1 != ta2
