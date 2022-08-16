import torch
import numpy as np

from torch import Tensor


import rayleaf


def test_simple():
    shape1 = (3, 1, 4, 1)
    shape2 = (1, 5, 9)

    arr1 = np.ones(shape1)
    arr2 = np.ones(shape2)
    arr = [arr1, arr2]

    ta = rayleaf.TensorArray(arr)
    flat_ta = Tensor(np.ones((np.prod(shape1) + np.prod(shape2))))

    assert torch.all(ta.__array__() == flat_ta)
