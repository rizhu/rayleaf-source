import torch


import rayleaf


def test_modify_tensor_after_creating_ta_simple():
    t = torch.ones(4, 4)
    ta1 = rayleaf.TensorArray([t])
    t[0][0] = 0
    ta2 = rayleaf.TensorArray([t])

    assert ta1 != ta2
