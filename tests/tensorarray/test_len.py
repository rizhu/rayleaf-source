import torch


import rayleaf

from rayleaf.models.femnist.cnn import ClientModel as CNN


def test_simple():
    t1 = torch.zeros(1, 2, 3, 4)
    t2 = torch.zeros(5)

    ta = rayleaf.TensorArray([t1, t2])
    assert len(ta) == 2


def test_model():
    model = CNN(0, 0.05, 62)
    
    ta = rayleaf.TensorArray(model)
    assert len(ta) == len(tuple(model.parameters()))
