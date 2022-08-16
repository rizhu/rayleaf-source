import numpy as np


import rayleaf


def test_simple_norm():
    ta = rayleaf.TensorArray([np.ones(9), np.ones(16)])

    assert np.linalg.norm(ta) == 5
