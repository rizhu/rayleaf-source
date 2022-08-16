import numpy as np


import rayleaf


def test_simple_array_ufunc():
    rng = np.random.RandomState(0)
    arr1 = rng.rand(3, 1, 4, 1)
    arr2 = rng.rand(5, 9)
    arr3 = rng.rand(2, 6, 5)
    arr4 = rng.rand(3, 5, 8)
    arr = [arr1, arr2, arr3, arr4]

    ta1 = rayleaf.TensorArray(arr)

    assert ta1 == 1 * ta1


def test_simple_add():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([2 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([3 * np.ones(shape) for shape in shapes])

    assert ta1 + ta2 == ta3
    assert ta1 + 2 == ta3


def test_simple_sub():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([2 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([-1 * np.ones(shape) for shape in shapes])

    assert ta1 - ta2 == ta3
    assert ta1 - 2 == ta3


def test_simple_mul():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([2 * np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([-3 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([-6 * np.ones(shape) for shape in shapes])

    assert ta1 * ta2 == ta3
    assert ta1 * -3 == ta3


def test_simple_pow():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([2 * np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([3 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([8 * np.ones(shape) for shape in shapes])

    assert ta1 ** ta2 == ta3
    assert ta1 ** 3 == ta3


def test_simple_truediv():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([6 * np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([3 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([2 * np.ones(shape) for shape in shapes])

    assert ta1 / ta2 == ta3
    assert ta1 / 3 == ta3


def test_simple_floordiv():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([8 * np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([3 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([2 * np.ones(shape) for shape in shapes])

    assert ta1 // ta2 == ta3
    assert ta1 // 3 == ta3


def test_simple_modulo():
    shape1 = (3, 1, 4, 1)
    shape2 = (5, 9)
    shape3 = (2, 6, 5)
    shape4 = (3, 5, 8)
    shapes = [shape1, shape2, shape3, shape4]

    ta1 = rayleaf.TensorArray([7 * np.ones(shape) for shape in shapes])
    ta2 = rayleaf.TensorArray([3 * np.ones(shape) for shape in shapes])
    ta3 = rayleaf.TensorArray([1 * np.ones(shape) for shape in shapes])

    assert ta1 % ta2 == ta3
    assert ta1 % 3 == ta3
