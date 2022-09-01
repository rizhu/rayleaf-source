from typing import Union


from numpy import ndarray
from scipy.linalg import block_diag
from torch import stack, Tensor


def _tensor_to_block_diag(tensor: Union[ndarray, Tensor]) -> Tensor:
    if len(tensor.shape) <= 2:
        return tensor
    
    mats = []
    for i in range(tensor.shape[0]):
        mats.append(_tensor_to_block_diag(tensor[i]))
    
    return Tensor(block_diag(*mats)).requires_grad_(False)


def _block_diag_to_tensor(mat: Union[ndarray, Tensor], shape: tuple, start_inds: list = [0, 0]) -> Tensor:
    row = start_inds[0]
    col = start_inds[1]

    if len(shape) == 1:
        res = Tensor(mat[row][col:col + shape[0]]).requires_grad_(False)
        start_inds[0] += 1
        start_inds[1] += shape[0]
        return res
    if len(shape) == 2:
        res = Tensor(mat[row:row + shape[0], col:col + shape[1]]).requires_grad_(False)
        start_inds[0] += shape[0]
        start_inds[1] += shape[1]
        return res

    tensors = []

    for _ in range(shape[0]):
        tensors.append(_block_diag_to_tensor(mat, shape[1:], start_inds))
    
    return stack(tensors).requires_grad_(False)

