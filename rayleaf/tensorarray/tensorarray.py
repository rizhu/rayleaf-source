import numpy as np
import torch

from torch import nn, Tensor


HANDLED_FUNCTIONS = {}

def implements(np_function):
   "Register an __array_function__ implementation for DiagonalArray objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator


class TensorArray(np.lib.mixins.NDArrayOperatorsMixin):


    def unflatten(flat_arr, shapes):
        try:
            if (flat_arr.ndim > 1):
                raise TypeError(f"Input array should be 1-dimensional but has shape {flat_arr.shape}")
        except AttributeError:
            raise TypeError(f"Input array should be a numpy.ndarray or torch.Tensor but found {type(flat_arr)}")

        output_tensors = []
        start_index = 0

        for shape in shapes:
            try:
                count = np.prod(shape)
                output_tensors.append(Tensor(flat_arr[start_index:start_index + count]).reshape(shape))
            except TypeError:
                raise TypeError(f"shapes parameter must be an iterable of iterables of ints")
            start_index += count

        if start_index != flat_arr.shape[0]:
            raise TypeError(f"Input array length and shapes mismatch. Input array has length {flat_arr.shape[0]}"
                " but shapes implies length of {start_index}")
        
        return TensorArray(output_tensors)


    def _find_shapes(arg) -> tuple:
        try:
            for iterable in arg:
                if not isinstance(iterable, torch.Size):
                    for element in iterable:
                        if not (isinstance(element, int) and element > 0):
                            raise ValueError(f"Shape iterable {arg} contains non-integer values or integers less than"
                                " or equal to 0")
            return arg
        except TypeError:
            try:
                return arg.shapes
            except AttributeError:
                raise TypeError(f"Input argument must be an iterable, Model, or TensorArray but found {type(arg)}")


    def _tensor_from_shapes(shapes, tensor_generator, device):
        shapes = TensorArray._find_shapes(shapes)
        tensors = []

        for shape in shapes:
            tensors.append(tensor_generator(shape).to(device))

        return TensorArray(tensors)


    def zeros(arg, device="cpu"):
        return TensorArray._tensor_from_shapes(shapes=arg, tensor_generator=torch.zeros, device=device)


    def ones(arg, device="cpu"):
        return TensorArray._tensor_from_shapes(shapes=arg, tensor_generator=torch.ones, device=device)


    def randn(arg, device="cpu"):
        return TensorArray._tensor_from_shapes(shapes=arg, tensor_generator=torch.randn, device=device)


    def __init__(self, tensors, device="cpu") -> None:
        if isinstance(tensors, nn.Module):
            tensors = tensors.parameters()

        tensors = list(tensors)

        if len(tensors) > 0:
            for t in tensors:
                if isinstance(t, Tensor):
                    device = t.device
                    break

        try:
            self._tensors = tuple(t.detach().clone().to(device) if isinstance(t, Tensor) \
                else Tensor(t).requires_grad_(False).to(device) for t in tensors)
        except TypeError:
            raise TypeError(f"Cannot create TensorArray from object that is not iterable and"
                " not an instance of torch.nn.Module. Found object of type {type(tensors)}")

        self._dtype = self.tensors[0].dtype if len(self.tensors) > 0 else None
        
        self._shapes = None
        self._size = None


    @property
    def tensors(self) -> tuple:
        return self._tensors


    @property
    def dtype(self):
        return self._dtype


    def _compute_shapes_and_size(self) -> None:
        self._shapes = []
        self._size = 0
        for param in self.tensors:
            self._shapes.append(param.shape)
            self._size += np.prod(param.shape)
        self._shapes = tuple(self._shapes)


    @property
    def shapes(self) -> tuple:
        if self._shapes is None:
            self._compute_shapes_and_size()
        
        return self._shapes


    @property
    def size(self) -> int:
        if self._size is None:
            self._compute_shapes_and_size()
        
        return self._size


    def __repr__(self):
        repr = f"{self.__class__.__name__}(params={self._tensors})"
        return repr


    def __str__(self):
        st = f"Params with shape {self.shapes}\n"
        for i, param in enumerate(self.tensors):
            st += f"Param {i}: {param}\n"
        return st


    def __len__(self):
        return len(self.shapes)


    def __iadd__(self, other):
        return self + other


    def __isub__(self, other):
        return self - other


    def __imul__(self, other):
        return self * other


    def __itruediv__(self, other):
        return self / other


    def __ifloordiv__(self, other):
        return self // other


    def __imod__(self, other):
        return self % other


    def __ipow__(self, other):
        return self ** other


    def _compare(self, other, operator, default):
        if type(self) != type(other):
            return default
        elif len(self) != len(other):
            return default
        elif self.shapes != other.shapes:
            return default
        
        res = []
        for i in range(len(self)):
            res.append(operator(self.tensors[i], other.tensors[i]))
        
        return self.__class__(res)


    def __eq__(self, other):
        return self._compare(other, lambda x, y: x == y, default=False)


    def __ne__(self, other):
        return self._compare(other, lambda x, y: x != y, default=True)

    
    def __array__(self, dtype=None) -> Tensor:
        flattened_layers = []
        for tensor in self._tensors:
            flattened_layers.append(tensor.flatten())
        flattened = torch.cat(flattened_layers)
        
        return flattened.to(dtype)


    def flat(self, dtype=None):
        return self.__array__(dtype=dtype)


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            first_shapes = None

            sanitized_inputs = []
            for input_ in inputs:
                if isinstance(input_, self.__class__):
                    if first_shapes is None:
                        first_shapes = input_.shapes
                    elif first_shapes != input_.shapes:
                        raise TypeError(f"Found mismatched dimensions: {first_shapes} and {input_.shapes}")

                    sanitized_inputs.append(input_.flat())
                else:
                    sanitized_inputs.append(input_)

            flat_output = ufunc(*sanitized_inputs, **kwargs)
                    
            return self.__class__.unflatten(flat_output, first_shapes)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


    @implements(np.linalg.norm)
    def norm(x, ord=None, axis=None, keepdims=False):
        return np.linalg.norm(x.flat(), ord=ord, axis=axis, keepdims=keepdims)


    def to(self, arg):
        new_tensors = [t.to(arg) for t in self.tensors]
        return TensorArray(new_tensors)
