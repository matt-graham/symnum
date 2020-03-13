"""Symbolic array classes."""

from itertools import product
from collections import Iterable
import sympy as sym
from sympy.tensor.array import permutedims
import numpy as np


SYMPY_ARRAY_TYPES = (sym.NDimArray, sym.MatrixBase)
ARRAY_TYPES = SYMPY_ARRAY_TYPES  + (np.ndarray,)
SCALAR_TYPES = (sym.Expr, int, float)


def flatten(iterable):
    """Recursively flatten nested iterables to a list."""
    flattened = []
    for el in iterable:
        if isinstance(el, Iterable):
            flattened.extend(flatten(el))
        else:
            flattened.append(el)
    return flattened


def broadcastable_shapes(shape_1, shape_2):
    """Check if two array shapes are compatible for broadcasting."""
    return all(
        (s1 == s2 or s1 == 1 or s2 == 1)
        for s1, s2 in zip(shape_1[::-1], shape_2[::-1]))


def binary_elementwise_op(op):
    """Wrap binary elementwise operations to give broadcasting semantics."""
    
    def wrapped_op(self, other):
        if isinstance(other, ARRAY_TYPES):
            if other.shape == self.shape:
                return SymbolicArray(
                    [op(s, o) for s, o in zip(flatten(self), flatten(other))], 
                    self.shape)
            elif broadcastable_shapes(self.shape, other.shape):
                broadcaster = np.broadcast(self, other)
                return SymbolicArray(
                    [op(s, o) for s, o in broadcaster], broadcaster.shape)
            else:
                raise ValueError(
                    f'operands could not be broadcast together with shapes '
                    f'{self.shape} {other.shape}.')
        elif isinstance(other,  SCALAR_TYPES) or np.isscalar(other):
            return SymbolicArray(
                [op(s, other) for s in flatten(self)], self.shape)
        else:
            return NotImplemented  
        
    return wrapped_op


def unary_elementwise_op(op):
    """Wrap unary elementwise operations to give broadcasting semantics."""
    
    def wrapped_op(self):
        return SymbolicArray([op(s) for s in flatten(self)], self.shape)  
        
    return wrapped_op


def slice_iterator(arr, axes):
    """Iterate over slices of array from indexing along a subset of axes."""
    if isinstance(axes, int):
        axes = (axes,)
    # Wrap negative axes
    axes = tuple(ax % arr.ndim for ax in axes)
    for indices in product(*[range(arr.shape[ax]) for ax in axes]):
        yield arr[tuple(
            indices[axes.index(ax)] if ax in axes else slice(None) 
            for ax in range(arr.ndim))]
        
        
def named_array(name, shape):
    """Create a symbolic array with common name prefix to elements."""
    if shape == () or shape is None or shape == 1:
        array = SymbolicArray([sym.Symbol(name)], ())
    elif isinstance(shape, (int, tuple, sym.containers.Tuple)):
        if isinstance(shape, int):
            shape = (shape,)
        array = SymbolicArray(
            [sym.Symbol(f'{name}[{", ".join([str(i) for i in index])}]') 
             for index in product(*(range(s) for s in shape))], shape)
    else:
        raise ValueError(
            f'Unrecognised shape type {type(shape)} with value {shape}.')
    array._name = name
    return array
    
    
def _infer_dtype(array_elements):
    """Infer safe dtype for array elements."""
    if all(el.is_integer for el in array_elements):
        return np.int64
    elif all(el.is_real for el in array_elements):
        return np.float64
    elif all(el.is_complex for el in array_elements):
        return np.complex128
    else:
        return np.object
    
    
def _matrix_multiply(left, right):
    """Perform symbolic matrix multiply of two 1D or 2D arrays."""
    if not (left.ndim in (1, 2) and right.ndim in (1, 2)):
        raise NotImplementedError(
            'Matrix multiplication only implemented for 1D and 2D operands.')
    elif not left.shape[-1] == right.shape[0]:
        raise ValueError(
            f'Incompatible shapes {left.shape} and {right.shape} for matrix '
            f'multiplication.')
    if left.ndim == 1 and right.ndim == 1:
        return sum(left * right)
    elif left.ndim == 2 and right.ndim == 1:
        return SymbolicArray(
            [sum(left[i, :] * right) for i in range(left.shape[0])], 
            shape=(left.shape[0],))
    elif left.ndim == 1 and right.ndim == 2:
        return SymbolicArray(
            [sum(left * right[:, i]) for i in range(right.shape[-1])], 
            shape=(right.shape[-1],))
    elif left.ndim == 2 and right.ndim == 2:
        return SymbolicArray(
            [sum(left[i, :] * right[:, j]) for i in range(left.shape[0]) 
             for j in range(right.shape[-1])], 
            shape=(left.shape[0], right.shape[-1]))


class SymbolicArray(sym.ImmutableDenseNDimArray):
    """Symbolic n-dimensional array with NumPy-like interface.
    
    Specifically implements NumPy style operator overloading and broadcasting 
    semantics.
    """
    
    __array_priority__ = 1
    
    def __new__(cls, iterable, shape=None, dtype=None):
        instance = super().__new__(SymbolicArray, iterable, shape)
        instance.dtype = dtype
        return instance
    
    def __array__(self, dtype=None):
        if len(self.free_symbols) > 0:
            if dtype is not None:
                raise ValueError(
                    f'Array contains free symbols, therefore cannot cast to '
                    f'NumPy array of dtype {dtype}.')
            else:
                dtype = np.object
        else:
            dtype = self._infer_dtype() if dtype is None else dtype        
        return np.array(self.tolist(), dtype)
             
    def _infer_dtype(self):
        return (self.dtype if self.dtype is not None 
                else _infer_dtype(self.flatten()))
    
    @binary_elementwise_op
    def __truediv__(self, other):
        return self / other
    
    @binary_elementwise_op
    def __rtruediv__(self, other):
        return other / self
    
    @binary_elementwise_op
    def __mul__(self, other):
        return self * other
    
    @binary_elementwise_op
    def __rmul__(self, other):
        return other * self
    
    @binary_elementwise_op
    def __add__(self, other):
        return self + other
    
    @binary_elementwise_op
    def __radd__(self, other):
        return other + self
    
    @binary_elementwise_op
    def __sub__(self, other):
        return self - other  
    
    @binary_elementwise_op
    def __rsub__(self, other):
        return other - self
    
    @binary_elementwise_op
    def __pow__(self, other):
        return self**other
    
    @binary_elementwise_op
    def __eq__(self, other):
        return self == other
    
    @binary_elementwise_op
    def __ne__(self, other):
        return self != other
    
    @binary_elementwise_op
    def __lt__(self, other):
        return self < other
    
    @binary_elementwise_op
    def __le__(self, other):
        return self <= other
    
    @binary_elementwise_op
    def __gt__(self, other):
        return self > other
    
    @binary_elementwise_op
    def __ge__(self, other):
        return self >= other    
    
    @unary_elementwise_op
    def __neg__(self):
        return -self
    
    @unary_elementwise_op
    def __pos__(self):
        return self
    
    @unary_elementwise_op
    def __abs__(self):
        return abs(self)
        
    @property
    def size(self):
        return np.prod(self.shape)
    
    @property
    def ndim(self):
        return len(self.shape)
    
    def flatten(self):
        return flatten(self.tolist())
    
    @property
    def flat(self):
        for el in self.flatten():
            yield el
            
    @property
    def T(self):
        return SymbolicArray(
            [self[tuple(indices[::-1])] 
             for indices in product(*[range(s) for s in self.shape[::-1]])],
            self.shape[::-1])

    def transpose(self, axes=None):
        if axes is None:
            return self.T
        else:
            return permutedims(self, axes)
            
    
    def reshape(self, shape):
        return SymbolicArray(self.flatten(), shape)
    
    def sum(self, axis=None):
        if axis is None:
            return sum(self.flatten())
        elif isinstance(axis, (tuple, list, int)):
            return sum(slice_iterator(self, axis))
        else:
            raise ValueError(f'Unrecognised axis type {type(axis)}.')
            
    def __matmul__(self, other):
        if not isinstance(other, SYMPY_ARRAY_TYPES + (np.ndarray,)):
            return NotImplemented
        other = SymbolicArray(other)
        return _matrix_multiply(self, other)
        
    def __rmatmul__(self, other):
        if not isinstance(other, SYMPY_ARRAY_TYPES + (np.ndarray,)):
            return NotImplemented
        other = SymbolicArray(other)
        return _matrix_multiply(other, self)
