"""Symbolic array classes."""

from __future__ import annotations

from itertools import product
from typing import get_args, Callable, Generator, Optional, Union
import sympy as sym
from sympy.tensor.array import permutedims
import numpy as np
from numpy.typing import NDArray, DTypeLike

SympyArray = Union[sym.NDimArray, sym.MatrixBase]
ArrayLike = Union[SympyArray, NDArray]
ScalarLike = Union[sym.Expr, int, float, complex]
ShapeLike = Union[int, tuple, sym.Tuple]


def is_array(obj) -> bool:
    """Check if object is a valid SymPy or NumPy array type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid array type.
    """
    return isinstance(obj, get_args(ArrayLike))


def is_sympy_array(obj) -> bool:
    """Check if object is a valid SymPy array type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid SymPy array type.
    """
    return isinstance(obj, get_args(SympyArray))


def is_scalar(obj) -> bool:
    """Check if object is a symbolic or numeric scalar type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid scalar type.
    """
    return isinstance(obj, get_args(ScalarLike)) or np.isscalar(obj)


def is_valid_shape(obj) -> bool:
    """Check if object is a valid array shape type.
    Args:
        obj: Object to check.

    Returns:
        Whether object is valid array shape type.
    """
    return isinstance(obj, get_args(ShapeLike))


def _broadcastable_shapes(shape_1: ShapeLike, shape_2: ShapeLike) -> bool:
    """Check if two array shapes are compatible for broadcasting."""
    return all(
        (s1 == s2 or s1 == 1 or s2 == 1) for s1, s2 in zip(shape_1[::-1], shape_2[::-1])
    )


def binary_broadcasting_func(
    func: Callable[[ScalarLike, ScalarLike], ScalarLike],
    name: Optional[str] = None,
    doc: Optional[str] = None,
) -> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    """Wrap binary function to give broadcasting semantics.

    Args:
        func: Binary argument function to wrap.
        name: Name to assign to wrapped function's `__name__` attribute.
        doc: Docstring to assign to wrapped function's `__doc__` attribute.

    Returns:
        Wrapped function which broadcasts on both arguments.
    """

    name = func.__name__ if name is None else name

    def wrapped_func(arg_1, arg_2):
        if is_scalar(arg_1) and is_scalar(arg_2):
            return func(arg_1, arg_2)
        elif is_scalar(arg_1) and is_array(arg_2):
            arg_2 = as_symbolic_array(arg_2)
            return SymbolicArray([func(arg_1, a2) for a2 in arg_2.flat], arg_2.shape)
        elif is_array(arg_1) and is_scalar(arg_2):
            arg_1 = as_symbolic_array(arg_1)
            return SymbolicArray([func(a1, arg_2) for a1 in arg_1.flat], arg_1.shape)
        elif is_array(arg_1) and is_array(arg_2):
            arg_1 = as_symbolic_array(arg_1)
            arg_2 = as_symbolic_array(arg_2)
            if arg_1.shape == arg_2.shape:
                return SymbolicArray(
                    [func(a1, a2) for a1, a2 in zip(arg_1.flat, arg_2.flat)],
                    arg_1.shape,
                )
            elif _broadcastable_shapes(arg_1.shape, arg_2.shape):
                broadcaster = np.broadcast(arg_1, arg_2)
                return SymbolicArray(
                    [func(a1, a2) for a1, a2 in broadcaster], broadcaster.shape
                )
            else:
                raise ValueError(
                    f"operands could not be broadcast together with shapes "
                    f"{arg_1.shape} {arg_2.shape}."
                )
        else:
            raise NotImplementedError(
                f"{name} not implemented for arguments of types {type(arg_1)} "
                f"and {type(arg_2)}."
            )

    wrapped_func.__name__ = name
    wrapped_func.__doc__ = func.__doc__ if doc is None else doc

    return wrapped_func


def unary_elementwise_func(
    func: Callable[[ScalarLike], ScalarLike],
    name: Optional[str] = None,
    doc: Optional[str] = None,
) -> Callable[[ArrayLike], ArrayLike]:
    """Wrap unary function to give elementwise semantics.
    Args:
        func: Binary argument function to wrap.
        name: Name to assign to wrapped function's `__name__` attribute.
        doc: Docstring to assign to wrapped function's `__doc__` attribute.

    Returns:
        Wrapped function which acts elementwise on argument.
    """

    name = func.__name__ if name is None else name

    def wrapped_func(arg):
        if is_scalar(arg):
            return func(arg)
        elif is_array(arg):
            arg = as_symbolic_array(arg)
            return SymbolicArray([func(a) for a in arg.flat], arg.shape)
        else:
            raise NotImplementedError(
                f"{name} not implemented for argument of type {type(arg)}."
            )

    wrapped_func.__name__ = name
    wrapped_func.__doc__ = func.__doc__ if doc is None else doc

    return wrapped_func


def slice_iterator(arr: ArrayLike, axes: Union[int, tuple[int, ...]]) -> Generator:
    """Iterate over slices of array from indexing along a subset of axes."""
    if isinstance(axes, int):
        axes = (axes,)
    # Wrap negative axes
    axes = tuple(ax % arr.ndim for ax in axes)
    for indices in product(*[range(arr.shape[ax]) for ax in axes]):
        yield arr[
            tuple(
                indices[axes.index(ax)] if ax in axes else slice(None)
                for ax in range(arr.ndim)
            )
        ]


def named_array(
    name: str, shape: ShapeLike, dtype: Optional[DTypeLike] = None
) -> SymbolicArray:
    """Create a symbolic array with common name prefix to elements.

    Args:
        name: Name prefix to use for symbolic array elements.
        shape: Dimensions of array.
        dtype: NumPy dtype to use for array.
        
    Returns:
        Symbolic array with elements `{name}[index_list]` for `index_list` iterating
        over strings of valid comma-separated indices, for example calling with
        `name="a"` and `shape=(2, 2)` would give a symbolic array of shape `(2, 2)` with
        elements `a[0, 0]`, `a[0, 1]`, `a[1, 0]` and `a[1, 1]`.
    """
    if dtype is None:
        dtype = np.float64
    assumptions = {
        "integer": np.issubdtype(dtype, np.integer),
        "real": not np.issubdtype(dtype, np.complexfloating),
        "complex": True,  # Complex numbers are superset of reals
    }
    if shape == () or shape is None:
        array = SymbolicArray([sym.Symbol(name, **assumptions)], (), dtype)
    elif is_valid_shape(shape):
        if isinstance(shape, int):
            shape = (shape,)
        array = SymbolicArray(
            [
                sym.Symbol(
                    f'{name}[{", ".join([str(i) for i in index])}]', **assumptions
                )
                for index in product(*(range(s) for s in shape))
            ],
            shape,
            dtype,
        )
    else:
        raise ValueError(f"Unrecognised shape type {type(shape)} with value {shape}.")
    array._name = name
    return array


def infer_dtype(array: SymbolicArray) -> DTypeLike:
    """Infer safe dtype for array.

    Args:
        array: Array to infer dtype for.

    Returns:
        NumPy dtype which can represent array elements.
    """
    if all(el.is_integer for el in array.flat):
        return np.int64
    elif all(el.is_real for el in array.flat):
        return np.float64
    elif all(el.is_complex for el in array.flat):
        return np.complex128
    else:
        return object


def _matrix_multiply(left: SymbolicArray, right: SymbolicArray) -> SymbolicArray:
    """Perform symbolic matrix multiply of two 1D or 2D arrays."""
    if not (left.ndim in (1, 2) and right.ndim in (1, 2)):
        raise NotImplementedError(
            "Matrix multiplication only implemented for 1D and 2D operands."
        )
    elif not left.shape[-1] == right.shape[0]:
        raise ValueError(
            f"Incompatible shapes {left.shape} and {right.shape} for matrix "
            f"multiplication."
        )
    if left.ndim == 1 and right.ndim == 1:
        return sum(left * right)
    elif left.ndim == 2 and right.ndim == 1:
        return SymbolicArray(
            [sum(left[i, :] * right) for i in range(left.shape[0])],
            shape=(left.shape[0],),
        )
    elif left.ndim == 1 and right.ndim == 2:
        return SymbolicArray(
            [sum(left * right[:, i]) for i in range(right.shape[-1])],
            shape=(right.shape[-1],),
        )
    elif left.ndim == 2 and right.ndim == 2:
        return SymbolicArray(
            [
                sum(left[i, :] * right[:, j])
                for i in range(left.shape[0])
                for j in range(right.shape[-1])
            ],
            shape=(left.shape[0], right.shape[-1]),
        )


def as_symbolic_array(array: Union[ArrayLike, SymbolicArray]) -> SymbolicArray:
    if isinstance(array, SymbolicArray):
        return array
    else:
        return SymbolicArray(array, array.shape)


class SymbolicArray(sym.ImmutableDenseNDimArray):
    """Symbolic n-dimensional array with NumPy-like interface.

    Specifically implements NumPy style operator overloading and broadcasting
    semantics.
    """

    __array_priority__ = 1

    def __new__(cls, iterable, shape=None, dtype=None):
        instance = super().__new__(SymbolicArray, iterable, shape)
        instance._dtype = dtype
        return instance

    def __array__(self, dtype=None):
        if len(self.free_symbols) > 0:
            if dtype is not None:
                raise ValueError(
                    f"Array contains free symbols, therefore cannot cast to "
                    f"NumPy array of dtype {dtype}."
                )
            else:
                dtype = object
        else:
            dtype = self.dtype if dtype is None else dtype
        return np.array(self.tolist(), dtype)

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = infer_dtype(self)
        return self._dtype

    @binary_broadcasting_func
    def __mul__(self, other):
        return self * other

    @binary_broadcasting_func
    def __rmul__(self, other):
        return other * self

    @binary_broadcasting_func
    def __truediv__(self, other):
        return self / other

    __div__ = __truediv__

    @binary_broadcasting_func
    def __rtruediv__(self, other):
        return other / self

    __rdiv__ = __rtruediv__

    @binary_broadcasting_func
    def __floordiv__(self, other):
        return self // other

    @binary_broadcasting_func
    def __rfloordiv__(self, other):
        return other // self

    @binary_broadcasting_func
    def __mod__(self, other):
        return self % other

    @binary_broadcasting_func
    def __add__(self, other):
        return self + other

    @binary_broadcasting_func
    def __radd__(self, other):
        return other + self

    @binary_broadcasting_func
    def __sub__(self, other):
        return self - other

    @binary_broadcasting_func
    def __rsub__(self, other):
        return other - self

    @binary_broadcasting_func
    def __pow__(self, other):
        return self**other

    @binary_broadcasting_func
    def __rpow__(self, other):
        return other**self

    @binary_broadcasting_func
    def __eq__(self, other):
        return self == other

    @binary_broadcasting_func
    def __ne__(self, other):
        return self != other

    @binary_broadcasting_func
    def __lt__(self, other):
        return self < other

    @binary_broadcasting_func
    def __le__(self, other):
        return self <= other

    @binary_broadcasting_func
    def __gt__(self, other):
        return self > other

    @binary_broadcasting_func
    def __ge__(self, other):
        return self >= other

    @unary_elementwise_func
    def __neg__(self):
        return -self

    @unary_elementwise_func
    def __pos__(self):
        return self

    @unary_elementwise_func
    def __abs__(self):
        return abs(self)

    def __matmul__(self, other):
        if not is_array(other):
            return NotImplemented
        other = SymbolicArray(other)
        return _matrix_multiply(self, other)

    def __rmatmul__(self, other):
        if not is_array(other):
            return NotImplemented
        other = SymbolicArray(other)
        return _matrix_multiply(other, self)

    @property
    @unary_elementwise_func
    def real(self):
        return sym.re(self)

    @property
    @unary_elementwise_func
    def imag(self):
        return sym.im(self)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def flatten(self):
        return SymbolicArray(self.flat, self.size)

    @property
    def flat(self):
        if self.shape == ():
            yield self._args[0][0]
        else:
            for idx in product(*(range(s) for s in self.shape)):
                yield self[idx]

    def tolist(self):
        if self.shape == ():
            return self._args[0][0]
        else:
            return super().tolist()

    @property
    def T(self):
        return SymbolicArray(
            [
                self[tuple(indices[::-1])]
                for indices in product(*[range(s) for s in self.shape[::-1]])
            ],
            self.shape[::-1],
        )

    def transpose(self, axes=None):
        if axes is None:
            return self.T
        else:
            return permutedims(self, axes)

    def reshape(self, shape):
        return SymbolicArray(self.flat, shape)

    def any(self, axis=None):
        if axis is None:
            return any(self.flat)
        else:
            raise NotImplementedError()

    def all(self, axis=None):
        if axis is None:
            return all(self.flat)
        else:
            raise NotImplementedError()

    def max(self, axis=None):
        if axis is None:
            return max(self.flat)
        else:
            raise NotImplementedError()

    def min(self, axis=None):
        if axis is None:
            return min(self.flat)
        else:
            raise NotImplementedError()

    def sum(self, axis=None):
        if axis is None:
            return sum(self.flat)
        elif isinstance(axis, (tuple, list, int)):
            return sum(slice_iterator(self, axis))
        else:
            raise ValueError(f"Unrecognised axis type {type(axis)}.")

    def prod(self, axis=None):
        if axis is None:
            return sym.prod(self.flat)
        elif isinstance(axis, (tuple, list, int)):
            return sym.prod(slice_iterator(self, axis))
        else:
            raise ValueError(f"Unrecognised axis type {type(axis)}.")
