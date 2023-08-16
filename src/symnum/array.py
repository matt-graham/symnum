"""Symbolic array classes."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Union, get_args, get_origin

import numpy as np
import sympy
from numpy.typing import DTypeLike, NDArray
from sympy.tensor.array import permutedims

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Callable, Optional, TypeAlias

SympyArray: TypeAlias = Union[sympy.NDimArray, sympy.MatrixBase]
ScalarLike: TypeAlias = Union[
    sympy.Expr,
    sympy.logic.boolalg.Boolean,
    bool,
    int,
    float,
    complex,
    np.number,
]
ShapeLike: TypeAlias = Union[int, tuple[int, ...], sympy.Tuple]


def _get_union_type_unsubscripted_args(union_type: type) -> tuple[type]:
    return tuple(
        get_origin(t) if get_origin(t) is not None else t for t in get_args(union_type)
    )


def is_array(obj) -> bool:
    """Check if object is a valid SymNum, SymPy or NumPy array type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid array type.
    """
    return isinstance(obj, _get_union_type_unsubscripted_args(ArrayLike))


def is_sympy_array(obj) -> bool:
    """Check if object is a valid SymPy array type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid SymPy array type.
    """
    return isinstance(obj, _get_union_type_unsubscripted_args(SympyArray))


def is_scalar(obj) -> bool:
    """Check if object is a symbolic or numeric scalar type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid scalar type.
    """
    return isinstance(
        obj,
        _get_union_type_unsubscripted_args(ScalarLike),
    ) or np.isscalar(obj)


def is_valid_shape(obj) -> bool:
    """Check if object is a valid array shape type.

    Args:
        obj: Object to check.

    Returns:
        Whether object is valid array shape type.
    """
    return isinstance(obj, _get_union_type_unsubscripted_args(ShapeLike))


def _broadcastable_shapes(shape_1: ShapeLike, shape_2: ShapeLike) -> bool:
    """Check if two array shapes are compatible for broadcasting."""
    return all(
        (s1 in (1, s2) or s2 == 1) for s1, s2 in zip(shape_1[::-1], shape_2[::-1])
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

    def wrapped_func(
        arg_1: Union[ScalarLike, ArrayLike],
        arg_2: Union[ScalarLike, ArrayLike],
    ) -> Union[ScalarLike, ArrayLike]:
        if is_scalar(arg_1) and is_scalar(arg_2):
            return func(arg_1, arg_2)
        elif is_scalar(arg_1) and is_array(arg_2):
            arg_2 = as_symbolic_array(arg_2)
            return SymbolicArray(
                [func(arg_1, a2) for a2 in arg_2.flat],
                shape=arg_2.shape,
            )
        elif is_array(arg_1) and is_scalar(arg_2):
            arg_1 = as_symbolic_array(arg_1)
            return SymbolicArray(
                [func(a1, arg_2) for a1 in arg_1.flat],
                shape=arg_1.shape,
            )
        elif is_array(arg_1) and is_array(arg_2):
            arg_1 = as_symbolic_array(arg_1)
            arg_2 = as_symbolic_array(arg_2)
            if arg_1.shape == arg_2.shape:
                return SymbolicArray(
                    [func(a1, a2) for a1, a2 in zip(arg_1.flat, arg_2.flat)],
                    shape=arg_1.shape,
                )
            elif _broadcastable_shapes(arg_1.shape, arg_2.shape):
                broadcaster = np.broadcast(arg_1, arg_2)
                return SymbolicArray(
                    [func(a1, a2) for a1, a2 in broadcaster],
                    shape=broadcaster.shape,
                )
            else:
                msg = (
                    f"operands could not be broadcast together with shapes "
                    f"{arg_1.shape} {arg_2.shape}."
                )
                raise ValueError(msg)
        else:
            msg = (
                f"{name} not implemented for arguments of types {type(arg_1)} "
                f"and {type(arg_2)}."
            )
            raise NotImplementedError(msg)

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

    def wrapped_func(arg: Union[ScalarLike, ArrayLike]) -> Union[ScalarLike, ArrayLike]:
        if is_scalar(arg):
            return func(arg)
        elif is_array(arg):
            arg = as_symbolic_array(arg)
            return SymbolicArray(
                [func(a) for a in arg.flat],
                shape=arg.shape,
            )
        else:
            msg = f"{name} not implemented for argument of type {type(arg)}."
            raise NotImplementedError(msg)

    wrapped_func.__name__ = name
    wrapped_func.__doc__ = func.__doc__ if doc is None else doc

    return wrapped_func


def _slice_iterator(
    arr: ArrayLike,
    axes: Union[int, tuple[int, ...]],
) -> Iterator[ArrayLike]:
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
    name: str,
    shape: ShapeLike,
    dtype: Optional[DTypeLike] = None,
) -> SymbolicArray:
    """Create a symbolic array with common name prefix to elements.

    Args:
        name: Name prefix to use for symbolic array elements.
        shape: Dimensions of array.
        dtype: NumPy dtype to use for array.

    Returns:
        Symbolic array with elements `name[index_list]` for `index_list` iterating
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
        array = SymbolicArray(
            [sympy.Symbol(name, **assumptions)],
            shape=(),
            dtype=dtype,
        )
    elif is_valid_shape(shape):
        if isinstance(shape, int):
            shape = (shape,)
        array = SymbolicArray(
            [
                sympy.Symbol(
                    f'{name}[{", ".join([str(i) for i in index])}]',
                    **assumptions,
                )
                for index in product(*(range(s) for s in shape))
            ],
            shape=shape,
            dtype=dtype,
        )
    else:
        msg = f"Unrecognised shape type {type(shape)} with value {shape}."
        raise ValueError(msg)
    array._name = name
    return array


def infer_array_dtype(array: SymbolicArray) -> DTypeLike:
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


def infer_scalar_dtype(scalar: ScalarLike) -> DTypeLike:
    """Infer safe dtype for array.

    Args:
        array: Scalar to infer dtype for.

    Returns:
        NumPy dtype which can represent array elements.
    """
    if isinstance(scalar, np.number):
        return scalar.dtype
    elif isinstance(scalar, (int, float, complex)):
        return np.dtype(type(scalar))
    elif isinstance(scalar, sympy.Expr):
        if scalar.is_integer:
            return np.dtype("int64")
        elif scalar.is_real:
            return np.dtype("float64")
        elif scalar.is_complex:
            return np.dtype("complex128")
        else:
            return object
    else:
        msg = "Argument is not of valid scalar type."
        raise TypeError(msg)


def _matrix_multiply(left: SymbolicArray, right: SymbolicArray) -> SymbolicArray:
    """Perform symbolic matrix multiply of two 1D or 2D arrays."""
    if not (left.ndim in (1, 2) and right.ndim in (1, 2)):
        msg = "Matrix multiplication only implemented for 1D and 2D operands."
        raise NotImplementedError(msg)
    if left.shape[-1] != right.shape[0]:
        msg = (
            f"Incompatible shapes {left.shape} and {right.shape} for matrix "
            f"multiplication."
        )
        raise ValueError(msg)
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
    else:
        msg = "Arguments unexpectedly failed to match any of branch conditions."
        raise RuntimeError(msg)


def as_symbolic_array(array: Union[ArrayLike, SymbolicArray]) -> SymbolicArray:
    """Convert an array to a `SymbolicArray` instance if not already an instance.

    Args:
        array: Array to return as :py:class:`SymbolicArray` instance.

    Returns:
        Original array if argument is already an :py:class:`SymbolicArray` instance and
        :py:class:`SymbolicArray`SymbolicArray` with equivalent shape and entries to
        array argument if not.
    """
    if isinstance(array, SymbolicArray):
        return array
    else:
        return SymbolicArray(array, shape=array.shape)


def _implements_numpy_ndarray_method(method):
    method.__doc__ = (
        f"SymNum implementation of :py:obj:`numpy.ndarray.{method.__name__}`."
    )
    return method


class SymbolicArray:
    """Symbolic `n`-dimensional array with NumPy-like interface.

    Specifically implements NumPy style operator overloading, broadcasting and indexing
    semantics.
    """

    __array_priority__ = 1

    def __init__(
        self,
        object_: Union[ScalarLike, ArrayLike, tuple, list],
        shape: Optional[tuple[int, ...]] = None,
        dtype: DTypeLike = None,
    ):
        """
        Args:
            object_: Object to construct symbolic array from. Either a nested sequence
                of (symbolic) scalars, an existing array like object or a single
                (symbolic) scalar.
            shape: Tuple defining shape of array (sizes of each dimension / axis). If
                :code:`object_` is array-like or scalar-like and :code:`shape` is set
                to :code:`None` (the default) the :code:`object_.shape` attribute will
                be used if array-like or a :code:`()` shape used if scalar-like.
            dtype: Object defining datatype to use for array when converting to NumPy
                arrays. If set to :code:`None` (default) then an appropriate datatype
                will be inferred from the properties of the symbolic expressions in the
                array.
        """
        if hasattr(object_, "shape") and object_.shape == ():
            # Avoid nesting 0-dimensional NumPy and symbolic arrays
            iterable = (object_[()],)
            shape = ()
        elif hasattr(object_, "__len__"):
            iterable = object_
        else:
            iterable = (object_,)
            shape = () if shape is None else shape
        self._base_array = sympy.ImmutableDenseNDimArray(iterable, shape)
        self._dtype = np.dtype(dtype) if dtype is not None else dtype

    def __getitem__(
        self,
        index: Union[int, tuple, slice],
    ) -> Union[SymbolicArray, sympy.Expr]:
        indexed_base_array = self._base_array[index]
        if is_scalar(indexed_base_array):
            return indexed_base_array
        return SymbolicArray(
            indexed_base_array,
            shape=indexed_base_array.shape,
            dtype=self._dtype,
        )

    def __len__(self):
        if self.shape == ():
            msg = "len() of unsized object"
            raise TypeError(msg)
        return self.shape[0]

    def __iter__(self) -> Iterator[Union[SymbolicArray, sympy.Expr]]:
        if self.shape == ():
            msg = "Iteration over a 0-d array"
            raise ValueError(msg)
        for i in range(self.shape[0]):
            yield self[i]

    @_implements_numpy_ndarray_method
    def __array__(self, dtype=None) -> NDArray:
        if len(self._base_array.free_symbols) > 0:
            if dtype is not None:
                msg = (
                    f"Array contains free symbols, therefore cannot cast to "
                    f"NumPy array of dtype {dtype}."
                )
                raise ValueError(msg)
            dtype = object
        else:
            dtype = self.dtype if dtype is None else dtype
        return np.array(self.tolist(), dtype)

    @property
    @_implements_numpy_ndarray_method
    def dtype(self) -> DTypeLike:
        if self._dtype is None:
            self._dtype = infer_array_dtype(self)
        return self._dtype

    @property
    @_implements_numpy_ndarray_method
    def shape(self) -> tuple[int, ...]:
        return self._base_array.shape

    def __str__(self) -> str:
        return f"SymbolicArray({self._base_array}, dtype={self._dtype})"

    def __repr__(self) -> str:
        return self.__str__()

    def _repr_latex_(self):
        return self._base_array._repr_latex_()

    def diff(self, *variables: Union[sympy.Expr, SymbolicArray]) -> SymbolicArray:
        """Compute derivative of array with respect to one or more arguments.

        Wrapper of :py:meth:`sympy.NDimArray.diff`.

        Args:
            *variables: One or more variables to differentiate with respect to.

        Returns:
            Array of derivatives.
        """
        return SymbolicArray(
            self._base_array.diff(*variables),
            dtype=self._dtype,
        )

    def subs(self, *args) -> SymbolicArray:
        """Substitute variables in an expression.

        Wrapper of :py:meth:`sympy.NDimArray.subs`.

        Args:
            *args: Either two arguments `(old, new)` corresponding to the old and new
                variables to subsitute for respectively, or an iterable of `(old, new)`
                pairs, with replacements processed in order given, or a dictionary
                mapping from old to new variables.

        Returns:
            Array with subsititutions applies.
        """
        return SymbolicArray(
            self._base_array.subs(*args),
            shape=self.shape,
            dtype=self._dtype,
        )

    def simplify(self, **kwargs) -> SymbolicArray:
        """Simplify symbolic expressions in array.

        Args:
            **kwargs: Any keyword arguments to :py:meth:`sympy.NDimArray.simplify`.

        Returns:
            Array with simplified symbolic expressions.
        """
        if self.shape == ():
            return SymbolicArray(
                self[()].simplify(**kwargs),
                shape=(),
                dtype=self._dtype,
            )
        else:
            return SymbolicArray(
                self._base_array.simplify(**kwargs),
                shape=self.shape,
                dtype=self._dtype,
            )

    @property
    def free_symbols(self) -> set[sympy.Symbol]:
        """Set of all free symbols in symbolic expressions defined in array."""
        return self._base_array.free_symbols

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __mul__(self, other):
        return self * other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __rmul__(self, other):
        return other * self

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __truediv__(self, other):
        return self / other

    __div__ = __truediv__

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __rtruediv__(self, other):
        return other / self

    __rdiv__ = __rtruediv__

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __floordiv__(self, other):
        return self // other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __rfloordiv__(self, other):
        return other // self

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __mod__(self, other):
        return self % other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __add__(self, other):
        return self + other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __radd__(self, other):
        return other + self

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __sub__(self, other):
        return self - other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __rsub__(self, other):
        return other - self

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __pow__(self, other):
        return self**other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __rpow__(self, other):
        return other**self

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __eq__(self, other):
        return self == other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __ne__(self, other):
        return self != other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __lt__(self, other):
        return self < other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __le__(self, other):
        return self <= other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __gt__(self, other):
        return self > other

    @binary_broadcasting_func
    @_implements_numpy_ndarray_method
    def __ge__(self, other):
        return self >= other

    @unary_elementwise_func
    @_implements_numpy_ndarray_method
    def __neg__(self):
        return -self

    @unary_elementwise_func
    @_implements_numpy_ndarray_method
    def __pos__(self):
        return self

    @unary_elementwise_func
    @_implements_numpy_ndarray_method
    def __abs__(self):
        return abs(self)

    @_implements_numpy_ndarray_method
    def __matmul__(self, other: ArrayLike) -> SymbolicArray:
        if not is_array(other):
            return NotImplemented
        return _matrix_multiply(self, as_symbolic_array(other))

    @_implements_numpy_ndarray_method
    def __rmatmul__(self, other: ArrayLike) -> SymbolicArray:
        if not is_array(other):
            return NotImplemented
        return _matrix_multiply(as_symbolic_array(other), self)

    @property
    @unary_elementwise_func
    @_implements_numpy_ndarray_method
    def real(self):
        return sympy.re(self)

    @property
    @unary_elementwise_func
    @_implements_numpy_ndarray_method
    def imag(self):
        return sympy.im(self)

    @property
    @_implements_numpy_ndarray_method
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    @_implements_numpy_ndarray_method
    def ndim(self) -> int:
        return len(self.shape)

    @_implements_numpy_ndarray_method
    def flatten(self) -> SymbolicArray:
        return self.reshape(self.size)

    @property
    @_implements_numpy_ndarray_method
    def flat(self) -> Iterator[sympy.Expr]:
        if self.shape == ():
            # SymPy arrays with shape () cannot be indexed so deal with this case
            # separately
            yield self[()]
        else:
            for idx in product(*(range(s) for s in self.shape)):
                yield self[idx]

    @_implements_numpy_ndarray_method
    def tolist(self) -> Union[list, sympy.Expr]:
        if self.shape == ():
            # SymPy arrays with shape () error when calling tolist method so deal with
            # this case separately
            return self[()]
        else:
            return self._base_array.tolist()

    @property
    @_implements_numpy_ndarray_method
    def T(self) -> SymbolicArray:  # noqa: N802
        return SymbolicArray(
            [
                self[tuple(indices[::-1])]
                for indices in product(*[range(s) for s in self.shape[::-1]])
            ],
            shape=self.shape[::-1],
            dtype=self._dtype,
        )

    @_implements_numpy_ndarray_method
    def transpose(self, axes: Optional[tuple[int, ...]] = None) -> SymbolicArray:
        if axes is None:
            return self.T
        else:
            return SymbolicArray(
                permutedims(self._base_array, axes),
                dtype=self._dtype,
            )

    @_implements_numpy_ndarray_method
    def reshape(self, shape: ShapeLike) -> SymbolicArray:
        return SymbolicArray(list(self.flat), shape=shape, dtype=self._dtype)

    @_implements_numpy_ndarray_method
    def sum(  # noqa: A003
        self,
        axis: Optional[Union[tuple, list, int]] = None,
    ) -> Union[SymbolicArray, sympy.Expr]:
        if self.shape == ():
            return self[()]
        elif axis is None:
            return sum(self.flat)
        elif isinstance(axis, (tuple, list, int)):
            return sum(_slice_iterator(self, axis))
        else:
            msg = f"Unrecognised axis type {type(axis)}."
            raise ValueError(msg)

    @_implements_numpy_ndarray_method
    def prod(
        self,
        axis: Optional[Union[tuple, list, int]] = None,
    ) -> Union[SymbolicArray, sympy.Expr]:
        if self.shape == ():
            return self[()]
        elif axis is None:
            return sympy.prod(self.flat)
        elif isinstance(axis, (tuple, list, int)):
            return sympy.prod(_slice_iterator(self, axis))
        else:
            msg = f"Unrecognised axis type {type(axis)}."
            raise ValueError(msg)


ArrayLike: TypeAlias = Union[SympyArray, NDArray, SymbolicArray]
