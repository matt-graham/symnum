"""Implementation of a subset of the NumPy API using SymPy primitives."""


from collections.abc import Iterable as _Iterable
from itertools import chain as _chain

import numpy as _np  # noqa: ICN001
import sympy as _sym
from sympy import S as _sym_singletons  # noqa: N811

from symnum.array import SymbolicArray as _SymbolicArray
from symnum.array import _slice_iterator
from symnum.array import binary_broadcasting_func as _binary_broadcasting_func
from symnum.array import is_sympy_array as _is_sympy_array
from symnum.array import unary_elementwise_func as _unary_elementwise_func

# Define mappings from objects in NumPy namespace to SymPy equivalents

_constants = {
    "pi": _sym_singletons.Pi,
    ("inf", "infty", "Inf", "Infinity", "PINF"): _sym_singletons.Infinity,
    ("nan", "NaN", "NAN"): _sym_singletons.NaN,
    "NINF": _sym_singletons.NegativeInfinity,
    "e": _sym_singletons.Exp1,
    "euler_gamma": _sym_singletons.EulerGamma,
    "newaxis": None,
}


def _angle(x, *, deg=False):
    if x == 0:
        return 0.0
    return _sym.arg(x) * 180 / _sym.pi if deg else _sym.arg(x)


_unary_elementwise_funcs = {
    "exp": _sym.exp,
    "expm1": lambda x: _sym.exp(x) - 1,
    "log": _sym.log,
    "log2": lambda x: _sym.log(x, 2),
    "log10": lambda x: _sym.log(x, 10),
    "log1p": lambda x: _sym.log(1 + x),
    "sin": _sym.sin,
    "cos": _sym.cos,
    "tan": _sym.tan,
    "arcsin": _sym.asin,
    "arccos": _sym.acos,
    "arctan": _sym.atan,
    "sinh": _sym.sinh,
    "cosh": _sym.cosh,
    "tanh": _sym.tanh,
    "arcsinh": _sym.asinh,
    "arccosh": _sym.acosh,
    "arctanh": _sym.atanh,
    "ceil": _sym.ceiling,
    "floor": _sym.floor,
    "sqrt": _sym.sqrt,
    ("abs", "absolute"): _sym.Abs,
    "sign": _sym.sign,
    "angle": _angle,
    ("conj", "conjugate"): _sym.conjugate,
    "real": _sym.re,
    "imag": _sym.im,
    "logical_not": _sym.Not,
    "isinf": lambda x: x
    in (
        _sym_singletons.Infinity,
        _sym_singletons.NegativeInfinity,
    ),
    "isposinf": lambda x: x == _sym_singletons.Infinity,
    "isneginf": lambda x: x == _sym_singletons.NegativeInfinity,
    "isnan": lambda x: x == _sym_singletons.NaN,
    "isreal": lambda x: x.is_real,
    "iscomplex": lambda x: not x.is_real,
    "isfinite": lambda x: x
    not in (
        _sym_singletons.Infinity,
        _sym_singletons.NegativeInfinity,
        _sym_singletons.NaN,
    ),
}


_binary_broadcasting_funcs = {
    "arctan2": _sym.atan2,
    "logical_and": _sym.And,
    "logical_or": _sym.Or,
    "logical_xor": _sym.Xor,
    "maximum": _sym.Max,
    "minimum": _sym.Min,
}


_binary_op_funcs = {
    "add": lambda x1, x2: x1 + x2,
    "subtract": lambda x1, x2: x1 - x2,
    "multiply": lambda x1, x2: x1 * x2,
    "divide": lambda x1, x2: x1 / x2,
    "power": lambda x1, x2: x1**x2,
    "matmul": lambda x1, x2: x1 @ x2,
}


def _wrap_numpy(numpy_name=None):
    def decorator(func):
        _numpy_name = func.__name__ if numpy_name is None else numpy_name
        func.__name__ = _numpy_name
        func.__doc__ = f"SymNum implementation of :py:obj:`numpy.{_numpy_name}`."
        return func

    return decorator


def _wrap_unary_elementwise_func(sympy_func, numpy_name):
    elementwise_func = _unary_elementwise_func(sympy_func, numpy_name, "")

    @_wrap_numpy(numpy_name)
    def wrapped(x, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            msg = f"Only first argument of {numpy_name} supported."
            raise NotImplementedError(msg)
        return elementwise_func(x)

    return wrapped


def _wrap_binary_broadcasting_func(sympy_func, numpy_name):
    broadcasting_func = _binary_broadcasting_func(sympy_func, numpy_name, "")

    @_wrap_numpy(numpy_name)
    def wrapped(x1, x2, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            msg = f"Only first two arguments of {numpy_name} supported."
            raise NotImplementedError(msg)
        return broadcasting_func(x1, x2)

    return wrapped


def _wrap_binary_op_func(op_func, numpy_name):
    @_wrap_numpy(numpy_name)
    def wrapped(x1, x2, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            msg = f"Only first two arguments of {numpy_name} supported."
            raise NotImplementedError(msg)
        return op_func(x1, x2)

    return wrapped


def _add_wrapped_funcs_to_namespace(func_mapping, namespace, wrapper):
    for name_or_names, sympy_func in func_mapping.items():
        if isinstance(name_or_names, tuple):
            for name in name_or_names:
                namespace[name] = wrapper(sympy_func, name)
        else:
            namespace[name_or_names] = wrapper(sympy_func, name_or_names)


def _populate_namespace(namespace):
    for name_or_names, val in _constants.items():
        if isinstance(name_or_names, tuple):
            for name in name_or_names:
                namespace[name] = val
        else:
            namespace[name_or_names] = val
    _add_wrapped_funcs_to_namespace(
        _unary_elementwise_funcs,
        namespace,
        _wrap_unary_elementwise_func,
    )
    _add_wrapped_funcs_to_namespace(
        _binary_broadcasting_funcs,
        namespace,
        _wrap_binary_broadcasting_func,
    )
    _add_wrapped_funcs_to_namespace(_binary_op_funcs, namespace, _wrap_binary_op_func)


_populate_namespace(globals())


# Array creation functions


def _flatten(iterable):
    """Recursively flatten nested iterables."""
    # Separately deal with shape () arrays as cannot be iterated over
    if hasattr(iterable, "shape") and iterable.shape == ():
        yield iterable[()]
    else:
        for el in iterable:
            if isinstance(el, _Iterable):
                yield from el
            else:
                yield el


def _contains_expr(iterable):
    return any(isinstance(el, _sym.Expr) for el in _flatten(iterable))


@_wrap_numpy()
def array(object_, dtype=None):  # noqa: D103
    if isinstance(object_, _SymbolicArray) or (
        isinstance(object_, _np.ndarray) and object_.dtype != object
    ):
        return object_
    elif (
        _is_sympy_array(object_)
        or isinstance(object_, _sym.Expr)
        or (isinstance(object_, _Iterable) and _contains_expr(object_))
    ):
        return _SymbolicArray(object_, dtype=dtype)
    else:
        return _np.array(object_, dtype)


@_wrap_numpy()
def eye(N, M=None, k=0):  # noqa: D103, N803
    M = N if M is None else M  # noqa: N806
    return _SymbolicArray(
        [1 if (j - i) == k else 0 for i in range(N) for j in range(M)],
        (N, M),
    )


@_wrap_numpy()
def identity(n):  # noqa: D103
    return eye(n, n, 0)


def _constant_array(val, shape):
    size = int(_np.prod(shape))
    return _SymbolicArray([val] * size, shape)


@_wrap_numpy()
def ones(shape):  # noqa: D103
    return _constant_array(1, shape)


@_wrap_numpy()
def zeros(shape):  # noqa: D103
    return _constant_array(0, shape)


@_wrap_numpy()
def full(shape, fill_value):  # noqa: D103
    return _constant_array(fill_value, shape)


# Array reductions


@_wrap_numpy()
def sum(a, axis=None):  # noqa: D103, A001
    return a.sum(axis)


@_wrap_numpy()
def prod(a, axis=None):  # noqa: D103
    return a.prod(axis)


# Array joining


@_wrap_numpy()
def concatenate(arrays, axis=0):  # noqa: D103
    if axis is None:
        # NumPy behaviour for axis=None case is to concatenate flattened arrays
        return array(list(_chain(*(a.flat for a in arrays))))
    for i in range(len(arrays)):
        if (axis >= 0 and axis > arrays[i].ndim - 1) or (
            axis < 0 and abs(axis) > arrays[i].ndim
        ):
            msg = (
                f"axis {axis} is out of bounds for array of dimension "
                f"{arrays[i].ndim}"
            )
            raise ValueError(msg)
    ndim = arrays[0].ndim
    for i in range(1, len(arrays)):
        if arrays[i].ndim != ndim:
            msg = (
                f"all the input arrays must have same number of dimensions, but"
                f" the array at index 0 has {arrays[0].ndim} dimension(s) and "
                f"the array at index {i} has {arrays[i].ndim} dimension(s)"
            )
            raise ValueError(msg)
        for d in set(range(arrays[0].ndim)) - {axis}:
            if arrays[0].shape[d] != arrays[i].shape[d]:
                msg = (
                    f"all the input array dimensions for the concatenation axis"
                    f" must match exactly, but along dimension {d}, the array "
                    f"at index 0 has size {arrays[0].shape[d]} and the array at"
                    f" index {i} has size {arrays[i].shape[d]}"
                )
                raise ValueError(msg)
    axis = ndim + axis if axis < 0 else axis
    array_slices = [
        slc.tolist() if hasattr(slc, "tolist") else slc
        for arr in arrays
        for slc in _slice_iterator(arr, axis)
    ]
    concatenated = array(array_slices)
    if axis != 0:
        concatenated = concatenated.transpose(
            (*tuple(range(1, axis + 1)), 0, *tuple(range(axis + 1, ndim))),
        )
    return concatenated
