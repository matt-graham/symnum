"""Implementation of a subset of the NumPy API using SymPy primitives."""


from collections import Iterable as _Iterable
import sympy as _sym
import numpy as _np
from symnum.array import (
    SymbolicArray as _SymbolicArray, is_sympy_array as _is_sympy_array,
    unary_elementwise_func as _unary_elementwise_func,
    binary_broadcasting_func as _binary_broadcasting_func,
    slice_iterator as _slice_iterator)
from sympy import S as _sym_singletons


# Define mappings from objects in NumPy namespace to SymPy equivalents

_constants = {
    'pi': _sym_singletons.Pi,
    ('inf', 'infty', 'INF', 'Infinity', 'PINF'): _sym_singletons.Infinity,
    ('nan', 'NaN', 'NAN'): _sym_singletons.NaN,
    'NINF': _sym_singletons.NegativeInfinity,
    'e': _sym_singletons.Exp1,
    'euler_gamma': _sym_singletons.EulerGamma,
    'newaxis': None,
}


_unary_elementwise_funcs = {
    'exp': _sym.exp,
    'expm1': lambda x: _sym.exp(x) - 1,
    'log': _sym.log,
    'log2': lambda x: _sym.log(x, 2),
    'log10': lambda x: _sym.log(x, 10),
    'log1p': lambda x: _sym.log(1 + x),
    'sin': _sym.sin,
    'cos': _sym.cos,
    'tan': _sym.tan,
    'arcsin': _sym.asin,
    'arccos': _sym.acos,
    'arctan': _sym.atan,
    'sinh': _sym.sinh,
    'cosh': _sym.cosh,
    'tanh': _sym.tanh,
    'arcsinh': _sym.asinh,
    'arccosh': _sym.acosh,
    'arctanh': _sym.atanh,
    'ceil': _sym.ceiling,
    'floor': _sym.floor,
    'sqrt': _sym.sqrt,
    ('abs', 'absolute'): _sym.Abs,
    'sign': _sym.sign,
    'angle': lambda x, deg=False: (
        _sym.arg(x) * 180 / _sym.pi if deg else _sym.arg(x)),
    ('conj', 'conjugate'): _sym.conjugate,
    'real': _sym.re,
    'imag': _sym.im,
    'logical_not': _sym.Not,
    'isinf': lambda x: x == _sym_S.Infinity or x == _sym_S.NegativeInfinity,
    'isposinf': lambda x: x == _sym_S.Infinity,
    'isneginf': lambda x:  x == _sym_S.NegativeInfinity,
    'isnan': lambda x: x == _sym_S.NaN,
    'isreal': lambda x: x.is_real,
    'iscomplex': lambda x: not x.is_real,
    'isfinite': lambda x: not (x == _sym_S.Infinity or
                               x == _sym_S.NegativeInfinity or x == _sym_S.NaN)
}


_binary_broadcasting_funcs = {
    'arctan2': _sym.atan2,
    'logical_and': _sym.And,
    'logical_or': _sym.Or,
    'logical_xor': _sym.Xor,
    'maximum': _sym.Max,
    'minimum': _sym.Min,
}


_binary_op_funcs = {
    'add': lambda x1, x2: x1 + x2,
    'subtract': lambda x1, x2: x1 - x2,
    'multiply': lambda x1, x2: x1 * x2,
    'divide': lambda x1, x2: x1 / x2,
    'power': lambda x1, x2: x1**x2,
    'matmul': lambda x1, x2: x1 @ x2,
}


def _wrap_numpy(numpy_name=None):

    def decorator(func):
        _numpy_name = func.__name__ if numpy_name is None else numpy_name
        try:
            func.__name__ = _numpy_name
            numpy_func_doc = getattr(_np, _numpy_name).__doc__
            if numpy_func_doc[0] == '\n':
                numpy_func_doc = numpy_func_doc[1:]
            func.__doc__ = f'symnum implementation of numpy.{_numpy_name}\n\n'
            func.__doc__ += numpy_func_doc
        finally:
            return func

    return decorator


def _wrap_unary_elementwise_func(sympy_func, numpy_name):

    elementwise_func = _unary_elementwise_func(sympy_func, numpy_name, '')

    @_wrap_numpy(numpy_name)
    def wrapped(x, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            raise NotImplementedError(
                f'Only first argument of {numpy_name} supported.')
        else:
            return elementwise_func(x)

    return wrapped


def _wrap_binary_broadcasting_func(sympy_func, numpy_name):

    broadcasting_func = _binary_broadcasting_func(sympy_func, numpy_name, '')

    @_wrap_numpy(numpy_name)
    def wrapped(x1, x2, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            raise NotImplementedError(
                f'Only first two arguments of {numpy_name} supported.')
        else:
            return broadcasting_func(x1, x2)

    return wrapped


def _wrap_binary_op_func(op_func, numpy_name):

    @_wrap_numpy(numpy_name)
    def wrapped(x1, x2, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            raise NotImplementedError(
                f'Only first two arguments of {numpy_name} supported.')
        else:
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
        _unary_elementwise_funcs, namespace, _wrap_unary_elementwise_func)
    _add_wrapped_funcs_to_namespace(
        _binary_broadcasting_funcs, namespace, _wrap_binary_broadcasting_func)
    _add_wrapped_funcs_to_namespace(
        _binary_op_funcs, namespace, _wrap_binary_op_func)


_populate_namespace(globals())


# Array creation functions

def _flatten(iterable):
    """Recursively flatten nested iterables to a list."""
    flattened = []
    for el in iterable:
        if isinstance(el, _Iterable):
            flattened.extend(_flatten(el))
        else:
            flattened.append(el)
    return flattened


def _contains_expr(iterable):
    return any([isinstance(el, _sym.Expr) for el in _flatten(iterable)])


@_wrap_numpy()
def array(object, dtype=None):
    if (_is_sympy_array(object) or isinstance(object, _sym.Expr)
            or (isinstance(object, _Iterable) and _contains_expr(object))):
        return _SymbolicArray(object, dtype=dtype)
    else:
        return _np.array(object, dtype)


@_wrap_numpy()
def eye(N, M=None, k=0):
    M = N if M is None else M
    return _SymbolicArray(
        [1 if (j - i) == k else 0 for i in range(N) for j in range(M)], (N, M))


@_wrap_numpy()
def identity(n):
    return eye(n, n, 0)


def _constant_array(val, shape):
    size = _np.prod(shape)
    return _SymbolicArray([val] * size, shape)


@_wrap_numpy()
def ones(shape):
    return _constant_array(1, shape)


@_wrap_numpy()
def zeros(shape):
    return _constant_array(0, shape)


@_wrap_numpy()
def full(shape, fill_value):
    return _constant_array(fill_value, shape)


# Array reductions


@_wrap_numpy()
def sum(a, axis=None):
    return a.sum(axis)


@_wrap_numpy()
def prod(a, axis=None):
    return a.prod(axis)


# Array joining


@_wrap_numpy()
def concatenate(arrays, axis=0):
    for i in range(len(arrays)):
        if (axis > 0 and axis > arrays[i].ndim - 1) or (
                axis < 0 and abs(axis) > arrays[i].ndim):
            raise ValueError(
                f'axis {axis} is out of bounds for array of dimension '
                f'{arrays[i].ndim}')
    ndim = arrays[0].ndim
    for i in range(1, len(arrays)):
        if arrays[i].ndim != ndim:
            raise ValueError(
                f'all the input arrays must have same number of dimensions, but'
                f' the array at index 0 has {arrays[0].ndim} dimension(s) and '
                f'the array at index {i} has {arrays[i].ndim} dimension(s)')
        for d in (set(range(arrays[0].ndim)) - set([axis])):
            if arrays[0].shape[d] != arrays[i].shape[d]:
                raise ValueError(
                    f'all the input array dimensions for the concatenation axis'
                    f' must match exactly, but along dimension {d}, the array '
                    f'at index 0 has size {arrays[0].shape[d]} and the array at'
                    f' index {i} has size {arrays[i].shape[d]}')
    array_slices = [slc for arr in arrays for slc in _slice_iterator(arr, axis)]
    concatenated = array(array_slices)
    if axis != 0:
        concatenated = concatenated.transpose(
            tuple(range(1, axis + 1)) + (0,) + tuple(range(axis + 1, ndim)))
    return concatenated
