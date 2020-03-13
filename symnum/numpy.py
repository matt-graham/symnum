"""Implementation of a subset of the NumPy API using SymPy primitives."""


from functools import wraps
from collections import Iterable
import sympy as sym
import numpy as np
from symnum.array import SymbolicArray, SYMPY_ARRAY_TYPES
from symnum.array import flatten as _flatten


def _contains_expr(iterable):
    return any([isinstance(el, sym.Expr) for el in _flatten(iterable)])


def array(object, dtype=None):
    if (isinstance(object, SYMPY_ARRAY_TYPES) or isinstance(object, sym.Expr)
        or (isinstance(object, Iterable) and _contains_expr(object))):
        return SymbolicArray(object, dtype=dtype)
    else:
        return np.array(object, dtype)

    
ones = np.ones
zeros = np.zeros
identity = np.identity
eye = np.eye
empty = np.empty
full = np.full


def elementwise_unary_func(sympy_func, numpy_func):
    
    @wraps(numpy_func)
    def wrapped_func(array):
        if isinstance(array, np.ndarray):
            return numpy_func(array)
        elif isinstance(array, SYMPY_ARRAY_TYPES):
            return array.applyfunc(sympy_func)
        elif isinstance(array, sym.Expr):
            return sympy_func(array)
        else:
            raise ValueError(f'Argument is of unknown type {type(array)}.')
    
    return wrapped_func


def sum(array, axis=None):
    return array.sum(axis)


sin = elementwise_unary_func(sym.sin, np.sin)
cos = elementwise_unary_func(sym.cos, np.cos)
tan = elementwise_unary_func(sym.tan, np.tan)
arcsin = elementwise_unary_func(sym.asin, np.arcsin)
arccos = elementwise_unary_func(sym.acos, np.arccos)
arctan = elementwise_unary_func(sym.atan, np.arctan)
sinh = elementwise_unary_func(sym.sinh, np.sinh)
cosh = elementwise_unary_func(sym.cosh, np.cosh)
tanh = elementwise_unary_func(sym.tanh, np.tanh)
arcsinh = elementwise_unary_func(sym.asinh, np.arcsinh)
arccosh = elementwise_unary_func(sym.acosh, np.arccosh)
arctanh = elementwise_unary_func(sym.atanh, np.arctanh)
exp = elementwise_unary_func(sym.exp, np.exp)
log = elementwise_unary_func(sym.log, np.log)
log2 = elementwise_unary_func(lambda x: sym.log(x, 2), np.log2)
log10 = elementwise_unary_func(lambda x: sym.log(x, 10), np.log10)