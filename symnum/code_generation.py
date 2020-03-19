"""Utility functions to generate NumPy function code."""

from collections import namedtuple
import sympy as sym
from sympy.printing.pycode import NumPyPrinter
import numpy
from symnum.array import (
    named_array, is_valid_shape, is_sympy_array, infer_dtype, SymbolicArray)


class FunctionExpression(sym.Expr):

    __slots__ = ['args', 'return_val']

    def __init__(self, args, return_val):
        self.args = args
        self.return_val = return_val


def _get_func_arg_names(func):
    if hasattr(func, '_arg_names'):
        return func._arg_names
    else:
        return func.__code__.co_varnames[:func.__code__.co_argcount]


def _construct_symbolic_arguments(*arg_shapes, arg_names=None):
    """Construct a tuple of symbolic array arguments with given shapes."""
    if arg_names is None:
        arg_names = [f'arg_{i}' for i in range(len(arg_shapes))]
    args = tuple(
        named_array(name, shape) if is_valid_shape(shape)
        else sym.Symbol(name)
        for shape, name in zip(arg_shapes, arg_names))
    return args


def numpify_func(sympy_func, *arg_shapes, **kwargs):
    """Generate a NumPy function from a SymPy symbolic array function.

    Args:
        sympy_func (Callable[..., Expr or SymbolicArray]): Function which takes
            one or more `SymbolicArray` as arguments and returns a symbolic
            scalar expression or `SymbolicArray` value.
        *arg_shapes: Variable length list of tuples or integers defining shapes
            of array arguments to `func`, e.g. if `func` takes two arguments
            `x` and `y` with `x` an array with shape `(2, 2)` and `y` an array
            with shape `(2, 4, 3)` the call signature would be of the form
            `numpify_func(func, (2, 2), (2, 4, 3))`.
        **kwargs: Any keyword arguments to the NumPy code generation function.

    Returns:
        numpy_func (Callable[..., scalar or ndarray]): Generated NumPy function
            which takes one or more `ndarray` arguments and return a scalar or
            `ndarray` value.
    """
    if len(arg_shapes) == 0 and hasattr(sympy_func, '_arg_shapes'):
        arg_shapes = sympy_func._arg_shapes
    arg_names = _get_func_arg_names(sympy_func)
    if len(arg_shapes) != len(arg_names):
        raise ValueError(
            'Shapes must be specified for all function arguments.')
    args = _construct_symbolic_arguments(*arg_shapes, arg_names=arg_names)
    if sympy_func.__name__ == '<lambda>':
        func_name = f'lambda_{id(sympy_func)}'
    else:
        func_name = f'{sympy_func.__name__}'
    numpy_func = generate_func(args, sympy_func(*args), func_name, **kwargs)
    numpy_func._sympy_func = sympy_func
    numpy_func._arg_shapes = arg_shapes
    numpy_func._arg_names = arg_names
    return numpy_func


def numpify(*arg_shapes, **kwargs):
    """Decorator to convert SymPy symbolic array function to a NumPy function.

    Args:
        *arg_shapes: Variable length list of tuples and integers defining
            shapes of array arguments to `func`, e.g. if `func` takes two
            arguments `x` and `y` with `x` an array with shape `(2, 2)` and `y`
            an array with shape `(2, 4, 3)` the call signature would be of the
            form `numpify((2, 2), (2, 4, 3))(func)`.
        **kwargs: Any keyword arguments to the NumPy code generation function.

    Returns:
        Callable[[Callable], Callable]: Decorator which takes a SymPy function
            of type `Callable[..., Expr or SymbolicArray]` which accepts one
            or more `SymbolicArray` as arguments and returns a symbolic scalar
            or `SymbolicArray` value, and returns a corresponding NumPy
            function of type `Callable[..., scalar or ndarray]` which accepts
            one or more `ndarray` arguments and returns a scalar or `ndarray`.
    """

    def decorator(func):
        return numpify_func(func, *arg_shapes, **kwargs)

    return decorator


def flatten_arrays(seq):
    """Flatten a sequence of arrays to a flat list."""
    flattened = []
    shape_size_and_dtypes = []
    for el in seq:
        if is_sympy_array(el):
            el = SymbolicArray(el)
            flattened += el.reshape(el.size).tolist()
            shape_size_and_dtypes.append((el.shape, el.size, el.dtype))
        else:
            flattened.append(el)
            shape_size_and_dtypes.append(())

    def unflatten(flattened):
        unflattened = []
        i = 0
        for s in shape_size_and_dtypes:
            if s == ():
                unflattened.append(flattened[i])
                i += 1
            else:
                shape, size, dtype = s
                unflattened.append(
                    SymbolicArray(flattened[i:i+size], shape, dtype))
                i += size
        return unflattened

    return flattened, unflatten


def to_tuple(lst):
    """Recursively convert nested lists to nested tuples."""
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


class TupleArrayPrinter(NumPyPrinter):
    """SymPy printer which uses nested tuples for array literals.

    Rather than printing array-like objects as numpy.array calls with a nested
    list argument (which is not numba compatible) a nested tuple argument is
    used instead.
    """

    def _print_arraylike(self, array):
        array_str = self._print(to_tuple(array.tolist()))
        return f"numpy.array({array_str})"

    _print_NDimArray = _print_arraylike
    _print_DenseNDimArray = _print_arraylike
    _print_ImmutableNDimArray = _print_arraylike
    _print_ImmutableDenseNDimArray = _print_arraylike
    _print_MatrixBase = _print_arraylike


def generate_input_list_string(inputs):
    """Generate a function argument string from a list of symbolic inputs."""
    input_strings = []
    for i in inputs:
        if isinstance(i, sym.Symbol):
            input_strings.append(str(i))
        elif hasattr(i, '_name'):
            input_strings.append(i._name)
        elif hasattr(i, 'free_symbols') and set(i) == i.free_symbols:
            input_strings.append(str(i.free_symbols.pop()).split('[')[0])
        else:
            raise ValueError(f'Input {i} not of valid type')
    return ", ".join(input_strings)


def generate_code(inputs, exprs, func_name='generated_function', printer=None):
    """Generate code for a Python function from symbolic expression(s)."""
    if printer is None:
        printer = TupleArrayPrinter()
    if not isinstance(exprs, list) and not isinstance(exprs, tuple):
        exprs = [exprs]
    elif isinstance(exprs, tuple):
        exprs = list(exprs)
    func_expr_args = {}
    ignore_set = set()
    for i, expr in enumerate(exprs):
        if isinstance(expr, FunctionExpression):
            func_expr_args[i] = expr.args
            for arg in expr.args:
                ignore_set.update(
                    (arg,) if isinstance(arg, sym.Symbol)
                    else arg.free_symbols)
            exprs[i] = expr.return_val
    flat_exprs, unflatten = flatten_arrays(exprs)
    intermediates, flat_outputs = sym.cse(
        flat_exprs, symbols=sym.numbered_symbols('_i'),
        optimizations='basic', ignore=ignore_set)
    outputs = unflatten(flat_outputs)
    code = f'def {func_name}({generate_input_list_string(inputs)}):\n    '
    code += '\n    '.join([f'{printer.doprint(i[0])} = {printer.doprint(i[1])}'
                           for i in intermediates])
    code += '\n    return (\n        '
    code += ',\n        '.join([
        (f'lambda {generate_input_list_string(func_expr_args[i])}: '
         if i in func_expr_args else '') +
        f'{printer.doprint(output)}' for i, output in enumerate(outputs)])
    code += '\n    )'
    return code


def generate_func(inputs, exprs, func_name='generated_function', printer=None,
                  numpy_module=None, exec_global=False):
    """Generate a Python function from symbolic expression(s)."""
    code = generate_code(inputs, exprs, func_name, printer)
    if exec_global:
        namespace = globals()
    else:
        namespace = {'numpy': numpy if numpy_module is None else numpy_module}
    exec(code, namespace)
    func = namespace[func_name]
    func.__doc__ = f'Automatically generated {func_name} function.\n\n{code}'
    return func
