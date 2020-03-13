"""Utility functions to generate NumPy function code."""

from collections import namedtuple
import sympy as sym
from sympy.printing.pycode import NumPyPrinter
import numpy
from symnum.array import named_array


class FunctionExpression(sym.Expr):
    
    __slots__ = ['args', 'return_val']
    
    def __init__(self, args, return_val):
        self.args = args
        self.return_val = return_val


def _construct_symbolic_arguments(*arg_shapes, arg_names=None):
    """Construct a tuple of symbolic array arguments with given shapes."""
    if arg_names is not None and len(arg_shapes) != len(arg_names):
        raise ValueError('Shapes must be specified for all function arguments.')
    elif arg_names is None:
        arg_names = [f'arg_{i}' for i in len(arg_shapes)]
    args = tuple(
        named_array(name, shape) if isinstance(shape, (int, tuple))
        else sym.Symbol(name) 
        for shape, name in zip(arg_shapes, arg_names))
    return args


def numpy_func(func, *arg_shapes, arg_names=None, func_name_prefix='', **kwargs):
    """Generate a NumPy function from a SymPy symbolic array function.
    
    Args:
        func (Callable[..., Expr or SymbolicArray]): Function which takes one
            or more `SymbolicArray` as arguments and returns a symbolic scalar
            expression or `SymbolicArray` value.
        arg_shapes (Iterable[Tuple]): List of tuples defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `numpy_func(func, (2, 2), (2, 4, 3), ...)`.
        func_name_prefix (string): Prefix to prepend on to name of `func` to
            when setting name of generated function.
        **kwargs: Any keyword arguments to the NumPy code generation function. 
        
    Returns:
        Callable[..., scalar or ndarray]: Generated NumPy function.
    """
    args = _construct_symbolic_arguments(*arg_shapes, arg_names=arg_names)
    if func.__name__ == '<lambda>':
        func_name = f'{func_name_prefix}lambda_{id(func)}'
    else:
        func_name = f'{func_name_prefix}{func.__name__}'
    return generate_func(args, func(*args), func_name, **kwargs)


def numpify(*arg_shapes, arg_names=None, **kwargs):
    """Decorator to convert NumPy function to  a SymPy symbolic array function.
    
    Args:
        *arg_shapes: Variable length list of tuples defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `numpy_func(func, (2, 2), (2, 4, 3), ...)`.
        **kwargs: Any keyword arguments to the NumPy code generation function. 
        
    Returns:
        Callable[[Callable], Callable]: Decorator which takes a SymPy function
            of type `Callable[..., Symbol or SymbolicArray]` which accepts one
            or more `SymbolicArray` as arguments and returns a symbolic scalar
            or `SymbolicArray` value, and returns a corresponding NumPy function 
            of type `Callable[..., scalar or ndarray]` which accepts one or more
            NumPy array arguments and returns a scalar or NumPy array.
    """
    
    def decorator(func):
        _arg_names = (
            func.__code__.co_varnames if arg_names is None else arg_names)
        return numpy_func(func, *arg_shapes, arg_names=_arg_names, **kwargs)
    
    return decorator


def flatten_arrays(seq):
    """Flatten a sequence of arrays to a flat list."""
    flattened = []
    shapes = []
    for el in seq:
        if isinstance(el, sym.Array):
            size = numpy.prod(el.shape)
            flattened += el.reshape(size).tolist()
            shapes.append((el.shape, size))
        else:
            flattened.append(el)
            shapes.append(())
    def unflatten(flattened):
        unflattened = []
        i = 0
        for s in shapes:
            if s == ():
                unflattened.append(flattened[i])
                i += 1
            else:
                shape, size = s
                unflattened.append(sym.Array(flattened[i:i+size], shape))
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
    
    def _print_arraylike(self, expr):
        exp_str = self._print(to_tuple(expr.tolist()))
        return f'numpy.array({exp_str}, dtype=numpy.float64)'
    
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
                    (arg,) if isinstance(arg, sym.Symbol) else arg.free_symbols)
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
        exec(code, globals())
        func = globals()[func_name]
    else:
        namespace = {'numpy': numpy if numpy_module is None else numpy_module}
        exec(code, namespace)
    func = namespace[func_name]
    func.__doc__ = f'Automatically generated {func_name} function.\n\n{code}'
    return func


def delayed_generate_func(*args, **kwargs):
    
    generated_func = None
    
    def wrapped(*inner_args, **inner_kwargs):
        nonlocal generated_func
        if generated_func is None:
            generated_func = generate_func(*args, **kwargs)
        return generated_func(*inner_args, **inner_kwargs)
    
    return wrapped