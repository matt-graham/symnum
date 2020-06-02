"""Autograd-style functional differential operators for NumPy functions."""

import inspect as _inspect
import symnum.diffops.symbolic as _diffops
from symnum.codegen import numpify_func as _numpify_func


def _wrap_sympy_operator(sympy_operator):

    def numpy_operator(func, *arg_shapes, wrt=0, return_aux=False, **kwargs):
        return _numpify_func(
            sympy_operator(func, wrt, return_aux), *arg_shapes, **kwargs)

    docstring = sympy_operator.__doc__.replace('symbolic', 'NumPy')

    arg_shapes_docstring = f"""\
        *arg_shapes: Variable length list of tuples defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `{sympy_operator.__name__}(func, (2, 2), (2, 4, 3), ...)`.
    """

    docstring = docstring.replace(
        '        wrt', f'{arg_shapes_docstring}        wrt')

    kwargs_docstring = f"""\
        **kwargs: Any keyword arguments to the NumPy code generation function
            `symnum.codegen.generate_func`. Useful options include:
            jit: If `True` enables just-in-time compilation of the generated
                function using Numba (requires `numba` package to be installed
                in the current Python environment). Default is `jit=False`.
            namespace: Namespace to define generated function in. Default is to
                create a temporary module and define function in that namespace.
                Set `namespace=globals()` to define in current global namespace.
            numpy_module (ModuleType): Module implementing NumPy API to use in
                NumPy API calls in generated function. Defaults to `numpy`."""

    docstring = docstring.replace(
        '\n\n    Returns', f'\n{kwargs_docstring}\n\n    Returns')

    numpy_operator.__doc__ = docstring

    return numpy_operator


def _populate_namespace(namespace):
    for (name, operator) in _inspect.getmembers(_diffops, _inspect.isfunction):
        if name[0] != '_':
            namespace[name] = _wrap_sympy_operator(operator)


_populate_namespace(globals())
