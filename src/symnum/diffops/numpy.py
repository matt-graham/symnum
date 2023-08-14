"""Autograd-style functional differential operators for NumPy functions."""

from __future__ import annotations

import inspect as _inspect
from typing import TYPE_CHECKING

import symnum.diffops.symbolic as _diffops
from symnum.codegen import numpify_func as _numpify_func

if TYPE_CHECKING:
    from typing import Callable


def _wrap_sympy_operator(sympy_operator):
    def numpy_operator(
        func: Callable,
        *arg_shapes: tuple[int, ...],
        wrt: int = 0,
        return_aux: bool = False,
        **kwargs,
    ) -> Callable:
        return _numpify_func(
            sympy_operator(func, wrt=wrt, return_aux=return_aux),
            *arg_shapes,
            **kwargs,
        )

    numpy_operator.__annotations__.update(sympy_operator.__annotations__)

    docstring = sympy_operator.__doc__.replace("symbolic", "NumPy")

    arg_shapes_docstring = f"""\
        *arg_shapes: Variable length list of tuples defining shapes of array arguments
            to `func`, e.g. if `func` takes two arguments `x` and `y` with `x` an array
            with shape `(2, 2)` and `y` an array with shape `(2, 4, 3)` the call
            signature would be of the form
            :code:`{sympy_operator.__name__}(func, (2, 2), (2, 4, 3), ...)`."""

    docstring = docstring.replace("        wrt", f"{arg_shapes_docstring}\n        wrt")

    kwargs_docstring = """\
        **kwargs: Any keyword arguments to the NumPy code generation function
            :py:func:`symnum.codegen.generate_func`."""

    docstring = docstring.replace(
        "\n\n    Returns",
        f"\n{kwargs_docstring}\n\n    Returns",
    )

    numpy_operator.__doc__ = docstring

    return numpy_operator


def _populate_namespace(namespace):
    for name, operator in _inspect.getmembers(_diffops, _inspect.isfunction):
        if name[0] != "_":
            namespace[name] = _wrap_sympy_operator(operator)


_populate_namespace(globals())
