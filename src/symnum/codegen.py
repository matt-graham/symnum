"""Utility functions to generate NumPy function code."""

from __future__ import annotations

import importlib.util as importlib_util
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy  # noqa: ICN001
import sympy
from sympy.printing.numpy import NumPyPrinter

from symnum.array import (
    SymbolicArray,
    as_symbolic_array,
    is_array,
    is_valid_shape,
    named_array,
)

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType
    from typing import Callable, Optional, Union

    from numpy.typing import NDArray
    from sympy.printing import Printer

    from symnum.array import ArrayLike, ScalarLike, ShapeLike


class FunctionExpression:
    """Function defined by a symbolic expression and set of symbolic arguments."""

    __slots__ = ["args", "return_val"]

    def __init__(
        self,
        args: tuple[Union[sympy.Expr, SymbolicArray], ...],
        return_val: Union[sympy.Expr, SymbolicArray],
    ):
        """
        Args:
            args: Symbolic arguments to function.
            return_val: Symbolic expression in arguments corresponding to function
                return value.
        """
        self.args = args
        self.return_val = return_val

    def __call__(
        self,
        *args: Union[sympy.Expr, SymbolicArray],
    ) -> Union[sympy.Expr, SymbolicArray]:
        """Evaluate symbolic function.

        Args:
            *args: Symbolic values to pass as positional arguments to function.

        Returns:
            Symbolic value corresponding to return value for function for passed
            symbolic arguments.
        """
        arg_subs = [
            (sym_old, sym_new)
            for arg_old, arg_new in zip(self.args, args)
            for sym_old, sym_new in zip(arg_old.flat, arg_new.flat)
        ]
        return self.return_val.subs(arg_subs)


def _get_func_arg_names(func: Callable):
    if hasattr(func, "_arg_names"):
        return func._arg_names
    else:
        return func.__code__.co_varnames[: func.__code__.co_argcount]


def _construct_symbolic_arguments(
    *arg_shapes: ShapeLike,
    arg_names: Optional[Iterable[str]] = None,
) -> tuple[SymbolicArray]:
    """Construct a tuple of symbolic array arguments with given shapes."""
    if arg_names is None:
        arg_names = [f"arg_{i}" for i in range(len(arg_shapes))]
    return tuple(
        named_array(name, shape) if is_valid_shape(shape) else sympy.Symbol(name)
        for shape, name in zip(arg_shapes, arg_names)
    )


def numpify_func(
    sympy_func: Callable[..., Union[sympy.Expr, SymbolicArray]],
    *arg_shapes: tuple[int],
    **kwargs,
) -> Callable[..., Union[ScalarLike, NDArray]]:
    """Generate a NumPy function from a SymPy symbolic array function.

    Args:
        sympy_func: Function which takes one or more :py:obj:`SymbolicArray` as
            arguments and returns a symbolic scalar expression or
            :py:obj:`SymbolicArray` value.
        *arg_shapes: Variable length list of tuples or integers defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y` with `x`
            an array with shape `(2, 2)` and `y` an array with shape `(2, 4, 3)` the
            call signature would be of the form `numpify_func(func, (2, 2), (2, 4, 3))`.
        **kwargs: Any keyword arguments to :py:func:`generate_func`.

    Returns:
        Generated NumPy function which takes one or more `numpy.ndarray` arguments and
        return a scalar or `numpy.ndarray` value.
    """
    if len(arg_shapes) == 0 and hasattr(sympy_func, "_arg_shapes"):
        arg_shapes = sympy_func._arg_shapes
    arg_names = _get_func_arg_names(sympy_func)
    if len(arg_shapes) != len(arg_names):
        msg = "Shapes must be specified for all function arguments."
        raise ValueError(msg)
    args = _construct_symbolic_arguments(*arg_shapes, arg_names=arg_names)
    if sympy_func.__name__ == "<lambda>":
        func_name = f"lambda_{id(sympy_func)}"
    else:
        func_name = f"{sympy_func.__name__}"
    numpy_func = generate_func(args, sympy_func(*args), func_name=func_name, **kwargs)
    numpy_func._sympy_func = sympy_func
    numpy_func._arg_shapes = arg_shapes
    numpy_func._arg_names = arg_names
    return numpy_func


def numpify(
    *arg_shapes: tuple[int],
    **kwargs,
) -> Callable[
    [Callable[..., Union[sympy.Expr, SymbolicArray]]],
    Callable[..., Union[ScalarLike, NDArray]],
]:
    """Decorator to convert SymPy symbolic array function to a NumPy function.

    Args:
        *arg_shapes: Variable length list of tuples and integers defining shapes of
            array arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `numpify((2, 2), (2, 4, 3))(func)`.
        **kwargs: Any keyword arguments to :py:func:`generate_func`.

    Returns:
        Decorator which takes a SymPy function of which accepts one or more
        :py:obj:`SymbolicArray` as arguments and returns a symbolic scalar or
        :py:obj:`SymbolicArray` value, and returns a corresponding NumPy function which
        accepts one or more :py:obj:`numpy.ndarray` arguments and returns a scalar or
        :py:obj:`numpy.ndarray`.
    """

    def decorator(func):
        return numpify_func(func, *arg_shapes, **kwargs)

    return decorator


def _flatten_arrays(
    seq: Iterable[ArrayLike],
) -> tuple[list[sympy.Expr], Callable[[list[sympy.Expr], list[SymbolicArray]]]]:
    """Flatten a sequence of arrays to a flat list."""
    flattened = []
    shape_size_and_dtypes = []
    for el in seq:
        if is_array(el):
            el = as_symbolic_array(el)  # noqa: PLW2901
            flattened += el.flatten().tolist()
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
                unflattened.append(SymbolicArray(flattened[i : i + size], shape, dtype))
                i += size
        return unflattened

    return flattened, unflatten


def _to_tuple(lst: list) -> tuple:
    """Recursively convert nested lists to nested tuples."""
    return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)


class TupleArrayPrinter(NumPyPrinter):
    """SymPy printer which uses nested tuples for array literals.

    Rather than printing array-like objects as numpy.array calls with a nested list
    argument (which is not numba compatible) a nested tuple argument is used instead.
    """

    def _print_arraylike(self, array: ArrayLike) -> str:
        array_str = self._print(_to_tuple(array.tolist()))
        return f"numpy.array({array_str})"

    _print_NDimArray = _print_arraylike  # noqa: N815
    _print_DenseNDimArray = _print_arraylike  # noqa: N815
    _print_ImmutableNDimArray = _print_arraylike  # noqa: N815
    _print_ImmutableDenseNDimArray = _print_arraylike  # noqa: N815
    _print_MatrixBase = _print_arraylike  # noqa: N815


def _generate_input_list_string(
    inputs: Iterable[Union[sympy.Expr, SymbolicArray]],
) -> str:
    """Generate a function argument string from a list of symbolic inputs."""
    input_strings = []
    for i in inputs:
        if isinstance(i, sympy.Symbol):
            input_strings.append(str(i))
        elif hasattr(i, "_name"):
            input_strings.append(i._name)
        elif hasattr(i, "free_symbols") and set(i) == i.free_symbols:
            input_strings.append(str(i.free_symbols.pop()).split("[")[0])
        else:
            msg = f"Input {i} not of valid type."
            raise ValueError(msg)
    return ", ".join(input_strings)


def _generate_code(
    inputs: Iterable[Union[sympy.Expr, SymbolicArray]],
    exprs: Iterable[Union[sympy.Expr, SymbolicArray]],
    func_name: str,
    *,
    printer: Optional[Printer] = None,
    simplify: bool = False,
) -> str:
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
                    (arg,) if isinstance(arg, sympy.Symbol) else arg.free_symbols,
                )
            exprs[i] = expr.return_val
    flat_exprs, unflatten = _flatten_arrays(exprs)
    if simplify:
        flat_exprs = [sympy.simplify(expr) for expr in flat_exprs]
    intermediates, flat_outputs = sympy.cse(
        flat_exprs,
        symbols=sympy.numbered_symbols("_i"),
        optimizations="basic",
        ignore=ignore_set,
    )
    outputs = unflatten(flat_outputs)
    code = f"def {func_name}({_generate_input_list_string(inputs)}):\n    "
    code += "\n    ".join(
        [f"{printer.doprint(i[0])} = {printer.doprint(i[1])}" for i in intermediates],
    )
    code += "\n    return (\n        "
    code += ",\n        ".join(
        [
            (
                f"lambda {_generate_input_list_string(func_expr_args[i])}: "
                if i in func_expr_args
                else ""
            )
            + f"{printer.doprint(output)}"
            for i, output in enumerate(outputs)
        ],
    )
    code += "\n    )"
    return code


def _create_temporary_module(prefix: str = "symnum_autogenerated_") -> ModuleType:
    """Create an empty Python module in a temporary file to act as namespace."""
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".py")
    temp_file_path = Path(temp_file.name)
    module_name = temp_file_path.stem
    spec = importlib_util.spec_from_file_location(module_name, temp_file_path)
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    return module


def generate_func(
    inputs: Iterable[Union[sympy.Expr, SymbolicArray]],
    exprs: Iterable[Union[sympy.Expr, SymbolicArray, FunctionExpression]],
    func_name: str,
    *,
    printer: Optional[Printer] = None,
    numpy_module: Optional[ModuleType] = None,
    namespace: Optional[dict] = None,
    jit: bool = False,
    simplify: bool = False,
):
    """Generate a Python function from symbolic expression(s).

    Args:
        inputs: List of symbolic inputs to use as function arguments.
        exprs: List of symbolic expressions to use as function return values.
        func_name: Name to define generated function with.
        printer: Instance of SymPy :py:class:`sympy.printing.printer.Printer` (sub)class
            which produces string representations of symbolic expressions to be used in
            code for generated function. Defaults to an instance of
            :py:class:`TupleArrayPrinter`, a subclass of
            :py:class:`sympy.printing.numpy.NumpyPrinter`, which prints `object`
            argument to :py:func:`numpy.array` as a nested tuple rather than a nested
            list, which retains compatibility with :py:mod:`numba` which only supports
            tuple array literals.
        numpy_module: Module implementing NumPy API to use in NumPy API calls in
            generated function. Defaults to :py:mod:`numpy`.
        namespace: Namespace to define generated function in. Default is to create a
            temporary module and define function in that namespace. Set
            :code:`namespace=globals()` to define in current global namespace.
        jit: If :code:`True` enables just-in-time compilation of the generated function
            using Numba (requires :py:mod:`numba` package to be installed in the current
            Python environment). Default is :code:`jit=False`.
        simplify: Whether to try to simplify symbolic expressions in :code:`exprs`
            before generating code.
    """
    code = _generate_code(inputs, exprs, func_name, printer=printer, simplify=simplify)
    if namespace is None:
        namespace = _create_temporary_module(f"symnum_{func_name}_module_").__dict__
    namespace["numpy"] = numpy if numpy_module is None else numpy_module
    namespace["math"] = math
    exec(code, namespace)  # noqa: S102
    func = namespace[func_name]
    func.__doc__ = f"Automatically generated {func_name} function.\n\n{code}"
    if NUMBA_AVAILABLE and jit:
        func = numba.njit(func)
    elif jit:
        warnings.warn(
            "Cannot just-in-time compile function as Numba is not installed.",
            stacklevel=2,
        )
    return func
