import numpy
import pytest
import sympy

import symnum
from symnum.array import SymbolicArray


def test_array_binary_operator_numeric(shape, binary_op, rng):
    numpy_array = rng.standard_normal(shape)
    symbolic_array = SymbolicArray(sympy.sympify(numpy_array))
    assert isinstance(symbolic_array, SymbolicArray)
    assert numpy.allclose(
        binary_op(numpy_array, numpy_array),
        numpy.array(binary_op(symbolic_array, symbolic_array), dtype=numpy.float64),
    )


def test_array_binary_operator_symbolic(symbolic_array_literal, binary_op):
    numpy_array = numpy.array(symbolic_array_literal)
    symbolic_array = SymbolicArray(symbolic_array_literal)
    assert isinstance(symbolic_array, SymbolicArray)
    assert numpy.all(
        binary_op(numpy_array, numpy_array)
        == binary_op(symbolic_array, symbolic_array),
    )


@pytest.mark.parametrize("method_name", ["sum", "prod"])
def test_reduction_method_numeric(method_name, shape, axis, rng):
    numpy_array = rng.standard_normal(shape)
    symbolic_array = SymbolicArray(sympy.sympify(numpy_array))
    assert numpy.allclose(
        getattr(numpy_array, method_name)(axis=axis),
        numpy.array(
            getattr(symbolic_array, method_name)(axis=axis),
            dtype=numpy.float64,
        ),
    )


@pytest.mark.parametrize("method_name", ["sum", "prod"])
def test_reduction_method_symbolic(
    symbolic_array_literal,
    method_name,
    axis,
):
    symbolic_array = symnum.numpy.array(symbolic_array_literal)
    numpy_array = numpy.array(symbolic_array_literal)
    assert numpy.all(
        getattr(numpy_array, method_name)(axis=axis)
        == getattr(symbolic_array, method_name)(axis=axis),
    )
