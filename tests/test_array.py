import numpy
import pytest
import sympy

import symnum
import symnum.array
from symnum.array import SymbolicArray


def test_array_binary_operator_numeric(shape, binary_op, rng):
    numpy_array = rng.standard_normal(shape)
    symbolic_array = SymbolicArray(sympy.sympify(numpy_array))
    assert numpy.allclose(
        binary_op(numpy_array, numpy_array),
        numpy.array(binary_op(symbolic_array, symbolic_array), dtype=numpy.float64),
    )


def test_array_binary_operator_symbolic(symbolic_array_literal, binary_op):
    numpy_array = numpy.array(symbolic_array_literal)
    symbolic_array = SymbolicArray(symbolic_array_literal)
    assert numpy.all(
        binary_op(numpy_array, numpy_array)
        == binary_op(symbolic_array, symbolic_array),
    )


def test_array_binary_comparison_operator_numeric(shape, binary_comparison_op, rng):
    numpy_array_1 = rng.standard_normal(shape)
    numpy_array_2 = rng.standard_normal(shape)
    symbolic_array_1 = SymbolicArray(sympy.sympify(numpy_array_1))
    symbolic_array_2 = SymbolicArray(sympy.sympify(numpy_array_2))
    assert numpy.allclose(
        binary_comparison_op(numpy_array_1, numpy_array_2),
        numpy.array(
            binary_comparison_op(symbolic_array_1, symbolic_array_2),
            dtype=bool,
        ),
    )
    assert numpy.allclose(
        binary_comparison_op(numpy_array_2, numpy_array_1),
        numpy.array(
            binary_comparison_op(symbolic_array_2, symbolic_array_1),
            dtype=bool,
        ),
    )


def test_array_binary_comparison_operator_symbolic(
    symbolic_array_literal, binary_comparison_op,
):
    symbolic_array = SymbolicArray(symbolic_array_literal)
    binary_comparison_array = binary_comparison_op(symbolic_array, symbolic_array)
    assert all(
        binary_comparison_op(s, s) == c
        for s, c in zip(symbolic_array.flat, binary_comparison_array.flat)
    )


def test_array_unary_operator_numeric(shape, unary_op, rng):
    numpy_array = rng.standard_normal(shape)
    symbolic_array = SymbolicArray(sympy.sympify(numpy_array))
    assert numpy.allclose(
        unary_op(numpy_array),
        numpy.array(unary_op(symbolic_array), dtype=numpy.float64),
    )


def test_array_unary_operator_symbolic(symbolic_array_literal, unary_op):
    numpy_array = numpy.array(symbolic_array_literal)
    symbolic_array = SymbolicArray(symbolic_array_literal)
    assert isinstance(symbolic_array, SymbolicArray)
    assert numpy.all(
        unary_op(numpy_array) == unary_op(symbolic_array),
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


def test_as_symbolic_array_sympy_array(symbolic_array_literal):
    sympy_array = sympy.NDimArray(symbolic_array_literal)
    symbolic_array = symnum.array.as_symbolic_array(sympy_array)
    assert isinstance(symbolic_array, SymbolicArray)
    assert all(
        s1 == s2
        for s1, s2 in zip(symbolic_array.flat, sympy_array.reshape(symbolic_array.size))
    )


def test_as_symbolic_array_numpy_array(shape, rng):
    numpy_array = rng.standard_normal(shape)
    symbolic_array = symnum.array.as_symbolic_array(numpy_array)
    assert isinstance(symbolic_array, SymbolicArray)
    assert numpy.all(
        numpy_array == numpy.array(symbolic_array, dtype=numpy.float64),
    )


def test_as_symbolic_array_identity(symbolic_array_literal):
    symbolic_array = SymbolicArray(symbolic_array_literal)
    assert symnum.array.as_symbolic_array(symbolic_array) is symbolic_array
