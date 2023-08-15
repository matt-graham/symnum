import numpy
import pytest
import sympy

import symnum
import symnum.numpy

REAL_VALUE_TEST_VALUES_AND_TYPE = ((-10.0, 0, 0.0, 0.5, 1, 1.0, 21.5), float)
NON_ZERO_REAL_VALUE_TEST_VALUES_AND_TYPE = ((-10.0, 0.5, 1, 1.0, 21.5), float)
NON_NEGATIVE_REAL_VALUE_TEST_VALUES_AND_TYPE = ((0, 0.0, 0.1, 0.5, 1, 1.0, 2.5), float)
POSITIVE_REAL_VALUE_TEST_VALUES_AND_TYPE = ((0.1, 0.5, 1, 1.0, 2.5), float)
SIGNED_UNIT_INTERVAL_TEST_VALUES_AND_TYPE = (
    (-1, -1.0, -0.5, 0, 0.0, 0.5, 1.0, 1),
    float,
)
COMPLEX_TEST_VALUES_AND_TYPE = (
    (-10.0, 0, 0.0, 0.5, 1, 1.0, 1j, 2.5j - 3.0, 0.2 + 0.5j),
    complex,
)
BOOLEAN_TEST_VALUES_AND_TYPE = ((True, False), bool)
INFINITE_TEST_VALUES_AND_TYPE = (
    (float("inf"), -float("inf"), float("nan"), 0.0),
    bool,
)

UNARY_ELEMENTWISE_FUNCTION_NAMES_TEST_VALUES_AND_TYPES = {
    "exp": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "expm1": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "log": POSITIVE_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "log2": POSITIVE_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "log10": POSITIVE_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "log1p": NON_NEGATIVE_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "sin": COMPLEX_TEST_VALUES_AND_TYPE,
    "cos": COMPLEX_TEST_VALUES_AND_TYPE,
    "tan": COMPLEX_TEST_VALUES_AND_TYPE,
    "arcsin": SIGNED_UNIT_INTERVAL_TEST_VALUES_AND_TYPE,
    "arccos": SIGNED_UNIT_INTERVAL_TEST_VALUES_AND_TYPE,
    "arctan": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "sinh": COMPLEX_TEST_VALUES_AND_TYPE,
    "cosh": COMPLEX_TEST_VALUES_AND_TYPE,
    "tanh": COMPLEX_TEST_VALUES_AND_TYPE,
    "arcsinh": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "arccosh": ((1.0, 2.5, 20.5), float),
    "arctanh": SIGNED_UNIT_INTERVAL_TEST_VALUES_AND_TYPE,
    "ceil": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "floor": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "sqrt": NON_NEGATIVE_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "absolute": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "sign": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "angle": COMPLEX_TEST_VALUES_AND_TYPE,
    "conjugate": COMPLEX_TEST_VALUES_AND_TYPE,
    "real": COMPLEX_TEST_VALUES_AND_TYPE,
    "imag": COMPLEX_TEST_VALUES_AND_TYPE,
    "logical_not": BOOLEAN_TEST_VALUES_AND_TYPE,
    "isinf": INFINITE_TEST_VALUES_AND_TYPE,
    "isposinf": INFINITE_TEST_VALUES_AND_TYPE,
    "isneginf": INFINITE_TEST_VALUES_AND_TYPE,
    "isnan": INFINITE_TEST_VALUES_AND_TYPE,
    "isreal": (COMPLEX_TEST_VALUES_AND_TYPE[0], bool),
    "iscomplex": (COMPLEX_TEST_VALUES_AND_TYPE[0], bool),
    "isfinite": INFINITE_TEST_VALUES_AND_TYPE,
}

BINARY_BROADCASTING_FUNCTION_NAMES_TEST_VALUES_AND_TYPES = {
    "arctan2": NON_ZERO_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "logical_and": BOOLEAN_TEST_VALUES_AND_TYPE,
    "logical_or": BOOLEAN_TEST_VALUES_AND_TYPE,
    "logical_xor": BOOLEAN_TEST_VALUES_AND_TYPE,
    "maximum": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "minimum": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "add": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "subtract": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "multiply": REAL_VALUE_TEST_VALUES_AND_TYPE,
    "divide": NON_ZERO_REAL_VALUE_TEST_VALUES_AND_TYPE,
    "power": NON_NEGATIVE_REAL_VALUE_TEST_VALUES_AND_TYPE,
}


@pytest.mark.parametrize(
    "constant_name",
    [
        "pi",
        "inf",
        "infty",
        "Inf",
        "Infinity",
        "PINF",
        "nan",
        "NaN",
        "NAN",
        "NINF",
        "e",
        "euler_gamma",
    ],
)
def test_constant_equality(constant_name):
    assert numpy.isclose(
        getattr(numpy, constant_name),
        float(getattr(symnum.numpy, constant_name)),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "function_name_test_values_and_type",
    list(UNARY_ELEMENTWISE_FUNCTION_NAMES_TEST_VALUES_AND_TYPES.items()),
    ids=list(UNARY_ELEMENTWISE_FUNCTION_NAMES_TEST_VALUES_AND_TYPES),
)
def test_unary_elementwise_function(function_name_test_values_and_type):
    function_name, (test_values, test_type) = function_name_test_values_and_type
    numpy_func = getattr(numpy, function_name)
    symnum_func = getattr(symnum.numpy, function_name)
    for val in test_values:
        assert numpy.isclose(
            numpy_func(val),
            test_type(symnum_func(sympy.sympify(val))),
        )
    numpy_array = numpy_func(numpy.array(test_values))
    symnum_array = symnum_func(symnum.numpy.array(test_values))
    assert numpy.allclose(
        numpy_array,
        numpy.array(symnum_array, dtype=test_type),
    )


@pytest.mark.parametrize(
    "function_name_test_values_and_type",
    list(BINARY_BROADCASTING_FUNCTION_NAMES_TEST_VALUES_AND_TYPES.items()),
    ids=list(BINARY_BROADCASTING_FUNCTION_NAMES_TEST_VALUES_AND_TYPES),
)
def test_binary_broadcasting_function(function_name_test_values_and_type):
    function_name, (test_values, test_type) = function_name_test_values_and_type
    numpy_func = getattr(numpy, function_name)
    symnum_func = getattr(symnum.numpy, function_name)
    for val_1 in test_values:
        for val_2 in test_values:
            assert numpy.isclose(
                numpy_func(val_1, val_2),
                test_type(symnum_func(sympy.sympify(val_1), sympy.sympify(val_2))),
            )
            for shape in ((1,), (5,), (1, 2), (2, 3, 2)):
                # broadcasting scalar and array
                assert numpy.allclose(
                    numpy_func(val_1, numpy.full(shape, val_2)),
                    numpy.array(
                        symnum_func(
                            sympy.sympify(val_1),
                            symnum.numpy.full(shape, sympy.sympify(val_2)),
                        ),
                        dtype=test_type,
                    ),
                )
                # broadcasting array and scalar
                assert numpy.allclose(
                    numpy_func(numpy.full(shape, val_1), val_2),
                    numpy.array(
                        symnum_func(
                            symnum.numpy.full(shape, sympy.sympify(val_1)),
                            sympy.sympify(val_2),
                        ),
                        dtype=test_type,
                    ),
                )
                # broadcasting array and same shaped array
                assert numpy.allclose(
                    numpy_func(numpy.full(shape, val_1), numpy.full(shape, val_2)),
                    numpy.array(
                        symnum_func(
                            symnum.numpy.full(shape, sympy.sympify(val_1)),
                            symnum.numpy.full(shape, sympy.sympify(val_2)),
                        ),
                        dtype=test_type,
                    ),
                )
                # broadcasting array and differently shaped array
                assert numpy.allclose(
                    numpy_func(
                        numpy.full(shape, val_1),
                        numpy.full((2, *shape), val_2),
                    ),
                    numpy.array(
                        symnum_func(
                            symnum.numpy.full(shape, sympy.sympify(val_1)),
                            symnum.numpy.full((2, *shape), sympy.sympify(val_2)),
                        ),
                        dtype=test_type,
                    ),
                )


def test_matmul(matmul_shapes, rng):
    shape_left, shape_right = matmul_shapes
    left_array = rng.standard_normal(shape_left)
    right_array = rng.standard_normal(shape_right)
    assert numpy.allclose(
        numpy.matmul(left_array, right_array),
        numpy.array(
            symnum.numpy.matmul(
                symnum.array.SymbolicArray(left_array),
                symnum.array.SymbolicArray(right_array),
            ),
            dtype=numpy.float64,
        ),
    )


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_identity(size):
    assert numpy.allclose(
        numpy.identity(size),
        numpy.array(
            symnum.numpy.identity(size),
            dtype=numpy.float64,
        ),
    )


@pytest.mark.parametrize("size_1", [1, 2, 3, 5])
@pytest.mark.parametrize("size_2", [1, 2, 3, 5])
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_eye(size_1, size_2, offset):
    assert numpy.allclose(
        numpy.eye(size_1, size_2, offset),
        numpy.array(
            symnum.numpy.eye(size_1, size_2, offset),
            dtype=numpy.float64,
        ),
    )


@pytest.mark.parametrize("function_name", ["zeros", "ones"])
def test_array_creation_function(function_name, shape):
    numpy_func = getattr(numpy, function_name)
    symnum_func = getattr(symnum.numpy, function_name)
    assert numpy.allclose(
        numpy_func(shape),
        numpy.array(symnum_func(shape), dtype=numpy.float64),
    )


@pytest.mark.parametrize("value", [-1.0, 0.5])
def test_full(shape, value):
    assert numpy.allclose(
        numpy.full(shape, value),
        numpy.array(symnum.numpy.full(shape, value), dtype=numpy.float64),
    )


def test_array_with_numpy_array(shape, rng):
    numpy_array = rng.standard_normal(shape)
    symnum_array = symnum.numpy.array(numpy_array)
    assert symnum_array is numpy_array
    assert isinstance(symnum.numpy.array(numpy_array.tolist()), numpy.ndarray)


def test_array_with_symbolic_array(shape, rng):
    symbolic_array = symnum.array.SymbolicArray(rng.standard_normal(shape))
    assert symnum.numpy.array(symbolic_array) is symbolic_array


def test_array_with_sympy_array(shape, rng):
    sympy_array = sympy.sympify(rng.standard_normal(shape))
    assert isinstance(symnum.numpy.array(sympy_array), symnum.array.SymbolicArray)


def test_array_with_nested_symbolic_iterable(symbolic_array_literal):
    symbolic_array = symnum.numpy.array(symbolic_array_literal)
    numpy_array = numpy.array(symbolic_array_literal)
    assert isinstance(symbolic_array, symnum.array.SymbolicArray)
    assert symbolic_array.shape == numpy_array.shape
    assert numpy.all(numpy_array == symbolic_array)


@pytest.mark.parametrize("function_name", ["sum", "prod"])
def test_reduction_numeric(function_name, shape, axis, rng):
    numpy_func = getattr(numpy, function_name)
    symnum_func = getattr(symnum.numpy, function_name)
    array = rng.standard_normal(shape)
    assert numpy.allclose(numpy_func(array, axis=axis), symnum_func(array, axis=axis))


@pytest.mark.parametrize("function_name", ["sum", "prod"])
def test_reduction_symbolic(
    symbolic_array_literal,
    function_name,
    axis,
):
    numpy_func = getattr(numpy, function_name)
    symnum_func = getattr(symnum.numpy, function_name)
    symbolic_array = symnum.numpy.array(symbolic_array_literal)
    numpy_array = numpy.array(symbolic_array_literal)
    assert numpy.all(
        numpy_func(numpy_array, axis=axis) == symnum_func(symbolic_array, axis=axis),
    )


@pytest.mark.parametrize("repeats", [2, 3])
def test_concatenate(non_zero_symbolic_array_literal, axis, repeats):
    symbolic_array = symnum.numpy.array(non_zero_symbolic_array_literal)
    numpy_array = numpy.array(non_zero_symbolic_array_literal)
    concatenated_symbolic_array = symnum.numpy.concatenate(
        (symbolic_array,) * repeats,
        axis=axis,
    )
    concatenated_numpy_array = numpy.concatenate((numpy_array,) * repeats, axis=axis)
    assert concatenated_numpy_array.shape == concatenated_symbolic_array.shape
    assert numpy.all(concatenated_numpy_array == concatenated_symbolic_array)
