import operator

import numpy
import pytest
import sympy


def pytest_addoption(parser):
    parser.addoption("--seed", type=int, nargs="*", default=[837412487624871])


def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"))


@pytest.fixture()
def rng(seed):
    return numpy.random.default_rng(seed)


@pytest.fixture(
    params=[(), (1,), (1, 1), (2,), (2, 3), (3, 2, 1)],
    ids=lambda s: f"shape-{s}",
)
def shape(request):
    return request.param


@pytest.fixture(
    params=[
        ((1, 1), (1, 1)),
        ((2, 2), (2, 2)),
        ((3, 3), (3, 3)),
        ((2, 3), (3, 1)),
        ((2,), (2, 3)),
        ((3,), (3,)),
        ((3, 1), (1,)),
    ],
    ids=lambda shapes: f"matmul_shapes-{shapes}",
)
def matmul_shapes(request):
    return request.param


@pytest.fixture(params=[None, 0, -1], ids=lambda a: f"axis-{a}")
def axis(request):
    return request.param


SYMBOLIC_X, SYMBOLIC_Y = sympy.symbols("x y")
NON_ZERO_DIMENSIONAL_SYMBOLIC_ARRAY_LITERALS = [
    (SYMBOLIC_X,),
    (SYMBOLIC_X, SYMBOLIC_X),
    ((SYMBOLIC_X, SYMBOLIC_X),),
    (SYMBOLIC_X, SYMBOLIC_Y),
    ((SYMBOLIC_X, SYMBOLIC_Y), (SYMBOLIC_Y, SYMBOLIC_X)),
    ((SYMBOLIC_X, 1), (1, 1)),
]
SYMBOLIC_ARRAY_LITERALS = [SYMBOLIC_X, *NON_ZERO_DIMENSIONAL_SYMBOLIC_ARRAY_LITERALS]


@pytest.fixture(params=SYMBOLIC_ARRAY_LITERALS, ids=lambda s: str(s))
def symbolic_array_literal(request):
    return request.param


@pytest.fixture(
    params=NON_ZERO_DIMENSIONAL_SYMBOLIC_ARRAY_LITERALS, ids=lambda s: str(s)
)
def non_zero_symbolic_array_literal(request):
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
    ],
)
def binary_op(request):
    return request.param


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def binary_comparison_op(request):
    return request.param


@pytest.fixture(
    params=[
        abs,
        operator.pos,
        operator.neg,
    ],
)
def unary_op(request):
    return request.param
