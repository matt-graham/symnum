import numpy as np
import pytest

import symnum
import symnum.numpy as snp

N_POINTS_TO_TEST = 5


def torus_function_and_derivatives(_):
    toroidal_rad = 1.0
    poloidal_rad = 0.5

    @symnum.numpify(3)
    def function(q):
        x, y, z = q
        return snp.array(
            [((x**2 + y**2) ** 0.5 - toroidal_rad) ** 2 + z**2 - poloidal_rad**2],
        )

    def jacobian_function(q):
        x, y, z = q
        r = (x**2 + y**2) ** 0.5
        return np.array(
            [[2 * x * (r - toroidal_rad) / r, 2 * y * (r - toroidal_rad) / r, 2 * z]],
        )

    def matrix_hessian_product_function(q):
        x, y, z = q
        r = (x**2 + y**2) ** 0.5
        r_cubed = r**3
        return lambda m: np.array(
            [
                2 * (toroidal_rad / r_cubed) * (m[0, 0] * x**2 + m[0, 1] * x * y)
                + 2 * m[0, 0] * (1 - toroidal_rad / r),
                2 * (toroidal_rad / r_cubed) * (m[0, 1] * y**2 + m[0, 0] * x * y)
                + 2 * m[0, 1] * (1 - toroidal_rad / r),
                2 * m[0, 2],
            ],
        )

    return {
        "function": function,
        "jacobian_function": jacobian_function,
        "matrix_hessian_product_function": matrix_hessian_product_function,
    }


def linear_function_and_derivatives(_):

    constr_matrix = np.array([[1.0, -1.0, 2.0, 3.0], [-3.0, 2.0, 0.0, 5.0]])

    @symnum.numpify(4)
    def function(q):
        return constr_matrix @ q

    def jacobian_function(_):
        return constr_matrix

    def matrix_hessian_product_function(_):
        return lambda _: np.zeros(constr_matrix.shape[1])

    return {
        "function": function,
        "jacobian_function": jacobian_function,
        "matrix_hessian_product_function": matrix_hessian_product_function,
    }


def quadratic_form_function_and_derivatives(_):

    matrix = np.array([[1.3, -0.2], [-0.2, 2.5]])

    @symnum.numpify(2)
    def quadratic_form(q):
        return q @ matrix @ q / 2

    def gradient_quadratic_form(q):
        return matrix @ q

    def hessian_quadratic_form(_):
        return matrix

    def matrix_tressian_product_quadratic_form(_):
        return lambda _: np.zeros(matrix.shape[0])

    return {
        "function": quadratic_form,
        "gradient_function": gradient_quadratic_form,
        "hessian_function": hessian_quadratic_form,
        "matrix_tressian_product_function": matrix_tressian_product_quadratic_form,
    }


def cubic_function_and_derivatives(dim_q):

    @symnum.numpify(dim_q)
    def cubic(q):
        return (q**3).sum() / 6

    def gradient_cubic(q):
        return q**2 / 2

    def hessian_cubic(q):
        return np.diag(q)

    def matrix_tressian_product_cubic(_):
        return lambda m: m.diagonal()

    return {
        "function": cubic,
        "gradient_function": gradient_cubic,
        "hessian_function": hessian_cubic,
        "matrix_tressian_product_function": matrix_tressian_product_cubic,
    }


def quartic_function_and_derivatives(dim_q):

    @symnum.numpify(dim_q)
    def quartic(q):
        return (q**4).sum() / 24

    def gradient_quartic(q):
        return q**3 / 6

    def hessian_quartic(q):
        return np.diag(q**2 / 2)

    def matrix_tressian_product_quartic(q):
        return lambda m: m.diagonal() * q

    return {
        "function": quartic,
        "gradient_function": gradient_quartic,
        "hessian_function": hessian_quartic,
        "matrix_tressian_product_function": matrix_tressian_product_quartic,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


VECTOR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS = [
    (torus_function_and_derivatives, 1, 3),
    (linear_function_and_derivatives, 2, 4),
]

SCALAR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS = [
    (quadratic_form_function_and_derivatives, 2),
    (cubic_function_and_derivatives, 1),
    (cubic_function_and_derivatives, 3),
    (quartic_function_and_derivatives, 2),
]


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    SCALAR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_gradient(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    gradient_function = symnum.gradient(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            gradient, value = gradient_function(q)
        else:
            gradient = gradient_function(q)
        assert np.allclose(function_and_derivatives["gradient_function"](q), gradient)
        if return_aux:
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    SCALAR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_hessian(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    hessian_function = symnum.hessian(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            hessian, gradient, value = hessian_function(q)
        else:
            hessian = hessian_function(q)
        assert np.allclose(function_and_derivatives["hessian_function"](q), hessian)
        if return_aux:
            assert np.allclose(
                function_and_derivatives["gradient_function"](q),
                gradient,
            )
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    SCALAR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_matrix_tressian_product(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    matrix_tressian_product_function = symnum.matrix_tressian_product(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            mtp, hessian, gradient, value = matrix_tressian_product_function(q)
        else:
            mtp = matrix_tressian_product_function(q)
        m = rng.standard_normal((dim_q, dim_q))
        assert np.allclose(
            function_and_derivatives["matrix_tressian_product_function"](q)(m),
            mtp(m),
        )
        if return_aux:
            assert np.allclose(function_and_derivatives["hessian_function"](q), hessian)
            assert np.allclose(
                function_and_derivatives["gradient_function"](q),
                gradient,
            )
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    VECTOR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_jacobian(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, _, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    jacobian_function = symnum.jacobian(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            jacobian, value = jacobian_function(q)
        else:
            jacobian = jacobian_function(q)
        assert np.allclose(function_and_derivatives["jacobian_function"](q), jacobian)
        if return_aux:
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    VECTOR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_matrix_hessian_product(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, dim_v, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    matrix_hessian_product_function = symnum.matrix_hessian_product(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            mhp, jacobian, value = matrix_hessian_product_function(q)
        else:
            mhp = matrix_hessian_product_function(q)
        m = rng.standard_normal((dim_v, dim_q))
        assert np.allclose(
            function_and_derivatives["matrix_hessian_product_function"](q)(m),
            mhp(m),
        )
        if return_aux:
            assert np.allclose(
                function_and_derivatives["jacobian_function"](q),
                jacobian,
            )
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    VECTOR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_vector_jacobian_product(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, dim_v, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    vector_jacobian_product_function = symnum.vector_jacobian_product(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            vjp, value = vector_jacobian_product_function(q)
        else:
            vjp = vector_jacobian_product_function(q)
        v = rng.standard_normal(dim_v)
        assert np.allclose(v @ function_and_derivatives["jacobian_function"](q), vjp(v))
        if return_aux:
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    VECTOR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_jacobian_vector_product(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, _, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    jacobian_vector_product_function = symnum.jacobian_vector_product(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            jvp, value = jacobian_vector_product_function(q)
        else:
            jvp = jacobian_vector_product_function(q)
        v = rng.standard_normal(q.shape)
        assert np.allclose(function_and_derivatives["jacobian_function"](q) @ v, jvp(v))
        if return_aux:
            assert np.allclose(function_and_derivatives["function"](q), value)


@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    SCALAR_TEST_FUNCTIONS_AND_DERIVATIVES_AND_DIMS,
    ids=lambda p: p[0].__name__,
)
@pytest.mark.parametrize("return_aux", [True, False])
def test_hessian_vector_product(function_and_derivatives_and_dim, return_aux, rng):
    construct_function_and_derivatives, dim_q = function_and_derivatives_and_dim
    function_and_derivatives = construct_function_and_derivatives(dim_q)
    hessian_vector_product_function = symnum.hessian_vector_product(
        function_and_derivatives["function"],
        return_aux=return_aux,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        if return_aux:
            hvp, grad, value = hessian_vector_product_function(q)
        else:
            hvp = hessian_vector_product_function(q)
        v = rng.standard_normal(q.shape)
        assert np.allclose(function_and_derivatives["hessian_function"](q) @ v, hvp(v))
        if return_aux:
            assert np.allclose(function_and_derivatives["gradient_function"](q), grad)
            assert np.allclose(function_and_derivatives["function"](q), value)
