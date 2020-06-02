"""Autograd-style functional differential operators for symbolic functions."""

from itertools import product as _product
import sympy as sym
from numpy import ndarray
from symnum.array import (
    named_array as _named_array, is_scalar as _is_scalar, SymbolicArray)
from symnum.codegen import FunctionExpression, _get_func_arg_names


def _get_sympy_func(func):
    if hasattr(func, '_sympy_func'):
        return func._sympy_func
    else:
        return func


def _wrap_derived(func, prefix=None, op='derivative'):

    def decorator(f):
        try:
            f.__name__ = (
                (f'{prefix}_' if prefix is not None else '') + func.__name__)
            f.__doc__ = (
                f'Automatically generated {op} of {func.__name__}.\n\n'
                f'Original docstring for {func.__name__}:\n\n{func.__doc__}')
            if hasattr(func, '_arg_shapes'):
                f._arg_shapes = func._arg_shapes
            f._arg_names = _get_func_arg_names(func)
        finally:
            return f

    return decorator


def _jacobian_transpose(jac, shape_val, shape_arg):
    n_dim_val = len(shape_val)
    n_dim_arg = len(shape_arg)
    n_dim = n_dim_arg + n_dim_val
    return jac.transpose(
        tuple(range(n_dim_arg, n_dim)) + tuple(range(n_dim_arg)))


def _generalised_dot(a, b, shape_out):
    return SymbolicArray([(a[indices] * b).sum() for indices in
                          _product(*[range(s) for s in shape_out])], shape_out)


def gradient(func, wrt=0, return_aux=False):
    """Generate a function to evaluate the gradient of a scalar-valued function.

    The passed function should take as arguments symbolic arrays and return a
    symbolic scalar, and likewise the returned function will take symbolic array
    arguments.

    Args:
        func (Callable[..., Scalar]): Function which takes one or more arrays as
            arguments and returns a scalar.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated gradient function returns both
            the gradient and value of `func` as a 2-tuple (True) or just the
            gradient (False).

    Returns:
        Callable[..., Union[Array, Tuple[Array, Scalar]]]: Generated gradient
            function.
    """

    @_wrap_derived(func, 'grad', 'gradient')
    def grad_func(*args):
        val = _get_sympy_func(func)(*args)
        if not _is_scalar(val) or (hasattr(val, 'shape') and val.shape != ()):
            raise ValueError(
                'gradient should only be used with scalar valued functions.')
        grad = sym.diff(val, args[wrt])
        return (grad, val) if return_aux else grad

    return grad_func


grad = gradient


def jacobian(func, wrt=0, return_aux=False):
    """Generate a function to evaluate the Jacobian of a function.

    The passed function should take as arguments and return symbolic array(s),
    and likewise the returned function will take symbolic array arguments.

    Args:
        func (Callable[..., Array]): Function which takes one or more arrays as
            arguments and returns an array.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated Jacobian function returns
            both the Jacobian and value of `func` as a 2-tuple (True) or just
            the Jacobian (False).

    Returns:
        Callable[..., Union[Array, Tuple[Array, Array]]]: Generated Jacobian
            function.
    """

    @_wrap_derived(func, 'jacob', 'Jacobian')
    def jacob_func(*args):
        val = _get_sympy_func(func)(*args)
        jacob = _jacobian_transpose(
            sym.diff(val, args[wrt]), val.shape, args[wrt].shape)
        return (jacob, val) if return_aux else jacob

    return jacob_func


def hessian(func, wrt=0, return_aux=False):
    """Generate a function to evaluate the Hessian of a scalar-valued function.

    The passed function should take as arguments symbolic arrays and return a
    symbolic scalar, and likewise the returned function will take symbolic array
    arguments.

    Args:
        func (Callable[..., Scalar]): Function which takes one or more arrays as
            arguments and returns a scalar.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated Hessian function returns the
            Hessian, gradient and value of `func` as a 3-tuple (True) or just
            the Hessian (False).

    Returns:
        Callable[..., Union[Array, Tuple[Array, Array, Scalar]]]: Generated
            Hessian function.
    """

    @_wrap_derived(func, 'hess', 'Hessian')
    def hess_func(*args):
        val = _get_sympy_func(func)(*args)
        if not _is_scalar(val) or (hasattr(val, 'shape') and val.shape != ()):
            raise ValueError(
                'hessian should only be used with scalar valued functions.')
        grad = sym.diff(val, args[wrt])
        hess = sym.diff(grad, args[wrt])
        return (hess, grad, val) if return_aux else hess

    return hess_func


def jacobian_vector_product(func, wrt=0, return_aux=False):
    """Generate an operator to evaluate Jacobian-vector-products for a function.

    The passed function should take as arguments and return symbolic array(s),
    and likewise the returned operator will take symbolic array arguments.

    For a single argument function `func`, `n`-dimensional input array `x` and
    `n`-dimensional 'vector' array `v` of the same shape as `x` then we have
    the following equivalence

        jacobian_vector_product(func)(x)(v) == (
            tensordot(jacobian(func)(x), v, n))

    where `tensordot` follows its NumPy semantics, i.e. `tensordot(a, b, n)`
    sums the products of components of `a` and `b` over the last `n` axes
    (dimensions) of `a` and first `n` dimensions of `b`.

    Args:
        func (Callable[..., Array]): Function which takes one or more arrays as
            arguments and returns an array.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated operator returns the
            Jacobian-vector-product function and value of `func` as a 2-tuple
            (True) or just the Jacobian-vector-product function (False).

    Returns:
        Callable[..., Union[Callable[Array, Array], Tuple[Callable, Array]]]:
            Generated Jacobian-vector-product operator.
    """

    @_wrap_derived(func, 'jvp', 'Jacobian-vector-product')
    def jvp_func(*args):
        val = _get_sympy_func(func)(*args)
        jacob = _jacobian_transpose(
            sym.diff(val, args[wrt]), val.shape, args[wrt].shape)
        v = _named_array('v', args[wrt].shape)
        jvp = _generalised_dot(jacob, v, val.shape)
        v_jvp = FunctionExpression((v,), jvp)
        return (v_jvp, val) if return_aux else v_jvp

    return jvp_func


def hessian_vector_product(func, wrt=0, return_aux=False):
    """Generate an operator to evaluate Hessian-vector-products for a function.

    The passed function should take as arguments symbolic arrays and return a
    symbolic scalar, and likewise the returned operator will take symbolic array
    arguments.

    For a single argument function `func`, `n`-dimensional input array `x` and
    `n`-dimensional 'vector' array `v` of the same shape as `x` then we have
    the following equivalence

        hessian_vector_product(func)(x)(v) == tensordot(hessian(func)(x), v, n)

    where `tensordot` follows its NumPy semantics, i.e. `tensordot(a, b, n)`
    sums the products of components of `a` and `b` over the last `n` axes
    (dimensions) of `a` and first `n` dimensions of `b`.

    Args:
        func (Callable[..., Scalar]): Function which takes one or more arrays as
            arguments and returns a scalar.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated operator returns the
            Hessian-vector-product function, gradient and value of `func` as a
            3-tuple (True) or just the Hessian-vector-product function (False).

    Returns:
        Callable[..., Union[Callable[Array, Array],
                            Tuple[Callable, Array, Scalar]]]:
            Generated Hessian-vector-product operator.
    """

    @_wrap_derived(func, 'hvp', 'Hessian-vector-product')
    def hvp_func(*args):
        hess, grad, val = hessian(func, wrt, return_aux=True)(*args)
        v = _named_array('v', args[wrt].shape)
        hvp = _generalised_dot(hess, v, args[wrt].shape)
        v_hvp = FunctionExpression((v,), hvp)
        return (v_hvp, grad, val) if return_aux else v_hvp

    return hvp_func


def vector_jacobian_product(func, wrt=0, return_aux=False):
    """Generate an operator to evaluate vector-Jacobian-products for a function.

    The passed function should take as arguments and return symbolic array(s),
    and likewise the returned function will act on symbolic arrays.

    For a single argument function `func`, input array `x` and `n`-dimensional
    'vector' array `v` of the same shape as `func(x)` then we have the
    following equivalence

        vector_jacobian_product(func)(x)(v) == (
            tensordot(v, jacobian(func)(x), n))

    where `tensordot` follows its NumPy semantics, i.e. `tensordot(a, b, n)`
    sums the products of components of `a` and `b` over the last `n` axes
    (dimensions) of `a` and first `n` dimensions of `b`.

    Args:
        func (Callable[..., Array]): Function which takes one or more arrays as
            arguments and returns an array.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated operator returns the
            vector-Jacobian-product function and value of `func` as a 2-tuple
            (True) or just the vector-Jacobian-product function (False).

    Returns:
        Callable[..., Union[Callable[Array, Array], Tuple[Callable, Array]]]:
            Generated vector-Jacobian-product operator.
    """

    @_wrap_derived(func, 'vjp', 'vector-Jacobian-product')
    def vjp_func(*args):
        val = _get_sympy_func(func)(*args)
        jacob_transposed = sym.diff(val, args[wrt])
        v = _named_array('v', val.shape)
        vjp = _generalised_dot(jacob_transposed, v, args[wrt].shape)
        v_vjp = FunctionExpression((v,), vjp)
        return (v_vjp, val) if return_aux else v_vjp

    return vjp_func


def matrix_hessian_product(func, wrt=0, return_aux=False):
    """Generate an operator to evaluate matrix-Hessian-products for a function.

    The passed function should take as arguments and return symbolic array(s),
    and likewise the returned function will act on symbolic arrays.

    For a single argument function `func`, `n`-dimensional input array `x` and
    `k + n`-dimensional 'matrix' array `m` with shape `func(x).shape + x.shape`
    then we have the following equivalence

        matrix_hessian_product(func)(x)(m) == (
            tensordot(m, jacobian(jacobian(func))(x), k + n))

    where `tensordot` follows its NumPy semantics, i.e. `tensordot(a, b, n)`
    sums the products of components of `a` and `b` over the last `n` axes
    (dimensions) of `a` and first `n` dimensions of `b`.

    Args:
        func (Callable[..., Array]): Function which takes one or more arrays as
            arguments and returns an array.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated operator return the
            matrix-Hessian-product function, Jacobian and value of `func` as a
            3-tuple (True) or just the matrix-Hessian-product function (False).

    Returns:
        Callable[..., Union[Callable[Array, Array],
                            Tuple[Callable, Array, Array]]]:
            Generated matrix-Hessian-product operator.
    """

    @_wrap_derived(func, 'mhp', 'matrix-Hessian-product')
    def mhp_func(*args):
        jac, val = jacobian(func, wrt, return_aux=True)(*args)
        hess = sym.diff(jac, args[wrt])
        m = _named_array('m', jac.shape)
        mhp = _generalised_dot(hess, m, args[wrt].shape)
        m_mhp = FunctionExpression((m,), mhp)
        return (m_mhp, jac, val) if return_aux else m_mhp

    return mhp_func


def matrix_tressian_product(func, wrt=0, return_aux=False):
    """Generate an operator to evaluate matrix-Tressian-products for a function.

    The passed function should take as arguments symbolic arrays and return a
    symbolic scalar, and likewise the returned operator will take symbolic array
    arguments.

    For a single argument function `func`, `n`-dimensional input array `x` and
    `2 * n`-dimensional 'matrix' array `m` of shape `x.shape + x.shape` then we
    have the following equivalence

        matrix_tressian_product(func)(x)(m) == (
            tensordot(jacobian(hessian(func))(x), 2 * n))

    where `tensordot` follows its NumPy semantics, i.e. `tensordot(a, b, n)`
    sums the products of components of `a` and `b` over the last `n` axes
    (dimensions) of `a` and first `n` dimensions of `b`.

    Args:
        func (Callable[..., Scalar]): Function which takes one or more arrays as
            arguments and returns a scalar.
        wrt (int): Index of argument to take derivatives with respect to.
        return_aux (bool): Whether the generated operator returns the
            matrix-Tressian-product function, Hessian, gradient and value of
            `func` as a 4-tuple (True) or just the matrix-Tressian-product
            function (False).

    Returns:
        Callable[..., Union[Callable[Array, Array],
                            Tuple[Callable, Array, Array, Scalar]]]:
            Generated matrix-Tressian-product operator.
    """

    @_wrap_derived(func, 'mtp', 'matrix-Tressian-product')
    def mtp_func(*args):
        hess, grad, val = hessian(func, wrt, return_aux=True)(*args)
        tress = sym.diff(hess, args[wrt])
        m = _named_array('m', hess.shape)
        mtp = _generalised_dot(tress, m, args[wrt].shape)
        m_mtp = FunctionExpression((m,), mtp)
        return (m_mtp, hess, grad, val) if return_aux else m_mtp

    return mtp_func
