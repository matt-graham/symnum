"""Autograd-style functional differential operators."""

from itertools import product
import sympy as sym
from symnum.array import named_array, SymbolicArray, is_scalar
from symnum.code_generation import (
    numpify_func, FunctionExpression, _get_func_arg_names)


__all__ = [
    'sympy_grad', 'sympy_jacobian', 'sympy_hessian',
    'sympy_jvp', 'sympy_vjp', 'sympy_mhp', 'sympy_mtp',
    'numpy_grad', 'numpy_jacobian', 'numpy_hessian',
    'numpy_jvp', 'numpy_vjp', 'numpy_mhp', 'numpy_mtp',
]


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


def sympy_grad(func, wrt=0, return_value=False):
    """Construct a function to compute the gradient of a scalar-valued function.

    The returned function takes as input and returns as output SymPy arrays.

    Args:
        func (Callable[..., Expr]): SymPy function which takes one or more
            symbolic arrays as arguments and returns a symbolic scalar
            expression.
        wrt (int): Index of argument to take derivatives with respect to.
        return_value (bool): Whether to return both gradient and value of
            `func` as a 2-tuple (true) or just gradient (False).

    Returns:
        Callable[..., SymbolicArray or Tuple[SymbolicArray, Symbol]]: Generated
            SymPy gradient function.
    """

    @_wrap_derived(func, 'grad', 'gradient')
    def grad_func(*args):
        val = _get_sympy_func(func)(*args)
        if not is_scalar(val) or (hasattr(val, 'shape') and val.shape != ()):
            raise ValueError(
                'grad should only be used with scalar valued functions.')
        grad = sym.diff(val, args[wrt])
        return (grad, val) if return_value else grad

    return grad_func


def _jacobian_transpose(jac, shape_val, shape_arg):
    n_dim_val = len(shape_val)
    n_dim_arg = len(shape_arg)
    n_dim = n_dim_arg + n_dim_val
    return jac.transpose(
        tuple(range(n_dim_arg, n_dim)) + tuple(range(n_dim_arg)))


def sympy_jacobian(func, wrt=0, return_value=False):
    """Construct a function to compute the Jacobian of an array-valued function.

    The returned function takes as input and returns as output SymPy arrays.

    Args:
        func (Callable[..., SymbolicArray]): SymPy function which takes one or
            more symbolic arrays as arguments and returns a symbolic array
            expression.
        wrt (int): Index of argument to take derivatives with respect to.
        return_value (bool): Whether to return both Jacobian and value of
            `func` as a 2-tuple (true) or just gradient (False).

    Returns:
        Callable[..., SymbolicArray or Tuple[SymbolicArray, SymbolicArray]]:
            Generated SymPy Jacobian function.
    """

    @_wrap_derived(func, 'jacob', 'Jacobian')
    def jacob_func(*args):
        val = _get_sympy_func(func)(*args)
        jacob = _jacobian_transpose(
            sym.diff(val, args[wrt]), val.shape, args[wrt].shape)
        return (jacob, val) if return_value else jacob

    return jacob_func


def sympy_hessian(func, wrt=0, return_grad_and_value=False):
    """Construct a function to compute the Hessian of a scalar-valued function.

    The returned function takes as input and returns as output SymPy arrays.

    Args:
        func (Callable[..., Expr]): SymPy function which takes one or more
            symbolic arrays as arguments and returns a scalar symbolic
            expression.
        wrt (int): Index of argument to take derivatives with respect to.
        return_grad_and_value (bool): Whether to return the Hessian, gradient
            and value of `func` as a 3-tuple (True) or just the Hessian (False)

    Returns:
        Callable[...,
                 SymbolicArray or Tuple[SymbolicArray, SymbolicArray, Symbol]]:
            Generated SymPy Hessian function.
    """

    @_wrap_derived(func, 'hess', 'Hessian')
    def hess_func(*args):
        val = _get_sympy_func(func)(*args)
        if not is_scalar(val) or (hasattr(val, 'shape') and val.shape != ()):
            raise ValueError(
                'hessian should only be used with scalar valued functions.')
        grad = sym.diff(val, args[wrt])
        hess = sym.diff(grad, args[wrt])
        return (hess, grad, val) if return_grad_and_value else hess

    return hess_func


def sympy_jvp(func, wrt=0, return_value=False):

    @_wrap_derived(func, 'jvp', 'Jacobian-vector-product')
    def jvp_func(*args):
        val = _get_sympy_func(func)(*args)
        jacob = _jacobian_transpose(
            sym.diff(val, args[wrt]), val.shape, args[wrt].shape)
        v = named_array('v', args[wrt].shape)
        jvp = SymbolicArray([
            (jacob[indices] * v).sum() for indices in
            product(*[range(s) for s in val.shape])], val.shape)
        v_jvp = FunctionExpression((v,), jvp)
        return (v_jvp, val) if return_value else v_jvp

    return jvp_func


def sympy_vjp(func, wrt=0, return_value=False):

    @_wrap_derived(func, 'vjp', 'vector-Jacobian-product')
    def vjp_func(*args):
        val = _get_sympy_func(func)(*args)
        jacob_transposed = sym.diff(val, args[wrt])
        v = named_array('v', val.shape)
        vjp = SymbolicArray([
            (jacob_transposed[indices] * v).sum() for indices in
            product(*[range(s) for s in args[wrt].shape])], args[wrt].shape)
        v_vjp = FunctionExpression((v,), vjp)
        return (v_vjp, val) if return_value else v_vjp

    return vjp_func


def sympy_mhp(func, wrt=0, return_jacobian_and_value=False):

    @_wrap_derived(func, 'mhp', 'matrix-Hessian-product')
    def mhp_func(*args):
        jac, val = sympy_jacobian(func, wrt, return_value=True)(*args)
        hess = sym.diff(jac, args[wrt])
        m = named_array('v', jac.shape)
        mhp = SymbolicArray([
            (hess[indices] * m).sum() for indices in
            product(*[range(s) for s in args[wrt].shape])], args[wrt].shape)
        m_mhp = FunctionExpression((m,), mhp)
        return (m_mhp, jac, val) if return_jacobian_and_value else m_mhp

    return mhp_func


def sympy_mtp(func, wrt=0, return_hessian_grad_and_value=False):

    @_wrap_derived(func, 'mtp', 'matrix-Tressian-product')
    def mtp_func(*args):
        hess, grad, val = sympy_hessian(
            func, wrt, return_grad_and_value=True)(*args)
        tress = sym.diff(hess, args[wrt])
        m = named_array('v', hess.shape)
        mtp = SymbolicArray([
            (tress[indices] * m).sum() for indices in
            product(*[range(s) for s in args[wrt].shape])], args[wrt].shape)
        m_mtp = FunctionExpression((m,), mtp)
        return (
            (m_mtp, hess, grad, val) if return_hessian_grad_and_value
            else m_mtp)

    return mtp_func


def numpy_grad(func, *arg_shapes, wrt=0, return_value=False, **kwargs):
    """Construct a function to compute the gradient of a scalar-valued function.

    The returned function takes as input and returns as output NumPy arrays.

    Args:
        func (Callable[..., Expr]): SymPy function which takes one or more
            symbolic arrays as arguments and returns a symbolic scalar
            expression.
        *arg_shapes: Variable length list of tuples defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `numpy_gradient(func, (2, 2), (2, 4, 3), ...)`.
        wrt (int): Index of argument to take derivatives with respect to.
        return_value (bool): Whether to return both the gradient and value of
            `func` as a 2-tuple (true) or just gradient (False).
        **kwargs: Any keyword arguments to the NumPy code generation function.

    Returns:
        Callable[..., ndarray or Tuple[ndarray, ndarray]]: Generated NumPy
            gradient function.
    """
    return numpify_func(
        sympy_grad(func, wrt, return_value), *arg_shapes, **kwargs)


def numpy_jacobian(func, *arg_shapes, wrt=0, return_value=False, **kwargs):
    """Construct a function to compute the Jacobian of an array-valued function.

    The returned function takes as input and returns as output NumPy arrays.

    Args:
        func (Callable[..., SymbolicArray]): SymPy function which takes one or
            more symbolic arrays as arguments and returns a symbolic array
            expression.
        *arg_shapes: Variable length list of tuples defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `numpy_jacobian(func, (2, 2), (2, 4, 3), ...)`.
        wrt (int): Index of argument to take derivatives with respect to.
        return_value (bool): Whether to return both the Jacobian and value of
            `func` as a 2-tuple (true) or just the Jacobian (False).
        **kwargs: Any keyword arguments to the NumPy code generation function.

    Returns:
        Callable[..., ndarray or Tuple[ndarray, scalar]]: Generated NumPy
            Jacobian function.
    """
    return numpify_func(
        sympy_jacobian(func, wrt, return_value), *arg_shapes, **kwargs)


def numpy_hessian(func, *arg_shapes, wrt=0, return_grad_and_value=False,
                  **kwargs):
    """Construct a function to compute the Hessian of a scalar-valued function.

    The returned function takes as input and returns as output NumPy arrays.

    Args:
        func (Callable[..., Expr]): SymPy function which takes one or more
            symbolic arrays as arguments and returns a symbolic scalar
            exprresion.
        *arg_shapes: Variable length list of tuples defining shapes of array
            arguments to `func`, e.g. if `func` takes two arguments `x` and `y`
            with `x` an array with shape `(2, 2)` and `y` an array with shape
            `(2, 4, 3)` the call signature would be of the form
            `numpy_hessian(func, (2, 2), (2, 4, 3), ...)`.
        wrt (int): Index of argument to take derivatives with respect to.
        return_grad_and_value (bool): Whether to return the Hessian, gradient
            and value of `func` as a 3-tuple (true) or just the Hessian (False)
        **kwargs: Any keyword arguments to the NumPy code generation function.

    Returns:
        Callable[..., ndarray or Tuple[ndarray, ndarray, scalar]]: Generated
            NumPy Hessian function.
    """
    return numpify_func(
        sympy_hessian(func, wrt, return_grad_and_value), *arg_shapes, **kwargs)


def numpy_jvp(func, *arg_shapes, wrt=0, return_value=False, **kwargs):
    return numpify_func(
        sympy_jvp(func, wrt, return_value), *arg_shapes, **kwargs)


def numpy_vjp(func, *arg_shapes, wrt=0, return_value=False, **kwargs):
    return numpify_func(
        sympy_vjp(func, wrt, return_value), *arg_shapes, **kwargs)


def numpy_mhp(func, *arg_shapes, wrt=0, return_jacobian_and_value=False,
              **kwargs):
    return numpify_func(
        sympy_mhp(func, wrt, return_jacobian_and_value), *arg_shapes, **kwargs)


def numpy_mtp(func, *arg_shapes, wrt=0, return_hessian_grad_and_value=False,
              **kwargs):
    return numpify_func(
        sympy_mtp(func, wrt, return_hessian_grad_and_value), *arg_shapes,
        **kwargs)
