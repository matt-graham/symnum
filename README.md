
<div style="text-align: center;" align="center">

<img src="https://raw.githubusercontent.com/matt-graham/symnum/main/images/logomark-dark-background.svg" alt="SymNum logo" width="120"/>

<h1>SymNum</h1>

<a href="https://badge.fury.io/py/symnum">
  <img src="https://badge.fury.io/py/symnum.svg" alt="PyPI version"/>
</a>
<a href="https://zenodo.org/badge/latestdoi/247063908">
  <img src="https://zenodo.org/badge/247063908.svg" alt="Zenodo DOI badge">
</a>
<a href="https://github.com/matt-graham/symnum/actions/workflows/tests.yml">
  <img src="https://github.com/matt-graham/symnum/actions/workflows/tests.yml/badge.svg" alt="Test status" />
</a>
<a href="https://matt-graham.github.io/symnum">
  <img src="https://github.com/matt-graham/symnum/actions/workflows/docs.yml/badge.svg" alt="Documentation status" />
</a>
</div>


## What is SymNum?

SymNum is a Python package that acts a bridge between
[NumPy](https://numpy.org/) and [SymPy](https://www.sympy.org/), providing a
NumPy-like interface that can be used to symbolically define functions which
take arrays as arguments and return arrays or scalars as values. A series of
[Autograd](https://github.com/HIPS/autograd) style functional differential
operators are also provided to construct derivatives of symbolic functions,
with the option to generate NumPy code to numerically evaluate these derivative
functions.

## Why use SymNum instead of Autograd or JAX?

SymNum is intended for use in generating the derivatives of 'simple' functions
which **compose a relatively small number of operations** and act on **small
array inputs**. By reducing interpreter overheads it can produce code which is
cheaper to evaluate than corresponding
[Autograd](https://github.com/HIPS/autograd)  or
[JAX](https://github.com/google/jax) functions (including those using  [JIT
compilation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#Using-jit-to-speed-up-functions))
in such cases, and which can be serialised with the inbuilt Python `pickle`
library allowing use for example in libraries which use `multiprocessing` to
implement parallelisation across multiple processes.

The original motivating use case for SymNum was to allow automatically
constructing the  derivatives of the sorts of functions of low dimensional
inputs which are  commonly used as toy examples to demonstrate inference and
optimisation algorithms. In these cases while manually deriving and
implementing derivatives is generally possible, this can still be labourious
and error prone, and distract from the purpose of giving a simple show case of
an algorithm. On the other hand the derivative functions produced by Autograd
and JAX in such cases are often much slower than manual implementations. SymNum
tries to fill this gap by providing the flexibility and ease of use that comes
from automatic differentiation while still being efficient for small toy
examples.


## Doesn't SymPy already have array support and allow export of NumPy functions?

Yes: SymNum is mainly a convenience wrapper around functionality already
provided by SymPy to make it easier to use for those already familiar with
NumPy and Autograd / JAX. Specifically SymPy has several inbuilt array like
classes, which can be broadly split in to the [array
types](https://docs.sympy.org/latest/modules/tensor/array.html) defined  in
`sympy.tensor.array` and the  [matrix
types](https://docs.sympy.org/latest/modules/matrices/matrices.html)  defined
in `sympy.matrices`.

Each of the inbuilt array and matrix classes supports some of the functionality
of NumPy's core `ndarray` class, however both have some issues which means they
don't provide an easy drop-in replacement, with for example matrix classes
being limited to two-dimensions, while both the inbuilt array and matrix
classes do not support the full broadcasting and operator overloading semantics
of NumPy arrays. The `SymbolicArray` class in `symnum.array` aims to provide a
more `ndarray` like interface, supporting broadcasting of elementwise binary
arithmetic operations like `*`, `/`, `+` and `-`, elementwise NumPy ufunc-like
mathematical functions like `numpy.log` via the `symnum.numpy` module, simple
array contractions over potentially multiple axes with the `sum` and `prod` 
methods and matrix multiplication with the `@` operator.

Similarly SymPy has extensive built in [code generation](https://docs.sympy.org/latest/modules/codegen.html) 
features, including the
[`lambdify`](https://docs.sympy.org/latest/modules/utilities/lambdify.html) 
function which supports generation of functions which operate on
NumPy arrays. It can be non-trivial however to use these functions to generate
code which perform indexing operations on array inputs, or to construct higher
order functions which return [closures](https://en.wikipedia.org/wiki/Closure_(computer_programming)). 
SymNum builds on top of the SymPy's code generation functionality to allow
simpler generation of NumPy functions using such features.


## Example

```Python
import numpy as np
import symnum.numpy as snp
from symnum import named_array, numpify_func, jacobian

# Define a function using the symnum.numpy interface.
def func(x):
    return (snp.array([[1., -0.5], [-2., 3.]]) @ 
            snp.array([snp.cos(-x[1]**2 + 3 * x[0]), snp.sin(x[0] - 1)]))

# Create a named symbolic array to act as input and evaluate func symbolically.
x = named_array(name='x', shape=2)
y = func(x)

# Alternatively we can symbolically 'trace' func and use this to generate a
# NumPy function which accepts ndarray arguments. To allow the tracing we
# need to manually specify the shapes of the arguments to the function.
x_np = np.array([0.2, 1.1])
func_np = numpify_func(func, x.shape)
y_np = func_np(x_np)

# We can also use a similar approach to generate a NumPy function to evaluate
# the Jacobian of func on ndarray arguments. The numpified function func_np 
# stores the symbolic function used to generate it and details of the argument
# shapes and so we can pass it as a sole argument to jacobian without
# specifying the argument shapes.
jacob_func_np = jacobian(func_np)
dy_dx_np = jacob_func_np(x_np)
```

See also the [demo Jupyter notebook](https://github.com/matt-graham/symnum/blob/main/Demo.ipynb).



## Current limitations

SymNum only supports a small subset of the NumPy API at the moment. A
non-exhaustive list of things that don't currently work

  * Indexed / sliced assignment to arrays e.g. `a[i, j] = x` and `a[:, j] = y`
  * Matrix multiplication with `@` of arrays with dimensions > 2.
  * Linear algebra operations in `numpy.linalg` and FFT functions in `numpy.fft`.
  * All `scipy` functions such as the special functions in `scipy.special`.
  * Similar to the limitations on using [Python control flow with the JIT
    transformation in JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Control-Flow),
    the symbolic tracing of functions with SymNum requires that only control
    flows that does not depend on the value of array arguments is used.

Some of these are not fundamental limitations and SymNum's coverage will 
improve (pull requests are very welcome!), however as the focus is on 
allowing automatic generation of derivatives of simple functions of smallish
arrays if your use case uses more complex NumPy features you are likely to 
find Autograd or JAX to be better bets.
