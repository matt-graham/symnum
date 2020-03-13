## SymNum - symbolically construct NumPy functions and their derivatives

SymNum is a Python package that acts a bridge between NumPy and SymPy, providing
a NumPy-like interface that can be used to symbolically define functions which 
take arrays as arguments and potentially return arrays as values. A series of
Autograd style functional differential operators are also provided to construct
derivatives of symbolically defined functions, with the option to generate 
optimised NumPy code to numerically evaluate these derivative functions.

An example of the interface

```Python
import numpy as np
import symnum.numpy as snp
from symnum import numpify, sympy_jacobian, numpy_jacobian

def func(θ):
    return snp.array([[1., -0.5], [-2., 3.]]) @ snp.array(
        [snp.cos(-θ[1]**2 + 3 * θ[0]), snp.sin(θ[0] - 1)])

# Evaluate func symbolically
θ = symnum.array.named_array('θ', 2)
f = func(θ)

# Evaluate Jacobian of func symbolically
jac = sympy_jacobian(func)(θ)

# Generate NumPy function to evaluate func numerically
# We need to manually specify the shapes of the arguments
func_np = numpify(func, θ.shape)

# Evaluate generated function on a NumPy array
θ_np = np.array([0.2, 1.1])
f_np = func_np(θ_np)

# Generate NumPy function to evaluate Jacobian of func
# We again need to manually specify the shapes of the arguments
jac_func_np = numpy_jacobian(func, θ.shape)
jac_np = jac_func_np(θ_np)
```

This is a very early stage project so expect lots of rough edges and changes
to the API!