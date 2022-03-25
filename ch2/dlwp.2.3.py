# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2.3 The gears of neural networks: Tensor operations

# 2.3.2 Broadcasting
import numpy as np
X = np.random.random((32,10))
y = np.random.random(10,)

X = np.random.random((3,5))
y = np.random.random(5,)

print(X)
print(f"X dtype: {X.dtype}")
print(f"X shape: {X.shape}")
print(f"X ndim: {X.ndim}")

print(y)
print(f"y dtype: {y.dtype}")
print(f"y shape: {y.shape}")
print(f"y ndim: {y.ndim}")

z = np.maximum(X, y)
print(z)
print(f"z dtype: {z.dtype}")
print(f"z shape: {z.shape}")
print(f"z ndim: {z.ndim}")

z = np.minimum(X, y)
print(z)

z = X + y
print(z)

# 2.3.3 Tensor product
import numpy as np
X = np.random.random((5,10))
x = np.random.random(5,)
y = np.random.random(5,)

print(f"X: {X}")
print(f"X dtype: {X.dtype}")
print(f"X shape: {X.shape}")
print(f"X ndim: {X.ndim}")

print(f"x: {x}")
print(f"x dtype: {x.dtype}")
print(f"x shape: {x.shape}")
print(f"x ndim: {x.ndim}")

print(f"y: {y}")
print(f"y dtype: {y.dtype}")
print(f"y shape: {y.shape}")
print(f"y ndim: {y.ndim}")

z = np.dot(x, y)
print(f"z: {z}")
print(f"z dtype: {z.dtype}")
print(f"z shape: {z.shape}")
print(f"z ndim: {z.ndim}")

z = np.dot(y, X)
print(f"z: {z}")
z2 = np.dot(y, X)
print(f"z: {z}")
print(f"z2: {z2}")

# 2.3.4 Tensor reshaping
import numpy as np
x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])

print(f"x: {x}")
print(f"x dtype: {x.dtype}")
print(f"x shape: {x.shape}")
print(f"x ndim: {x.ndim}")

x = x.reshape((2,3))
print(f"x: {x}")
print(f"x dtype: {x.dtype}")
print(f"x shape: {x.shape}")
print(f"x ndim: {x.ndim}")

x = np.transpose(x)
print(f"x: {x}")
print(f"x dtype: {x.dtype}")
print(f"x shape: {x.shape}")
print(f"x ndim: {x.ndim}")

# numpy.gradient
import numpy as np
y = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)
j = np.gradient(y)
print(j)

# Derivative
import sympy 
from sympy import *

x, y = symbols('x y') 
expr = x**2 + 10 * y + y**3
print("Expression : {} ".format(expr)) 

# Use sympy.Derivative() method 
expr_diff = Derivative(expr, x) 
  
#Print ("Etymology of expression with respect to x: {}".Format.(Expr_diff)
print("Value of the derivative : {} ".format(expr_diff.doit())) 

