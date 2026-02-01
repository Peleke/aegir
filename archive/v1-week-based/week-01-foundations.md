# Week 1: Foundations

**Goal**: Comfortable with NumPy/PyTorch, linear algebra and calculus refreshed.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Implement gradient descent from scratch.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 1.1 | 01a-numpy-pytorch-primer | 60 min | Tensors, broadcasting, indexing |
| 1.2 | 01a continued | 45 min | Autograd, GPU basics |
| 1.3 | 01b-linear-algebra-refresh | 60 min | Vectors, matrices, transformations |
| 1.4 | 01b continued | 45 min | Eigenvalues, SVD |
| 1.5 | 01c-calculus-refresh | 60 min | Derivatives, gradients, chain rule |

---

## Day 1.1-1.2: NumPy & PyTorch Primer

### Learning Objectives
- [ ] Create and manipulate tensors
- [ ] Understand broadcasting rules
- [ ] Use advanced indexing
- [ ] Compute gradients with autograd

### Key Concepts

**Tensors** are n-dimensional arrays:
```python
import torch

# 0D: scalar
scalar = torch.tensor(3.14)

# 1D: vector
vector = torch.tensor([1, 2, 3])

# 2D: matrix
matrix = torch.tensor([[1, 2], [3, 4]])

# 3D: batch of matrices
batch = torch.randn(32, 3, 3)  # 32 matrices, each 3×3
```

**Broadcasting** lets you operate on different shapes:
```python
# Add scalar to matrix (broadcasts)
matrix + 1

# Add vector to matrix (broadcasts along rows)
matrix + vector  # vector must match last dim
```

**Autograd** computes gradients automatically:
```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
```

### Exercises

1. **Tensor Surgery**: Create a 10×10 matrix and set the diagonal to zeros using indexing.
2. **Broadcasting Challenge**: Normalize each row of a matrix to sum to 1.
3. **Gradient Computation**: Compute the gradient of f(x,y) = x²y + sin(xy) at (1, π).

### Resources
- [PyTorch Tensors Tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## Day 1.3-1.4: Linear Algebra Refresh

### Learning Objectives
- [ ] Visualize vectors as arrows and transformations
- [ ] Understand matrix multiplication geometrically
- [ ] Compute and interpret eigenvalues/eigenvectors
- [ ] Use SVD for dimensionality reduction

### Key Concepts

**Vectors** are directions + magnitudes:
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_vector(v, color='blue', label=None):
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color, label=label)

v = np.array([2, 1])
plot_vector(v, label='v')
plt.xlim(-1, 4)
plt.ylim(-1, 3)
plt.grid()
plt.legend()
```

**Matrix multiplication** = applying a transformation:
```python
# Rotation matrix (45 degrees)
theta = np.pi/4
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

v_rotated = R @ v  # Apply rotation
```

**Eigenvalues and eigenvectors**: Av = λv
- Eigenvectors: directions that only get scaled
- Eigenvalues: the scaling factors

```python
A = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: [3, 1]
# eigenvectors: [[0.707, -0.707], [0.707, 0.707]]
```

### Why This Matters for Us

- **Embeddings** are vectors in high-dimensional space
- **Attention** uses matrix multiplication
- **PCA/SVD** finds principal directions in embedding space
- **Jacobians** are matrices of partial derivatives (Week 5-6)

### Exercises

1. **Transformation Visualization**: Plot a unit circle, then apply a 2×2 matrix and plot the result. What shape do you get?
2. **Eigenvalue Intuition**: Find a matrix where both eigenvalues are negative. What happens to vectors under repeated application?
3. **SVD Experiment**: Take a 100×50 matrix of random numbers. Use SVD to find the rank-10 approximation. How much "energy" is preserved?

### Resources
- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — **HIGHLY RECOMMENDED**
- Wiki: [Glossary](../wiki/glossary.md) — eigenvalue, eigenvector

---

## Day 1.5: Calculus Refresh

### Learning Objectives
- [ ] Compute derivatives symbolically and numerically
- [ ] Understand gradients as "direction of steepest ascent"
- [ ] Apply the chain rule
- [ ] Implement gradient descent

### Key Concepts

**Derivatives** measure rate of change:
```python
# Numerical derivative
def derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

f = lambda x: x**2
print(derivative(f, 3))  # ≈ 6 (exact: 2*3 = 6)
```

**Gradients** are vectors of partial derivatives:
```python
# For f(x, y) = x² + y²
# ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

def gradient(x, y):
    return np.array([2*x, 2*y])
```

**Chain rule**: (f ∘ g)' = f'(g(x)) · g'(x)

This is how backpropagation works!

**Gradient descent**: Move opposite to gradient to minimize:
```python
def gradient_descent(f, grad_f, x0, lr=0.1, n_steps=100):
    x = x0
    history = [x.copy()]

    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        history.append(x.copy())

    return x, history

# Minimize f(x,y) = x² + y²
def f(xy):
    return xy[0]**2 + xy[1]**2

def grad_f(xy):
    return np.array([2*xy[0], 2*xy[1]])

x_opt, history = gradient_descent(f, grad_f, np.array([3.0, 4.0]))
print(x_opt)  # Close to [0, 0]
```

### Why This Matters for Us

- **Embeddings** are learned via gradient descent
- **Variational calculus** (Week 6) generalizes this to function spaces
- **Dynamical systems** (Week 5) use derivatives to describe motion
- **Backpropagation** is just the chain rule

### Exercises

1. **Gradient Visualization**: Plot the function f(x,y) = sin(x) + cos(y) as a surface. Overlay gradient arrows at several points.
2. **Learning Rate Experiment**: Run gradient descent on f(x) = x⁴ - 3x³ + 2. Try different learning rates. What happens if it's too large? Too small?
3. **Implement Backprop**: For a simple 2-layer network, compute gradients by hand using chain rule. Verify with PyTorch autograd.

### Resources
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — backprop visualization

---

## Week 1 Milestone: Gradient Descent from Scratch

By the end of Week 1, you should be able to:

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Define a loss landscape
def rosenbrock(xy):
    x, y = xy
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# 2. Run gradient descent
x = np.array([-1.0, 1.0])
lr = 0.001
history = [x.copy()]

for _ in range(10000):
    x = x - lr * rosenbrock_grad(x)
    history.append(x.copy())

# 3. Visualize the optimization path
history = np.array(history)
plt.figure(figsize=(10, 8))

# Contour plot
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
zz = rosenbrock([xx, yy])
plt.contour(xx, yy, zz, levels=np.logspace(-1, 3, 20), cmap='viridis')

# Optimization path
plt.plot(history[:, 0], history[:, 1], 'r.-', markersize=2, linewidth=0.5)
plt.plot(1, 1, 'g*', markersize=15)  # Optimum
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on Rosenbrock Function')
plt.colorbar(label='f(x,y)')
plt.show()

print(f"Final position: {x}")
print(f"Final value: {rosenbrock(x)}")
```

**Success criteria**: Your optimization path should wind toward (1, 1), the global minimum.

---

## Reflection Questions

1. Why do we need autograd instead of computing gradients by hand?
2. What's the geometric interpretation of eigenvalues?
3. Why is gradient descent called "greedy"? What are its limitations?

---

## Week 1 Complete!

You now have the mathematical tools for the rest of the curriculum:
- Tensor manipulation (NumPy/PyTorch)
- Linear algebra (transformations, eigenthings)
- Calculus (gradients, optimization)

**Key insight**: All of machine learning is optimization. We define a loss, compute gradients, and descend. The magic is in choosing what to optimize.

→ [Week 2: Embeddings](week-02-embeddings.md)
