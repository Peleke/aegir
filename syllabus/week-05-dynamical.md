# Week 5: Dynamical Systems

**Goal**: Think in phase space. Concepts as attractors.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Visualize a 2D attractor landscape.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 5.1 | 05a-phase-portraits | 60 min | State space, vector fields |
| 5.2 | 05b-attractors-basins | 60 min | Fixed points, stability |
| 5.3 | 05b continued | 45 min | Basins of attraction |
| 5.4 | 05c-gradient-systems | 60 min | Energy landscapes |
| 5.5 | Connection to concepts | 45 min | Concepts as attractors |

---

## Day 5.1: Phase Portraits

### Learning Objectives
- [ ] Understand state space and trajectories
- [ ] Plot vector fields
- [ ] Simulate and visualize flow

### The Core Idea

A **dynamical system** tells you: given where you are, where do you go next?

```
dx/dt = f(x)
```

- **x** is your state (position, velocity, concept activation, ...)
- **f(x)** is the rule (physics, neural dynamics, reasoning, ...)
- **x(t)** is your trajectory through state space

### Vector Fields

At every point, f(x) is an arrow showing direction of motion.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(f, xlim=(-3, 3), ylim=(-3, 3), density=20):
    """Plot a 2D vector field."""
    x = np.linspace(*xlim, density)
    y = np.linspace(*ylim, density)
    X, Y = np.meshgrid(x, y)

    # Compute vectors
    U, V = f(X, Y)

    # Normalize for visualization
    M = np.sqrt(U**2 + V**2)
    M[M == 0] = 1
    U_norm, V_norm = U/M, V/M

    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, U_norm, V_norm, M, cmap='coolwarm', alpha=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.colorbar(label='Speed')
    return plt.gca()

# Simple attractor at origin
def attractor(x, y):
    return -x, -y

ax = plot_vector_field(attractor)
ax.set_title('Attractor at origin: dx/dt = -x, dy/dt = -y')
plt.show()
```

### Simulating Trajectories

```python
from scipy.integrate import solve_ivp

def simulate_trajectory(f, x0, t_span, t_eval=None):
    """Simulate a trajectory from initial condition x0."""
    def ode(t, y):
        return f(y[0], y[1])

    if t_eval is None:
        t_eval = np.linspace(*t_span, 200)

    sol = solve_ivp(ode, t_span, x0, t_eval=t_eval, dense_output=True)
    return sol

# Simulate from multiple initial conditions
def plot_trajectories(f, initial_conditions, t_span=(0, 10)):
    ax = plot_vector_field(f)

    for x0 in initial_conditions:
        sol = simulate_trajectory(f, x0, t_span)
        ax.plot(sol.y[0], sol.y[1], 'k-', linewidth=1)
        ax.plot(x0[0], x0[1], 'go', markersize=8)  # Start
        ax.plot(sol.y[0, -1], sol.y[1, -1], 'ro', markersize=8)  # End

    plt.title('Trajectories in phase space')
    plt.show()

# Example: Damped oscillator
def damped_oscillator(x, y):
    return y, -x - 0.3*y  # dx/dt = y, dy/dt = -x - 0.3y

initial_conditions = [[2, 0], [-2, 1], [0, 2], [1, -2]]
plot_trajectories(damped_oscillator, initial_conditions)
```

### Exercises

1. **Saddle Point**: Plot the vector field for dx/dt = x, dy/dt = -y. What happens at origin?
2. **Limit Cycle**: Plot dx/dt = y + x(1-x²-y²), dy/dt = -x + y(1-x²-y²). What's the attractor?
3. **Multiple Attractors**: Create a system with two stable fixed points.

---

## Day 5.2-5.3: Attractors and Basins

### Learning Objectives
- [ ] Classify fixed points (stable, unstable, saddle)
- [ ] Compute basins of attraction
- [ ] Visualize the basin structure

### Fixed Points

A **fixed point** is where f(x*) = 0 — you're not moving.

```python
from scipy.optimize import fsolve

def find_fixed_points(f, guesses):
    """Find fixed points near each guess."""
    fixed_points = []
    for guess in guesses:
        def f_vec(xy):
            return list(f(xy[0], xy[1]))
        root = fsolve(f_vec, guess)
        fixed_points.append(root)
    return np.unique(np.round(fixed_points, 5), axis=0)

# Example: double-well system
def double_well(x, y):
    return y, -4*x**3 + 4*x - 0.5*y  # Potential: V = x^4 - 2x^2

guesses = [[0, 0], [1, 0], [-1, 0], [0.5, 0], [-0.5, 0]]
fps = find_fixed_points(double_well, guesses)
print("Fixed points:", fps)
```

### Stability Analysis

Near a fixed point, linearize: f(x) ≈ J(x - x*) where J is the Jacobian.

```python
def jacobian(f, x, h=1e-6):
    """Numerical Jacobian at point x."""
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        f_plus = np.array(f(x_plus[0], x_plus[1]))
        f_minus = np.array(f(x_minus[0], x_minus[1]))
        J[:, i] = (f_plus - f_minus) / (2*h)
    return J

def classify_fixed_point(f, fp):
    """Classify a fixed point by eigenvalues of Jacobian."""
    J = jacobian(f, fp)
    eigvals = np.linalg.eigvals(J)

    real_parts = eigvals.real
    if all(r < 0 for r in real_parts):
        return "stable (attractor)"
    elif all(r > 0 for r in real_parts):
        return "unstable (repeller)"
    else:
        return "saddle"

for fp in fps:
    classification = classify_fixed_point(double_well, fp)
    print(f"{fp} -> {classification}")
```

### Basins of Attraction

The **basin of attraction** is all points that flow to a given attractor.

```python
def compute_basin(f, xlim, ylim, resolution=100, t_max=50):
    """
    Compute which attractor each initial condition flows to.
    Returns a 2D array of attractor labels.
    """
    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    basin = np.zeros((resolution, resolution))

    attractors = []  # Will discover as we go

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            sol = simulate_trajectory(f, [xi, yj], (0, t_max))
            final = np.array([sol.y[0, -1], sol.y[1, -1]])

            # Find which attractor this goes to
            found = False
            for k, att in enumerate(attractors):
                if np.linalg.norm(final - att) < 0.1:
                    basin[j, i] = k
                    found = True
                    break

            if not found:
                attractors.append(final)
                basin[j, i] = len(attractors) - 1

    return basin, attractors

basin, attractors = compute_basin(double_well, (-2, 2), (-2, 2))

plt.figure(figsize=(10, 8))
plt.imshow(basin, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu')
plt.colorbar(label='Attractor index')
plt.title('Basins of Attraction')
for att in attractors:
    plt.plot(att[0], att[1], 'ko', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### Exercises

1. **Basin Boundaries**: Where is the boundary between basins? What happens to trajectories starting there?
2. **Saddle's Role**: Identify the saddle point. How does it relate to the basin boundary?
3. **Parameter Dependence**: Change the damping coefficient. How do basins change?

---

## Day 5.4: Gradient Systems

### Learning Objectives
- [ ] Understand gradient descent as dynamics
- [ ] Visualize energy landscapes
- [ ] Connect to optimization

### Gradient Systems

A **gradient system** is where dynamics are gradient descent on a potential:

```
dx/dt = -∇V(x)
```

You always flow "downhill" on V.

```python
def gradient_system_from_potential(V):
    """Create a dynamical system from a potential function."""
    def f(x, y):
        h = 1e-6
        dVdx = (V(x+h, y) - V(x-h, y)) / (2*h)
        dVdy = (V(x, y+h) - V(x, y-h)) / (2*h)
        return -dVdx, -dVdy
    return f

# Double-well potential
def V_double_well(x, y):
    return x**4 - 2*x**2 + 0.5*y**2

f = gradient_system_from_potential(V_double_well)
```

### Visualizing Energy Landscape

```python
def plot_energy_landscape(V, xlim, ylim, n_levels=20):
    """Plot potential energy landscape with contours."""
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x, y)
    Z = V(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Contour plot
    axes[0].contour(X, Y, Z, levels=n_levels, cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Energy Contours (top view)')

    # 3D surface
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('V(x,y)')
    ax3d.set_title('Energy Landscape (3D)')

    plt.tight_layout()
    plt.show()

plot_energy_landscape(V_double_well, (-2, 2), (-2, 2))
```

### Why Gradient Systems Matter

- **No cycles**: Trajectories can't loop (energy always decreases)
- **Attractors = minima**: Fixed points are exactly the local minima of V
- **Connection to ML**: Training neural networks is gradient descent on loss

### Exercises

1. **Design a Landscape**: Create a potential with 3 local minima. Visualize basins.
2. **Non-Gradient System**: Show that dx/dt = y, dy/dt = -x is NOT a gradient system.
3. **Saddle Escape**: Simulate a trajectory starting near a saddle. Where does it go?

---

## Day 5.5: Concepts as Attractors

### The Big Idea

**Concepts could be attractors in thought-space.**

- A concept is a basin of attraction
- Thoughts "fall" into the nearest concept
- Salience shapes the energy landscape

```python
def conceptual_potential(embeddings, concept_centers, sigma=0.1):
    """
    Create a potential where concept centers are minima.

    V(x) = -Σᵢ exp(-||x - cᵢ||² / 2σ²)

    This creates a "well" at each concept center.
    """
    def V(x):
        total = 0
        for center in concept_centers:
            dist_sq = np.sum((x - center)**2)
            total -= np.exp(-dist_sq / (2*sigma**2))
        return total
    return V

# Example: 2D concept space with 3 concepts
concept_centers = np.array([
    [1, 1],
    [-1, 1],
    [0, -1]
])

V = conceptual_potential(None, concept_centers, sigma=0.5)
f = gradient_system_from_potential(V)

plot_energy_landscape(V, (-2, 2), (-2, 2))
plot_trajectories(f, [
    [0, 0], [0.5, 0.5], [-0.5, 0.5], [0, -0.5],
    [1.5, 0], [-1.5, 0], [0, 1.5], [0.5, -1.5]
])
```

### Connection to Week 4

From Week 4, we have concept clusters → Now they become attractors!

```python
# From clustering (Week 4)
concept_centers = [c['centroid'] for c in concepts]

# Project to 2D for visualization
from umap import UMAP
reducer = UMAP(n_components=2)
centers_2d = reducer.fit_transform(np.array(concept_centers))

# Create potential and visualize
V = conceptual_potential(None, centers_2d, sigma=0.3)
plot_energy_landscape(V, (-3, 3), (-3, 3))
```

---

## Week 5 Milestone

By the end of Week 5, you should have:

1. **Phase portrait intuition**: Can read vector field visualizations
2. **Attractor/basin computation**: Code to find and classify fixed points
3. **Energy landscape visualization**: 2D and 3D potential plots
4. **Conceptual connection**: See how concepts could be attractors

### Success Criteria

```python
# You can:
# 1. Plot a vector field
# 2. Simulate trajectories
# 3. Find fixed points and classify stability
# 4. Compute and visualize basins
# 5. Create gradient systems from potentials

# You understand:
# - Attractors "pull" trajectories
# - Basins define "catchment areas"
# - Gradient systems always reach minima
# - Concepts = attractors is a useful metaphor
```

---

## Research Log Entry

```markdown
## Week 5 Observations

### Dynamical Intuition
- Phase space is [X]-dimensional for our embedding space
- Trajectories = paths of reasoning/attention?

### Attractor Exploration
- Created potential with [N] concept attractors
- Basin boundaries are [description]
- Trajectories from random points converge in [~T] steps

### Key Questions
- What determines the "depth" of a concept attractor?
- How should concept attractors interact?
- What does "reasoning trajectory" mean in practice?
```

---

→ [Week 6: Variational Calculus](week-06-variational.md)
