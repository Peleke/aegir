# Week 7: Differential Geometry

**Goal**: Understand curved spaces. Geodesics on manifolds.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Compute geodesics on a sphere.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 7.1 | 07a-manifolds-intuition | 60 min | Locally flat, globally curved |
| 7.2 | 07b-metrics-distances | 60 min | How to measure on curved spaces |
| 7.3 | 07c-geodesics | 60 min | Shortest paths |
| 7.4 | 07c continued | 45 min | Numerical geodesic computation |
| 7.5 | Application | 45 min | Concept space as a manifold |

---

## Day 7.1: Manifolds

### Learning Objectives
- [ ] Understand "locally flat, globally interesting"
- [ ] Visualize common manifolds
- [ ] See why embedding space could be a manifold

### The Core Idea

A **manifold** is a space that locally looks like ℝⁿ.

```
Your backyard looks flat (local).
The Earth is round (global).
The Earth is a 2D manifold.
```

### Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_sphere():
    """The 2-sphere S²: locally looks like R², globally round."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.7, cmap='viridis')
    ax.set_title('Sphere (S²)')
    plt.show()

def plot_torus():
    """The torus: locally looks like R², globally a donut."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)

    R, r = 2, 0.5  # Major and minor radii
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.7, cmap='plasma')
    ax.set_title('Torus')
    plt.show()

plot_sphere()
plot_torus()
```

### Charts and Coordinates

A **chart** is a local coordinate system.

```python
def sphere_chart_north(x, y, z):
    """
    Stereographic projection from north pole.
    Maps sphere (minus north pole) to R².
    """
    # Project from (0, 0, 1) onto z=0 plane
    u = x / (1 - z)
    v = y / (1 - z)
    return u, v

def sphere_chart_north_inverse(u, v):
    """Inverse: from R² back to sphere."""
    denom = 1 + u**2 + v**2
    x = 2*u / denom
    y = 2*v / denom
    z = (u**2 + v**2 - 1) / denom
    return x, y, z

# Demonstration
import plotly.graph_objects as go

# Points on sphere
theta = np.linspace(0, 2*np.pi, 20)
phi = np.linspace(0.1, np.pi-0.1, 10)  # Avoid poles
THETA, PHI = np.meshgrid(theta, phi)

x_sphere = np.sin(PHI) * np.cos(THETA)
y_sphere = np.sin(PHI) * np.sin(THETA)
z_sphere = np.cos(PHI)

# Project to plane
u, v = sphere_chart_north(x_sphere, y_sphere, z_sphere)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(u.flatten(), v.flatten(), c=z_sphere.flatten(), cmap='viridis')
axes[0].set_title('Stereographic projection (chart)')
axes[0].set_xlabel('u')
axes[0].set_ylabel('v')
axes[0].axis('equal')

# Show on sphere
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.scatter(x_sphere, y_sphere, z_sphere, c=z_sphere, cmap='viridis')
ax3d.set_title('Points on sphere')
plt.tight_layout()
plt.show()
```

### Why Manifolds for Concepts?

Embedding space is ℝⁿ (flat). But the **data manifold** might be curved:
- Not all of ℝⁿ contains valid concepts
- The "concept surface" might be lower-dimensional
- Distances along the surface ≠ straight-line distances

### Exercises

1. **Klein Bottle**: Research and describe the Klein bottle. Why can't it exist in 3D?
2. **Intrinsic vs Extrinsic**: The sphere is 2D (intrinsic) but embedded in 3D (extrinsic). What's the intrinsic dimension of a curve?
3. **Local Charts**: Sketch how to cover a torus with overlapping charts.

---

## Day 7.2: Metrics and Distances

### Learning Objectives
- [ ] Define the Riemannian metric
- [ ] Compute distances on curved spaces
- [ ] Understand the line element

### The Metric Tensor

A **metric** g tells you how to measure distances:

```
ds² = Σᵢⱼ gᵢⱼ dxⁱ dxʲ
```

This is the **line element** — infinitesimal distance squared.

### Flat Space

In ℝⁿ, the metric is the identity:
```
ds² = dx² + dy² + dz² = (Pythagorean theorem)
```

```python
def euclidean_metric(x):
    """Euclidean metric: identity matrix."""
    return np.eye(len(x))

def compute_length_flat(path, dt=0.01):
    """Length of path in flat space."""
    t = np.arange(0, 1, dt)
    length = 0
    for i in range(len(t)-1):
        dx = path(t[i+1]) - path(t[i])
        ds = np.sqrt(dx @ euclidean_metric(path(t[i])) @ dx)
        length += ds
    return length

# Straight line from (0,0) to (1,1)
straight = lambda t: np.array([t, t])
print(f"Length of straight line: {compute_length_flat(straight):.4f}")  # √2
```

### Sphere Metric

On a sphere of radius R, using latitude θ and longitude φ:

```
ds² = R²(dθ² + sin²θ dφ²)
```

```python
def sphere_metric(theta, phi, R=1):
    """Metric tensor on sphere at (theta, phi)."""
    return R**2 * np.array([
        [1, 0],
        [0, np.sin(theta)**2]
    ])

# Near equator (theta ≈ π/2)
print("Metric at equator:")
print(sphere_metric(np.pi/2, 0))

# Near pole (theta ≈ 0.1)
print("\nMetric near pole:")
print(sphere_metric(0.1, 0))
# Notice: longitude contribution sin²(0.1) ≈ 0.01 is small
# Near poles, moving in φ costs little distance!
```

### Distances on the Sphere

```python
def haversine_distance(lat1, lon1, lat2, lon2, R=1):
    """
    Great circle distance between two points.
    (This is the geodesic distance on a sphere.)
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# NYC to London
nyc = (np.radians(40.7), np.radians(-74.0))
london = (np.radians(51.5), np.radians(-0.1))

dist = haversine_distance(*nyc, *london, R=6371)  # Earth radius in km
print(f"NYC to London: {dist:.0f} km")
```

### Learned Metrics

What if the metric depends on the data?

```python
def learned_metric(x, data_points, sigma=0.5):
    """
    Metric that's "tighter" where data is dense.

    g(x) = I / density(x)

    Where data is dense, distances are smaller.
    """
    # Estimate local density
    distances = np.linalg.norm(data_points - x, axis=1)
    density = np.sum(np.exp(-distances**2 / (2*sigma**2)))

    # Metric inversely proportional to density
    return np.eye(len(x)) / (density + 0.1)
```

### Exercises

1. **Metric Visualization**: Plot how g₁₁ and g₂₂ vary across the sphere.
2. **Curved vs Flat**: Compare Euclidean distance to geodesic distance for two sphere points.
3. **Custom Metric**: Define a metric where one region is "stretched" and visualize it.

---

## Day 7.3-7.4: Geodesics

### Learning Objectives
- [ ] Define geodesics as shortest paths
- [ ] Derive the geodesic equation
- [ ] Compute geodesics numerically

### What's a Geodesic?

A **geodesic** is the shortest path between two points on a manifold.

- Flat space: straight lines
- Sphere: great circles
- General manifold: solutions to geodesic equation

### The Geodesic Equation

```
d²xᵏ/dt² + Σᵢⱼ Γᵏᵢⱼ (dxⁱ/dt)(dxʲ/dt) = 0
```

Where Γᵏᵢⱼ are **Christoffel symbols** (computed from metric).

### Christoffel Symbols

```python
def christoffel_symbols(metric_func, x, h=1e-5):
    """
    Compute Christoffel symbols Γᵏᵢⱼ at point x.

    Γᵏᵢⱼ = ½ gᵏˡ (∂ᵢgⱼˡ + ∂ⱼgᵢˡ - ∂ˡgᵢⱼ)
    """
    n = len(x)
    g = metric_func(x)
    g_inv = np.linalg.inv(g)

    # Compute metric derivatives
    dg = np.zeros((n, n, n))  # dg[k, i, j] = ∂g_ij/∂x_k
    for k in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[k] += h
        x_minus[k] -= h
        dg[k] = (metric_func(x_plus) - metric_func(x_minus)) / (2*h)

    # Compute Christoffel symbols
    Gamma = np.zeros((n, n, n))  # Gamma[k, i, j] = Γᵏᵢⱼ
    for k in range(n):
        for i in range(n):
            for j in range(n):
                for l in range(n):
                    Gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                        dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                    )

    return Gamma
```

### Geodesic ODE

```python
def geodesic_ode(t, y, metric_func):
    """
    ODE system for geodesics.

    y = [x⁰, x¹, ..., ẋ⁰, ẋ¹, ...]

    d²xᵏ/dt² = -Γᵏᵢⱼ ẋⁱ ẋʲ
    """
    n = len(y) // 2
    x = y[:n]
    v = y[n:]

    Gamma = christoffel_symbols(metric_func, x)

    # Compute acceleration
    a = np.zeros(n)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                a[k] -= Gamma[k, i, j] * v[i] * v[j]

    return np.concatenate([v, a])
```

### Geodesics on a Sphere

```python
from scipy.integrate import solve_ivp

def sphere_metric_numpy(coords):
    """Metric on unit sphere in (theta, phi) coordinates."""
    theta = coords[0]
    return np.array([
        [1, 0],
        [0, np.sin(theta)**2 + 1e-6]  # Small epsilon to avoid singularity
    ])

def solve_geodesic(metric_func, start, velocity, t_span=(0, 1)):
    """Solve geodesic equation from initial position and velocity."""
    y0 = np.concatenate([start, velocity])

    def ode(t, y):
        return geodesic_ode(t, y, metric_func)

    sol = solve_ivp(ode, t_span, y0, dense_output=True, max_step=0.01)
    return sol

# Initial conditions: start at equator, head northeast
theta0, phi0 = np.pi/2, 0  # Equator, prime meridian
vtheta, vphi = -0.5, 0.5   # Northeast direction

sol = solve_geodesic(sphere_metric_numpy, [theta0, phi0], [vtheta, vphi], t_span=(0, 5))

# Convert to 3D for visualization
t = np.linspace(0, 5, 200)
path = sol.sol(t)[:2]
theta, phi = path[0], path[1]

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sphere surface
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, alpha=0.3, cmap='Blues')

# Geodesic
ax.plot(x, y, z, 'r-', linewidth=2, label='Geodesic')
ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, label='Start')

ax.set_title('Geodesic on Sphere (Great Circle)')
ax.legend()
plt.show()
```

### Exercises

1. **Verify Great Circles**: Show that sphere geodesics are indeed great circles.
2. **Flat Limit**: As sphere radius → ∞, do geodesics become straight lines?
3. **Torus Geodesics**: Implement geodesics on a torus. Are they always great-circle-like?

---

## Day 7.5: Concept Space as a Manifold

### The Big Picture

If concept space is curved:
- **Shortest paths curve** — reasoning might not be straight-line
- **Some regions are denser** — more concepts packed in
- **Metric encodes structure** — learned from data

### Data-Driven Metrics

```python
def data_manifold_metric(x, embeddings, k=10, sigma=0.5):
    """
    Metric learned from data.

    Idea: distances should be smaller where data is dense.
    """
    # Find k nearest neighbors
    distances = np.linalg.norm(embeddings - x, axis=1)
    knn_indices = np.argsort(distances)[:k]
    knn_distances = distances[knn_indices]

    # Local density estimate
    density = np.exp(-knn_distances.mean() / sigma)

    # Local covariance (captures data direction)
    local_points = embeddings[knn_indices]
    cov = np.cov(local_points.T)

    # Metric = inverse covariance scaled by density
    metric = np.linalg.inv(cov + 0.01*np.eye(len(x))) * density

    return metric
```

### Geodesics Through Concept Space

```python
def concept_space_geodesic(start_concept, end_concept, embeddings):
    """
    Find geodesic between two concepts using data-driven metric.
    """
    # Define metric
    def metric(x):
        return data_manifold_metric(x, embeddings)

    # Solve BVP
    def ode(t, y):
        n = len(start_concept)
        x = y[:n]
        v = y[n:]

        # Numerical Christoffel symbols
        Gamma = christoffel_symbols(metric, x)

        a = np.zeros(n)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    a[k] -= Gamma[k, i, j] * v[i] * v[j]

        return np.concatenate([v, a])

    # ... (boundary value problem setup)
    pass
```

### Connection to Weeks 5-6

| Week 5 | Week 6 | Week 7 |
|--------|--------|--------|
| Energy landscape | Action integral | Metric |
| Gradient flow | Euler-Lagrange | Geodesic equation |
| Attractors | Optimal paths | Shortest paths |

**All the same thing, different perspectives:**
- Gradient flow on energy = geodesic on metric where energy defines distance

---

## Week 7 Milestone

By the end of Week 7, you should have:

1. **Manifold intuition**: Curved spaces, local vs global
2. **Metric understanding**: How distance depends on position
3. **Geodesic computation**: Numerical geodesics on sphere
4. **Data manifold concept**: How embeddings form a manifold

### Success Criteria

```python
# You can:
# 1. Define a metric tensor
# 2. Compute Christoffel symbols
# 3. Solve the geodesic equation
# 4. Visualize geodesics on curved surfaces

# You understand:
# - Manifolds are locally flat, globally curved
# - Metrics define distance
# - Geodesics are shortest paths
# - Concept space could be a data manifold
```

---

## Research Log Entry

```markdown
## Week 7 Observations

### Manifold Intuition
- Concept space is [N]-dimensional embedding space
- The data manifold is probably lower-dimensional
- Curvature = some regions have more concepts per volume

### Metric Explorations
- Euclidean metric ignores data structure
- Data-driven metric respects density
- Computation is [fast/slow] in [N] dimensions

### Geodesic Insights
- Geodesics curve toward [dense/sparse] regions
- Path length: Euclidean [X] vs geodesic [Y]
- Geodesics might represent "natural" reasoning paths

### Questions for Week 8
- Can we learn the metric?
- How do GNNs use graph structure like a metric?
- Is there a connection between message passing and geodesics?
```

---

→ [Week 8: Geometric Deep Learning](week-08-gdl.md)
