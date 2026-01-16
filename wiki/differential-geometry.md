# Differential Geometry Primer

## TL;DR

Differential geometry studies curved spaces. **Manifolds** are spaces that look flat locally. **Metrics** define distances. **Geodesics** are shortest paths. Concept space might be curved — and that curvature matters.

---

## Core Intuition

The Earth is curved, but your backyard looks flat. That's a manifold: globally interesting, locally boring.

**Why this matters**: If concept space has non-trivial geometry, then:
- "Distance" between concepts depends on the path
- Shortest paths (geodesics) might curve
- Some regions might be "denser" than others

---

## Manifolds

A **manifold** is a space that locally looks like ℝⁿ (flat n-dimensional space).

### Examples

| Manifold | Dimension | Locally Looks Like |
|----------|-----------|-------------------|
| Circle (S¹) | 1 | Line |
| Sphere (S²) | 2 | Plane |
| Torus | 2 | Plane |
| Embedding space | n | ℝⁿ |

### Charts and Atlases

A **chart** is a local coordinate system — a way to assign coordinates to points in a region.

An **atlas** is a collection of overlapping charts that cover the whole manifold.

```
        Chart 1              Chart 2
    ┌───────────────┐   ┌───────────────┐
    │    (x, y)     │   │    (u, v)     │
    │       ╲       │   │       ╱       │
    │        ╲      │   │      ╱        │
    └─────────●─────┘   └─────●─────────┘
              │               │
              └───────────────┘
                 Overlap region
                 (transition map)
```

**For concept spaces**: We might need multiple coordinate systems to describe the space — one set of coordinates near "contract testing," another near "distributed systems."

---

## Metrics: How to Measure

A **metric** (or **Riemannian metric**) tells you how to measure distances and angles.

At each point, the metric is a matrix g_ij that defines:

```
ds² = Σᵢⱼ gᵢⱼ dxⁱ dxʲ
```

This is the **line element** — the infinitesimal distance squared.

### Flat Space

In flat space (ℝⁿ), the metric is just the identity:
```
ds² = dx² + dy² + dz² + ...
```

This is Pythagoras in infinitesimal form.

### Curved Space Example: Sphere

On a sphere of radius R, using latitude θ and longitude φ:
```
ds² = R²(dθ² + sin²θ dφ²)
```

Near the equator (θ ≈ π/2), both directions are equally "wide."
Near the poles (θ ≈ 0), longitude lines converge — moving in φ costs less distance.

### Python Example

```python
import numpy as np

def flat_metric(x):
    """Euclidean metric: identity matrix everywhere."""
    n = len(x)
    return np.eye(n)

def sphere_metric(theta, phi, R=1):
    """Metric on a sphere at point (theta, phi)."""
    return R**2 * np.array([
        [1, 0],
        [0, np.sin(theta)**2]
    ])

# At equator (theta = pi/2)
print("Equator metric:")
print(sphere_metric(np.pi/2, 0))

# Near pole (theta = 0.1)
print("\nNear pole metric:")
print(sphere_metric(0.1, 0))
```

---

## Geodesics: Shortest Paths

A **geodesic** is the shortest path between two points on a manifold.

- In flat space: straight lines
- On a sphere: great circles
- In concept space: ???

### The Geodesic Equation

Geodesics satisfy:
```
d²xᵏ/dt² + Σᵢⱼ Γᵏᵢⱼ (dxⁱ/dt)(dxʲ/dt) = 0
```

Where Γᵏᵢⱼ are **Christoffel symbols** (derived from the metric).

**Intuition**: Even if you're "going straight" (no acceleration in your local frame), the curvature of space can bend your path.

### Geodesics on a Sphere

Great circles are geodesics. Airplanes fly along great circles, not straight lines on the map, because great circles are actually shorter in 3D space.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def great_circle(lat1, lon1, lat2, lon2, n_points=100):
    """Generate points along a great circle between two locations."""
    # Convert to radians
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    # Interpolate
    t = np.linspace(0, 1, n_points)

    # Spherical interpolation (simplified)
    lats = lat1 + t * (lat2 - lat1)
    lons = lon1 + t * (lon2 - lon1)

    # Convert to Cartesian for plotting
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)

    return x, y, z

# Plot geodesic from New York to London
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sphere
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, alpha=0.3)

# Geodesic
x, y, z = great_circle(40.7, -74.0, 51.5, -0.1)  # NYC to London
ax.plot(x, y, z, 'r-', linewidth=2)

plt.title("Great circle (geodesic) on sphere")
plt.show()
```

---

## Curvature

**Curvature** measures how much a manifold deviates from flatness.

### Gaussian Curvature (2D)

For a 2D surface, Gaussian curvature K at a point:
- K > 0: Sphere-like (curves the same way in all directions)
- K = 0: Flat (cylinder, plane)
- K < 0: Saddle-like (curves opposite ways)

```
    K > 0           K = 0           K < 0
    (sphere)        (cylinder)      (saddle)
      ╭─╮             │             ╲   ╱
     ╱   ╲            │              ╲ ╱
    ╱     ╲           │               ╳
    ╲     ╱           │              ╱ ╲
     ╲   ╱            │             ╱   ╲
      ╰─╯             │
```

### Riemann Curvature (nD)

In higher dimensions, curvature is described by the **Riemann tensor** R^ρ_σμν.

**Intuition**: If you parallel transport a vector around a small loop, it comes back rotated. The Riemann tensor measures this rotation.

---

## Tangent Spaces

At each point p on a manifold, there's a **tangent space** T_pM — the space of all possible velocities/directions at that point.

```
                  Tangent plane at p
                 ╱
                ╱
    ●──────────●──────────●
   ╱          ╱p         ╱
  ╱          ╱          ╱
 ●──────────●──────────●
              Manifold
```

For embeddings:
- The embedding space is ℝⁿ
- But the "concept manifold" might be lower-dimensional
- Tangent space at a concept = directions of "nearby" concepts

---

## Relevance for Aegir

| Concept | Application |
|---------|-------------|
| Manifold | Concept space as a curved manifold embedded in ℝⁿ |
| Metric | Salience-weighted distance (closer = more related) |
| Geodesic | Natural reasoning path (shortest path on salience landscape) |
| Curvature | Some regions are "denser" with concepts |
| Tangent space | Local directions of thought at a concept |

**Key insight**: If we learn a metric g_ij over concept space, geodesics become "natural" reasoning paths. The geometry encodes the structure.

---

## Key Formulas

```
Line element:      ds² = Σᵢⱼ gᵢⱼ dxⁱ dxʲ
Geodesic equation: d²xᵏ/dt² + Γᵏᵢⱼ (dxⁱ/dt)(dxʲ/dt) = 0
Christoffel:       Γᵏᵢⱼ = ½gᵏˡ(∂ᵢgⱼˡ + ∂ⱼgᵢˡ - ∂ˡgᵢⱼ)
Gaussian curv:     K = (1/R₁)(1/R₂) for principal radii
```

---

## Going Deeper

- **Book**: Do Carmo, "Riemannian Geometry" (first 3 chapters)
- **Visual**: [Eigenchris YouTube series on Tensors](https://www.youtube.com/playlist?list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx)
- **Interactive**: [The Shape of Space](http://www.geom.uiuc.edu/zoo/toptype/torus/)
