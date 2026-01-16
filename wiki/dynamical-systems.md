# Dynamical Systems Primer

## TL;DR

A dynamical system is a rule for how state evolves. **Attractors** are where trajectories converge. **Basins** are the regions that flow to each attractor. Concepts could be attractors in thought-space.

---

## Core Intuition

A dynamical system tells you: given where you are, where do you go next?

```
dx/dt = f(x)
```

- x is your state (position, concept, etc.)
- f(x) is the rule (physics, reasoning dynamics, etc.)
- The solution x(t) is your trajectory through state space

---

## Phase Space

**Phase space** is the space of all possible states.

For a pendulum:
- State = (angle θ, angular velocity ω)
- Phase space = 2D plane (θ, ω)

Every point in phase space determines the future evolution. The dynamics are deterministic: same starting point → same trajectory.

### Why Phase Space Matters

Instead of tracking x(t) over time, we can study the geometry of all possible trajectories at once. Patterns emerge:
- Where do trajectories converge? (attractors)
- Where do they diverge? (repellers)
- Are there cycles? (limit cycles)

---

## Vector Fields and Flow

f(x) defines a **vector field** — an arrow at every point showing the direction of motion.

```
At point x, the arrow is f(x).
```

The **flow** is the set of all trajectories following these arrows. Imagine releasing dye at many points and watching it spread.

### Python Example

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(f, xlim, ylim, density=20):
    """Plot a 2D vector field."""
    x = np.linspace(*xlim, density)
    y = np.linspace(*ylim, density)
    X, Y = np.meshgrid(x, y)

    # Compute vectors at each point
    U, V = f(X, Y)

    # Normalize for visualization
    M = np.sqrt(U**2 + V**2)
    M[M == 0] = 1
    U, V = U/M, V/M

    plt.quiver(X, Y, U, V, M, cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('y')

# Example: Simple attractor at origin
def attractor(x, y):
    return -x, -y  # Everything flows toward (0,0)

plot_vector_field(attractor, (-2, 2), (-2, 2))
plt.title("Attractor at origin")
plt.show()
```

---

## Fixed Points

A **fixed point** is where f(x*) = 0 — you're not moving.

At a fixed point, the trajectory stays put forever (in the absence of noise).

### Stability

Not all fixed points are equal:

- **Stable (attractor)**: Nearby trajectories flow toward it
- **Unstable (repeller)**: Nearby trajectories flow away
- **Saddle**: Some directions stable, some unstable

### Linear Stability Analysis

Near a fixed point x*, linearize: f(x) ≈ f(x*) + J(x - x*) where J is the Jacobian matrix.

Eigenvalues of J determine stability:
- All negative real parts → stable (attractor)
- Any positive real part → unstable
- Complex → oscillatory behavior

---

## Attractors

More generally, an **attractor** is a set that trajectories converge to:

| Type | Description | Example |
|------|-------------|---------|
| Fixed point | Single state | Ball at rest |
| Limit cycle | Periodic orbit | Heartbeat |
| Strange attractor | Fractal set | Weather (chaos) |

### Basin of Attraction

The **basin of attraction** is all starting points that eventually reach the attractor.

```
     Basin of A          Basin of B
    ─────────────────────────────────
   /                   \
  ↘                     ↙
   →→→  A  ←←←       →→→  B  ←←←
  ↗                     ↖
   \                   /
    ─────────────────────────────────
```

If you start in basin A, you end up at A. Doesn't matter exactly where — all roads lead to A.

### Relevance for Concepts

**Concepts as attractors**: A concept is a basin of attraction in thought-space. Any thought that starts in the basin "falls" toward the concept.

**Salience as basin size/depth**: More salient concepts have deeper, wider basins. They capture more initial conditions.

---

## Gradient Systems

A special (and important) case where dynamics are gradient descent on a potential:

```
dx/dt = -∇V(x)
```

You always flow "downhill" on the energy landscape V(x).

### Properties

- Always converges to local minimum (no cycles)
- Minimizes V (energy, cost, etc.)
- Attractors are exactly the local minima

### Example: Double Well

```python
import numpy as np
import matplotlib.pyplot as plt

# Double-well potential: V(x) = x^4 - 2x^2
def V(x):
    return x**4 - 2*x**2

def gradient(x):
    return 4*x**3 - 4*x  # dV/dx

x = np.linspace(-2, 2, 100)
plt.plot(x, V(x), 'b-', label='V(x)')
plt.plot(x, -gradient(x), 'r--', label='-dV/dx (flow direction)')
plt.axhline(0, color='gray', linestyle=':')
plt.legend()
plt.title("Double-well potential with two attractors")
plt.xlabel('x')
plt.show()

# Fixed points: where gradient = 0
# 4x³ - 4x = 0 → x(x² - 1) = 0 → x = -1, 0, 1
# x = ±1 are attractors (stable minima)
# x = 0 is a repeller (unstable maximum)
```

---

## Bifurcations

What happens when parameters change?

A **bifurcation** is when the qualitative behavior changes:
- New fixed points appear/disappear
- Stability changes
- Cycles emerge

This could model how concept spaces reorganize with learning.

---

## Relevance for Aegir

| Concept | Application |
|---------|-------------|
| Phase space | Concept space as the space of all possible "mental states" |
| Attractors | Concepts are attractor basins — thoughts fall into them |
| Vector field | Salience gradient determines flow direction |
| Gradient systems | If reasoning minimizes some potential, it's a gradient flow |
| Basins | The "catchment area" of each concept |
| Bifurcations | Learning reorganizes the attractor landscape |

**Key insight**: If we define a salience potential V(x) over concept space, reasoning becomes gradient descent. The path is determined by the landscape, not by explicit rules.

---

## Key Formulas

```
Dynamical system:     dx/dt = f(x)
Fixed point:          f(x*) = 0
Gradient system:      dx/dt = -∇V(x)
Stability:            Determined by eigenvalues of Jacobian at fixed point
Basin:                Set of initial conditions that flow to an attractor
```

---

## Going Deeper

- **Book**: Strogatz, "Nonlinear Dynamics and Chaos" (Chapters 1-6)
- **Visual**: [3Blue1Brown on differential equations](https://www.youtube.com/watch?v=p_di4Zn4wz4)
- **Interactive**: [Complexity Explorables](https://www.complexity-explorables.org/)
