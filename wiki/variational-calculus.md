# Variational Calculus Primer

## TL;DR

Regular calculus finds the x that minimizes f(x). Variational calculus finds the **function** q(t) that minimizes a **functional** S[q]. This is how physics describes natural motion — and potentially, natural reasoning.

---

## Core Intuition

**Regular optimization**: Given f(x), find x* that minimizes f.
- Example: Find the point where a ball rests (minimum potential energy)

**Variational optimization**: Given S[q], find the function q*(t) that minimizes S.
- Example: Find the **path** a ball takes as it falls (minimizes action)

The key shift: we're optimizing over **entire paths**, not just points.

---

## Functionals

A **functional** takes a function and returns a number:

```
S[q] = ∫₀ᵀ L(q(t), q̇(t), t) dt
```

Where:
- q(t) is a path (function of time)
- q̇(t) = dq/dt is the velocity
- L is the **Lagrangian** (a function of position, velocity, and time)
- S is the **action** (a number that depends on the whole path)

### Example: Length of a Curve

The length of a curve y = q(x) from x=0 to x=1:

```
Length[q] = ∫₀¹ √(1 + q'(x)²) dx
```

Different curves have different lengths. Which one minimizes length between two points?

---

## The Euler-Lagrange Equation

The path q(t) that makes S[q] stationary (δS = 0) satisfies:

```
d/dt(∂L/∂q̇) - ∂L/∂q = 0
```

This is a differential equation. Solve it to find the optimal path.

### Derivation Sketch

1. Consider a small perturbation: q(t) → q(t) + εη(t)
2. Require δS = dS/dε|_{ε=0} = 0
3. Use integration by parts
4. Since η(t) is arbitrary, the integrand must vanish

---

## Example: Shortest Path

**Problem**: Find the shortest path between (0, a) and (1, b).

**Lagrangian**: L = √(1 + q'²) (arc length element)

**Euler-Lagrange**:
```
d/dx(∂L/∂q') = 0  (since ∂L/∂q = 0)

∂L/∂q' = q'/√(1 + q'²)

d/dx(q'/√(1 + q'²)) = 0

→ q'/√(1 + q'²) = constant

→ q' = constant

→ q(x) = mx + c  (straight line!)
```

**Result**: The shortest path is a straight line. Variational calculus derived this from first principles.

---

## The Principle of Least Action

In physics, nature "chooses" the path that makes the action stationary:

```
S = ∫ L dt = ∫ (T - V) dt
```

Where:
- T = kinetic energy = ½mv²
- V = potential energy
- L = T - V is the Lagrangian

### Why This Matters

All of classical mechanics follows from this one principle:
- Newton's laws? Derived from Euler-Lagrange.
- Conservation of energy? Derived.
- Conservation of momentum? Derived.
- Symmetries → Conservation laws (Noether's theorem)

It's not "F = ma is fundamental." It's "paths minimize action is fundamental."

---

## Example: Free Particle

**Setup**: Particle of mass m, no forces.

**Lagrangian**: L = T - V = ½mq̇² - 0 = ½mq̇²

**Euler-Lagrange**:
```
d/dt(∂L/∂q̇) - ∂L/∂q = 0
d/dt(mq̇) - 0 = 0
mq̈ = 0
q̈ = 0
```

**Result**: q(t) = at + b. Free particles move in straight lines at constant velocity.

---

## Example: Harmonic Oscillator

**Setup**: Particle in a spring potential V = ½kq²

**Lagrangian**: L = ½mq̇² - ½kq²

**Euler-Lagrange**:
```
d/dt(mq̇) - (-kq) = 0
mq̈ + kq = 0
q̈ = -(k/m)q
```

**Result**: q(t) = A cos(ωt + φ) where ω = √(k/m). Simple harmonic motion.

---

## Relevance for Aegir

**The Big Idea**: What if reasoning has an action principle?

```
S[path] = ∫ L(concept, velocity, t) dt
```

Where:
- concept = position in concept space
- velocity = rate of conceptual change
- L = something like (effort - relevance)

Then:
- Natural reasoning paths minimize action
- Justification is intrinsic: "this path exists because it's optimal"
- Euler-Lagrange gives the dynamics of thought

---

## Python Example

```python
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def shortest_path_bvp():
    """
    Find shortest path from (0, 0) to (1, 1)
    in a space with metric g(x, y).

    For flat space, this is a straight line.
    For curved space, it's a geodesic.
    """

    def ode(t, y):
        # y = [q, q']
        # For flat space: q'' = 0
        q, qp = y
        return [qp, 0]  # [q', q'']

    def bc(ya, yb):
        # Boundary conditions: q(0) = 0, q(1) = 1
        return [ya[0] - 0, yb[0] - 1]

    # Initial guess: straight line
    t = np.linspace(0, 1, 10)
    y_guess = np.vstack([t, np.ones_like(t)])

    sol = solve_bvp(ode, bc, t, y_guess)

    return sol

sol = shortest_path_bvp()
plt.plot(sol.x, sol.y[0])
plt.title("Shortest path (geodesic in flat space)")
plt.show()
```

---

## Key Formulas

```
Action:            S[q] = ∫ L(q, q̇, t) dt
Euler-Lagrange:    d/dt(∂L/∂q̇) = ∂L/∂q
Lagrangian:        L = T - V (kinetic - potential)
Free particle:     L = ½mq̇²  →  q̈ = 0
Harmonic osc:      L = ½mq̇² - ½kq²  →  q̈ = -ω²q
```

---

## Going Deeper

- **Book**: Gelfand & Fomin, "Calculus of Variations" (first 3 chapters)
- **Book**: Landau & Lifshitz, "Mechanics" (for physics context)
- **Video**: Physics lectures on Lagrangian mechanics (MIT OCW)
