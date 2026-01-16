# Week 6: Variational Calculus

**Goal**: Understand "paths that minimize something."

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Derive straight line as shortest path. Simulate least-action reasoning.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 6.1 | 06a-functionals-intuition | 60 min | Functions of functions |
| 6.2 | 06b-euler-lagrange | 60 min | The master equation |
| 6.3 | 06b continued | 45 min | Solve classic problems |
| 6.4 | 06c-least-action-reasoning | 60 min | Action on concept space |
| 6.5 | Integration | 45 min | Connect to dynamics (Week 5) |

---

## Day 6.1: Functionals

### Learning Objectives
- [ ] Distinguish functions from functionals
- [ ] Compute the action integral
- [ ] Understand why paths are the optimization target

### The Key Shift

| Regular Optimization | Variational Optimization |
|---------------------|-------------------------|
| Find x* that minimizes f(x) | Find path q*(t) that minimizes S[q] |
| Input: number | Input: function |
| Output: number | Output: number |
| Example: Find minimum of a curve | Example: Find the shortest path |

### Functionals

A **functional** takes a function and returns a number:

```python
import numpy as np
from scipy.integrate import quad

def path_length(q, dq, t_span):
    """
    Functional: Length of path q(t).

    L = ∫ √(1 + q'(t)²) dt
    """
    def integrand(t):
        return np.sqrt(1 + dq(t)**2)

    length, _ = quad(integrand, t_span[0], t_span[1])
    return length

# Example: straight line q(t) = t from t=0 to t=1
q_straight = lambda t: t
dq_straight = lambda t: 1

print(f"Length of straight line: {path_length(q_straight, dq_straight, (0, 1)):.4f}")
# Should be √2 ≈ 1.414

# Curved path q(t) = t²
q_curved = lambda t: t**2
dq_curved = lambda t: 2*t

print(f"Length of parabola: {path_length(q_curved, dq_curved, (0, 1)):.4f}")
# Should be longer
```

### The Action

In physics, the action is:

```
S[q] = ∫ L(q, q̇, t) dt
```

Where L is the **Lagrangian** = (kinetic - potential) energy.

```python
def action(q, dq, L, t_span, n_points=1000):
    """
    Compute action S = ∫ L(q, dq, t) dt
    """
    t = np.linspace(t_span[0], t_span[1], n_points)
    dt = t[1] - t[0]

    integrand = np.array([L(q(ti), dq(ti), ti) for ti in t])
    return np.sum(integrand) * dt

# Free particle Lagrangian: L = ½mv² (kinetic energy only)
def L_free_particle(q, dq, t, m=1):
    return 0.5 * m * dq**2

# Compare action for straight vs curved paths
# Both go from q(0)=0 to q(1)=1
print(f"Action (straight): {action(q_straight, dq_straight, L_free_particle, (0, 1)):.4f}")
print(f"Action (curved): {action(q_curved, dq_curved, L_free_particle, (0, 1)):.4f}")
# Straight line has lower action!
```

### Exercises

1. **Arc Length Functional**: Derive the formula for arc length as a functional.
2. **Travel Time**: What's the functional for travel time if speed depends on position?
3. **Compare Paths**: Generate 5 random paths between same endpoints. Which has lowest action?

---

## Day 6.2-6.3: Euler-Lagrange Equation

### Learning Objectives
- [ ] Derive Euler-Lagrange from δS = 0
- [ ] Solve for optimal paths
- [ ] Apply to physics problems

### The Master Equation

The path q(t) that makes S[q] stationary satisfies:

```
d/dt(∂L/∂q̇) - ∂L/∂q = 0
```

This is a differential equation. Solve it → get optimal path.

### Derivation Sketch

```python
# Conceptual (not runnable, for understanding)

# 1. Consider perturbation: q(t) → q(t) + εη(t)
#    where η(0) = η(T) = 0 (endpoints fixed)

# 2. Taylor expand:
#    S[q + εη] ≈ S[q] + ε·δS + O(ε²)

# 3. For S to be stationary:
#    δS = 0 for all perturbations η

# 4. After integration by parts:
#    ∫ [∂L/∂q - d/dt(∂L/∂q̇)] η dt = 0

# 5. Since η is arbitrary:
#    ∂L/∂q - d/dt(∂L/∂q̇) = 0

# This is the Euler-Lagrange equation!
```

### Example: Free Particle

```python
# Lagrangian: L = ½mq̇²
# ∂L/∂q = 0
# ∂L/∂q̇ = mq̇
# Euler-Lagrange: d/dt(mq̇) - 0 = 0
#                 mq̈ = 0
#                 q̈ = 0
# Solution: q(t) = at + b (straight line!)
```

### Example: Harmonic Oscillator

```python
# Lagrangian: L = ½mq̇² - ½kq²
# ∂L/∂q = -kq
# ∂L/∂q̇ = mq̇
# Euler-Lagrange: d/dt(mq̇) - (-kq) = 0
#                 mq̈ + kq = 0
#                 q̈ = -(k/m)q
# Solution: q(t) = A cos(ωt + φ) where ω = √(k/m)

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def harmonic_oscillator(t, y, k=1, m=1):
    q, qdot = y
    return [qdot, -(k/m)*q]

sol = solve_ivp(harmonic_oscillator, [0, 20], [1, 0], dense_output=True)
t = np.linspace(0, 20, 500)
plt.plot(t, sol.sol(t)[0])
plt.xlabel('t')
plt.ylabel('q(t)')
plt.title('Harmonic Oscillator (from Euler-Lagrange)')
plt.grid(True)
plt.show()
```

### Solving Numerically: Boundary Value Problems

```python
from scipy.integrate import solve_bvp

def shortest_path_bvp(a, b, t_span=(0, 1)):
    """
    Find shortest path from (0, a) to (1, b).

    For flat space, Euler-Lagrange gives q'' = 0.
    """
    def ode(t, y):
        # y = [q, q']
        # From E-L: q'' = 0
        q, qp = y
        return [qp, np.zeros_like(q)]

    def bc(ya, yb):
        # Boundary conditions: q(0) = a, q(1) = b
        return [ya[0] - a, yb[0] - b]

    # Initial guess
    t_guess = np.linspace(*t_span, 10)
    y_guess = np.vstack([
        a + (b-a)*t_guess,  # Linear interpolation for q
        np.ones_like(t_guess) * (b-a)  # Constant slope for q'
    ])

    sol = solve_bvp(ode, bc, t_guess, y_guess)
    return sol

# Solve for path from (0, 0) to (1, 2)
sol = shortest_path_bvp(0, 2)

t = np.linspace(0, 1, 100)
plt.plot(t, sol.sol(t)[0])
plt.xlabel('t')
plt.ylabel('q(t)')
plt.title('Shortest path (geodesic in flat space)')
plt.grid(True)
plt.show()
```

### Exercises

1. **Brachistochrone**: What's the fastest slide curve under gravity? (Famous problem!)
2. **Catenary**: What curve does a hanging chain make? (Minimize potential energy.)
3. **Verify Numerically**: Check that E-L solutions actually minimize action.

---

## Day 6.4: Least Action Reasoning

### Learning Objectives
- [ ] Define an action for reasoning
- [ ] Interpret reasoning paths variationally
- [ ] Connect to Week 5 dynamics

### The Big Idea

What if reasoning has an action principle?

```
S[path] = ∫ L(concept, velocity, t) dt
```

Where:
- **concept** = position in concept space
- **velocity** = rate of conceptual change
- **L** = something like (effort - relevance)

### A Simple Model

```python
def reasoning_lagrangian(q, dq, t, salience_field, effort_weight=1.0):
    """
    L = effort - relevance

    effort ∝ |dq/dt|²  (changing thoughts takes energy)
    relevance = salience at current position
    """
    effort = effort_weight * np.sum(dq**2)
    relevance = salience_field(q)
    return effort - relevance

def simulate_reasoning(start, goal, salience_field, t_span=(0, 5)):
    """
    Find the reasoning path from start to goal that minimizes action.
    """
    # Convert to boundary value problem
    # E-L: d/dt(∂L/∂q̇) = ∂L/∂q
    # With L = |q̇|² - V(q):
    # 2q̈ = ∇V(q)
    # q̈ = ½∇V(q)

    def numerical_gradient(V, q, h=1e-5):
        grad = np.zeros_like(q)
        for i in range(len(q)):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += h
            q_minus[i] -= h
            grad[i] = (V(q_plus) - V(q_minus)) / (2*h)
        return grad

    def ode(t, y):
        n = len(start)
        q = y[:n]
        qdot = y[n:]
        qddot = 0.5 * numerical_gradient(salience_field, q)
        return np.concatenate([qdot, qddot])

    def bc(ya, yb):
        n = len(start)
        return np.concatenate([
            ya[:n] - start,  # q(0) = start
            yb[:n] - goal    # q(T) = goal
        ])

    # Solve
    t_guess = np.linspace(*t_span, 20)
    n = len(start)

    # Linear interpolation guess
    q_guess = np.array([start + (goal - start) * ti / t_span[1] for ti in t_guess]).T
    qdot_guess = np.ones((n, len(t_guess))) * (goal - start).reshape(-1, 1) / t_span[1]
    y_guess = np.vstack([q_guess, qdot_guess])

    sol = solve_bvp(ode, bc, t_guess, y_guess, verbose=2)
    return sol
```

### Visualization

```python
def plot_reasoning_path(sol, salience_field, xlim=(-2, 2), ylim=(-2, 2)):
    """Visualize reasoning path on salience landscape."""
    # Background: salience field
    x = np.linspace(*xlim, 50)
    y = np.linspace(*ylim, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[salience_field(np.array([xi, yi])) for xi in x] for yi in y])

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Salience')

    # Reasoning path
    path = sol.sol(np.linspace(sol.x[0], sol.x[-1], 100))[:2]
    plt.plot(path[0], path[1], 'w-', linewidth=2, label='Reasoning path')
    plt.plot(path[0, 0], path[1, 0], 'go', markersize=10, label='Start')
    plt.plot(path[0, -1], path[1, -1], 'ro', markersize=10, label='Goal')

    plt.xlabel('Concept dimension 1')
    plt.ylabel('Concept dimension 2')
    plt.title('Least Action Reasoning Path')
    plt.legend()
    plt.show()
```

### Exercises

1. **Different Salience Fields**: Try peaked vs flat vs multi-peaked salience. How do paths change?
2. **Effort Weighting**: What happens if effort_weight is very high? Very low?
3. **Compare to Gradient Flow**: Is least-action path same as gradient descent?

---

## Day 6.5: Connection to Week 5

### Lagrangian → Hamiltonian → Phase Space

The Euler-Lagrange dynamics can be rewritten as:

```
Hamiltonian: H(q, p) = p·q̇ - L(q, q̇)
Hamilton's equations:
  dq/dt = ∂H/∂p
  dp/dt = -∂H/∂q
```

This is a dynamical system in phase space!

```python
def lagrangian_to_hamiltonian(L, q, qdot):
    """
    Convert Lagrangian to Hamiltonian.

    p = ∂L/∂q̇ (momentum)
    H = p·q̇ - L
    """
    # Numerical partial derivative
    h = 1e-6
    p = (L(q, qdot+h) - L(q, qdot-h)) / (2*h)
    H = p * qdot - L(q, qdot)
    return H, p

# For harmonic oscillator:
# L = ½mq̇² - ½kq²
# p = ∂L/∂q̇ = mq̇
# H = p·q̇ - L = mq̇² - ½mq̇² + ½kq² = ½mq̇² + ½kq² = ½p²/m + ½kq²
# This is total energy (kinetic + potential)!
```

### The Beautiful Connection

| Week 5 (Dynamical Systems) | Week 6 (Variational) |
|---------------------------|---------------------|
| Vector field f(x) | Euler-Lagrange equations |
| Fixed points | Equilibrium paths |
| Gradient systems | Energy minimization |
| Trajectories | Extremal paths |

**Same phenomena, different languages.**

---

## Week 6 Milestone

By the end of Week 6, you should have:

1. **Functional intuition**: Understand action as "path quality"
2. **Euler-Lagrange mastery**: Derive and solve for optimal paths
3. **Reasoning model**: Sketch of least-action reasoning
4. **Connection made**: See variational ↔ dynamical equivalence

### Success Criteria

```python
# You can:
# 1. Compute action for a given path
# 2. Solve Euler-Lagrange as BVP
# 3. Derive E-L for simple Lagrangians
# 4. Interpret least-action for reasoning

# You understand:
# - Optimal paths minimize action
# - E-L is a differential equation for optimal paths
# - Lagrangian ↔ Hamiltonian ↔ Phase space
# - Reasoning could be variational
```

---

## Research Log Entry

```markdown
## Week 6 Observations

### Variational Intuition
- Action = ∫(effort - relevance) makes sense for reasoning
- Lower action paths are "more natural"

### Euler-Lagrange Results
- Free particle: straight line
- Harmonic oscillator: sinusoid
- On concept space: [your observations]

### Key Insight
The variational framework provides:
1. A reason WHY paths exist (they minimize something)
2. A way to COMPUTE optimal paths
3. A CONNECTION to dynamics

### Questions for Week 7
- What if concept space is curved?
- How does metric affect geodesics?
- Is salience the metric or the potential?
```

---

→ [Week 7: Differential Geometry](week-07-geometry.md)
