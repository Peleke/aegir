# Week 4: Learning Dynamics

**Goal**: Measure *how* rules are learning, not just *what* they've learned.

**Time**: 3-4 sessions × 30 min = 1.5-2 hours

**Milestone**: Implement velocity/acceleration tracking with phase detection.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 4.1 | `04a-velocity.ipynb` | 30 min | First derivative of confidence |
| 4.2 | `04b-acceleration.ipynb` | 30 min | Second derivative |
| 4.3 | `04c-phase-transitions.ipynb` | 30 min | Rule lifecycle stages |
| 4.4 | `04d-detection.ipynb` | 30 min | [EXTEND] Detecting emergence in real-time |

---

## Day 4.1: Confidence Velocity

### Learning Objectives
- [ ] Compute velocity as Δconfidence over time window
- [ ] Interpret positive/negative velocity
- [ ] Visualize velocity over rule lifecycle

### Linguistics Anchor: Grammaticalization

> Words change category over time. "While" was a noun (a period of time), then became a conjunction. This is grammaticalization — a slow drift in function.
>
> The *velocity* of grammaticalization can be measured: how fast is "literally" shifting from literal to emphatic? Usage data shows acceleration in the 2010s.
>
> Rules undergo the same process. A rule might drift from CORE to FOUNDATIONAL, or decay from CORE to PERIPHERAL.

### The Formula

```python
def compute_velocity(
    confidence_history: list[float],
    window: int = 10
) -> float:
    """
    Compute velocity as rate of change in confidence.

    velocity = (confidence_now - confidence_past) / window
    """
    if len(confidence_history) < window:
        return 0.0
    recent = confidence_history[-1]
    past = confidence_history[-window]
    return (recent - past) / window
```

### Interpretation

| Velocity | Meaning |
|----------|---------|
| **v > 0** | Rule gaining trust (being reinforced) |
| **v ≈ 0** | Rule stable (equilibrium) |
| **v < 0** | Rule losing trust (being contradicted) |

---

## Day 4.2: Confidence Acceleration

### Learning Objectives
- [ ] Compute acceleration as Δvelocity over time
- [ ] Understand what acceleration reveals
- [ ] Detect inflection points

### The Formula

```python
def compute_acceleration(
    confidence_history: list[float],
    window: int = 10
) -> float:
    """
    Compute acceleration as rate of change in velocity.

    acceleration = (velocity_now - velocity_past) / window
    """
    if len(confidence_history) < 2 * window:
        return 0.0

    v_now = compute_velocity(confidence_history, window)
    v_past = compute_velocity(confidence_history[:-window], window)
    return (v_now - v_past) / window
```

### Interpretation

| Acceleration | Meaning |
|--------------|---------|
| **a > 0, v > 0** | Growth accelerating (rapid emergence) |
| **a < 0, v > 0** | Growth slowing (approaching equilibrium) |
| **a > 0, v < 0** | Decay slowing (bottoming out) |
| **a < 0, v < 0** | Decay accelerating (rapid collapse) |

### Key Insight

Acceleration detects *inflection points*. A rule might have v > 0 (still growing), but a < 0 (growth slowing). This is "second-order" learning: the rule is approaching its equilibrium.

---

## Day 4.3: Phase Transitions

### Learning Objectives
- [ ] Define lifecycle phases for rules
- [ ] Implement phase detection logic
- [ ] Visualize phase transitions

### The Rule Lifecycle

```
NASCENT → EMERGING → STABLE → [FOUNDATIONAL or DECLINING]
   │         │          │              │
   v         v          v              v
 wide β    narrowing   sharp β      varies
 v ≈ 0     v > 0       v ≈ 0        v < 0
```

### Phase Definitions

```python
class Phase(Enum):
    NASCENT = "nascent"       # New, few observations
    EMERGING = "emerging"     # Gaining confidence
    STABLE = "stable"         # Equilibrium reached
    DECLINING = "declining"   # Losing confidence
    DORMANT = "dormant"       # No recent activity

def detect_phase(
    rule: Rule,
    velocity: float,
    acceleration: float,
    observation_count: int
) -> Phase:
    """
    Detect current lifecycle phase.
    """
    if observation_count < 5:
        return Phase.NASCENT
    if velocity > 0.01 and acceleration >= 0:
        return Phase.EMERGING
    if abs(velocity) < 0.005:
        return Phase.STABLE
    if velocity < -0.01:
        return Phase.DECLINING
    return Phase.DORMANT
```

### Visualization

```python
def plot_lifecycle(rule: Rule, history: list[Observation]) -> None:
    """
    Plot confidence over time with phase annotations.
    """
    # TODO: Implement with phase-colored regions
```

---

## Day 4.4: Real-Time Detection [EXTEND]

### Learning Objectives
- [ ] Implement streaming phase detection
- [ ] Trigger alerts on phase transitions
- [ ] Build the full dynamics tracker

### Implementation

```python
@dataclass
class DynamicsTracker:
    history: dict[str, list[float]] = field(default_factory=dict)
    window: int = 10

    def record(self, rule_id: str, confidence: float) -> None:
        """Record a confidence observation."""
        if rule_id not in self.history:
            self.history[rule_id] = []
        self.history[rule_id].append(confidence)

    def get_dynamics(self, rule_id: str) -> RuleDynamics:
        """Compute current dynamics for a rule."""
        history = self.history.get(rule_id, [])
        return RuleDynamics(
            velocity=compute_velocity(history, self.window),
            acceleration=compute_acceleration(history, self.window),
            phase=detect_phase(...)
        )

    def detect_transitions(self) -> list[Transition]:
        """Find rules that changed phase since last check."""
        # TODO: Implement
```

### Interesting Patterns

**Rapid Emergence**: A rule goes from NASCENT to FOUNDATIONAL in few sessions. This might indicate:
- A genuinely critical rule
- An easy-to-test rule (lots of feedback)
- Possible overfitting to recent context

**Oscillation**: A rule alternates between EMERGING and DECLINING. This might indicate:
- Context-dependent rule (applies in some situations, not others)
- Conflicting usage patterns
- Need for rule refinement

---

## Week 4 Milestone: Dynamics Tracker

Deliverable:

```python
# You've implemented:
def compute_velocity(history, window) -> float: ...
def compute_acceleration(history, window) -> float: ...
def detect_phase(rule, v, a, n) -> Phase: ...
class DynamicsTracker: ...
```

You can now track *how* rules learn, not just what they've learned.

---

## Why This Matters

Static salience tells you where rules are *now*. Dynamics tell you where they're *going*.

| Metric | Question Answered |
|--------|-------------------|
| Confidence | How much do we trust this rule? |
| Velocity | Is trust growing or shrinking? |
| Acceleration | Is the trend strengthening or weakening? |
| Phase | What stage of lifecycle is the rule in? |

For the paper, dynamics provide the "learning curves" that make the system publishable.

---

## Reflection Questions

1. A rule has v > 0 but a < 0. What's happening? What do you predict next?
2. How would you detect a "false equilibrium" (stable but at wrong confidence)?
3. What window size should you use? What are the tradeoffs?

---

## [OPTIONAL DEPTH] Kalman Filtering

For noisy observations, you might want to smooth the confidence estimates. Kalman filtering provides optimal linear smoothing.

For our use case, simple moving windows are probably sufficient. But if you want online optimal estimation: [link to appendix]

**Time estimate**: 40-50 min rabbit hole. Skip freely.

---

→ [Week 5: Context-Aware Selection](week-05-context.md)
