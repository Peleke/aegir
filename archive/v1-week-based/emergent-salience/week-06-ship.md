# Week 6: Integration & Paper

**Goal**: Ship the buildlog implementation and draft the paper.

**Time**: 3-4 sessions × 30 min = 1.5-2 hours

**Milestone**: Working `salience.py` in buildlog-template + paper outline.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 6.1 | `06a-integration.ipynb` | 30 min | Assemble all components |
| 6.2 | `06b-experiments.ipynb` | 30 min | Run experiments, generate figures |
| 6.3 | `06c-regret-analysis.ipynb` | 30 min | Formal regret bounds |
| 6.4 | `06d-paper-draft.ipynb` | 30 min | [EXTEND] Structure the arXiv paper |

---

## Day 6.1: Integration

### Learning Objectives
- [ ] Assemble all components into `salience.py`
- [ ] Write the public API
- [ ] Test against buildlog-template

### The Final Module

```python
# buildlog-template/src/buildlog/salience.py

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Types ---

class Tier(Enum):
    PERIPHERAL = "peripheral"
    EMERGING = "emerging"
    CORE = "core"
    FOUNDATIONAL = "foundational"
    CONSTITUTIONAL = "constitutional"

class Phase(Enum):
    NASCENT = "nascent"
    EMERGING = "emerging"
    STABLE = "stable"
    DECLINING = "declining"
    DORMANT = "dormant"

class Enforcement(Enum):
    BLOCK = "block"
    WARN = "warn"
    NOTIFY = "notify"

# --- Core Data Structures ---

@dataclass
class Rule:
    id: str
    text: str
    embedding: np.ndarray
    alpha: float = 1.0  # reinforcements + 1
    beta: float = 1.0   # contradictions + 1

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def reinforce(self) -> None:
        self.alpha += 1

    def contradict(self) -> None:
        self.beta += 1

@dataclass
class Constitution:
    rules: list["ConstitutionalRule"]
    preaxiomatic: set[str]

    @classmethod
    def from_yaml(cls, path: Path) -> "Constitution":
        ...

# --- Core Functions ---

def calculate_salience(
    rule: Rule,
    corpus: np.ndarray,
    k: int = 5,
    alpha_weight: float = 1.0
) -> float:
    """Calculate emergent salience for a rule."""
    ...

def assign_tiers(
    rules: list[Rule],
    corpus: np.ndarray,
    constitution: Constitution | None = None
) -> dict[str, Tier]:
    """Assign tiers based on salience percentiles."""
    ...

def thompson_select(
    rules: list[Rule],
    corpus: np.ndarray,
    alpha_weight: float = 1.0
) -> Rule:
    """Select a rule using Thompson Sampling weighted by salience."""
    ...

def check_constitutional(
    action: str,
    constitution: Constitution,
    overrides: set[str]
) -> tuple[bool, ConstitutionalRule | None]:
    """Check action against constitutional rules."""
    ...

# --- Dynamics ---

@dataclass
class DynamicsTracker:
    """Track learning dynamics across sessions."""
    ...

# --- Context-Aware (Advanced) ---

@dataclass
class LinUCBArm:
    """LinUCB arm for context-aware selection."""
    ...

def context_aware_select(
    rules: list[Rule],
    context: np.ndarray,
    arms: dict[str, LinUCBArm],
    global_weight: float = 0.3
) -> Rule:
    """Select rule using global salience + contextual relevance."""
    ...
```

### Integration Checklist

- [ ] All components compile
- [ ] Unit tests pass
- [ ] Integration with buildlog skills.py works
- [ ] Constitutional conflict detection works
- [ ] Tier display in render/skill.py works

---

## Day 6.2: Experiments

### Learning Objectives
- [ ] Design reproducible experiments
- [ ] Generate publication-quality figures
- [ ] Document experimental setup

### Experiment 1: Regret Comparison

Compare Thompson Sampling vs baselines:

```python
def run_regret_experiment(
    n_rules: int = 20,
    n_sessions: int = 500,
    true_probs: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Compare cumulative regret across algorithms.

    Algorithms:
    - Random selection
    - ε-greedy (ε=0.1)
    - UCB1
    - Thompson Sampling
    - Thompson + Salience
    """
    ...

# Generate figure
def plot_regret_comparison(results: pd.DataFrame) -> plt.Figure:
    """Publication-quality regret curves."""
    ...
```

### Experiment 2: Tier Emergence

Visualize how tiers stabilize over time:

```python
def run_tier_experiment(
    n_sessions: int = 200
) -> pd.DataFrame:
    """Track tier assignments over simulated sessions."""
    ...

def plot_tier_evolution(results: pd.DataFrame) -> plt.Figure:
    """Sankey or area chart showing tier flow."""
    ...
```

### Experiment 3: Context-Aware vs Global

```python
def run_context_experiment(
    n_contexts: int = 5,
    n_sessions_per_context: int = 100
) -> pd.DataFrame:
    """Compare context-blind vs context-aware selection."""
    ...
```

---

## Day 6.3: Regret Analysis

### Learning Objectives
- [ ] Understand the regret bound O(√(KT log K))
- [ ] Verify empirically
- [ ] Document assumptions

### The Regret Bound

For Thompson Sampling with K arms over T rounds:

```
E[Regret(T)] = O(√(KT log K))
```

This means:
- Regret grows sublinearly in T
- With more arms (K), regret grows as √K
- Logarithmic factor from uncertainty quantification

### Empirical Verification

```python
def verify_regret_bound(
    K_values: list[int] = [5, 10, 20, 50],
    T: int = 1000,
    n_trials: int = 50
) -> pd.DataFrame:
    """
    Run experiments and check if regret matches O(√(KT log K)).
    """
    results = []
    for K in K_values:
        for trial in range(n_trials):
            regret = run_thompson_experiment(K, T)
            theoretical = np.sqrt(K * T * np.log(K))
            results.append({
                'K': K, 'T': T, 'trial': trial,
                'regret': regret, 'theoretical': theoretical,
                'ratio': regret / theoretical
            })
    return pd.DataFrame(results)
```

### Key Assumptions

The bound assumes:
1. Rewards are bounded in [0, 1]
2. Arms are independent
3. Reward distributions are stationary (or slowly changing)

Document how buildlog satisfies (or relaxes) these.

---

## Day 6.4: Paper Draft [EXTEND]

### Learning Objectives
- [ ] Structure the paper
- [ ] Write abstract and introduction
- [ ] Outline remaining sections

### Paper Outline

```markdown
# Emergent Salience: RL-Backed Rule Hierarchies for AI Coding Assistants

## Abstract
- Problem: Static rule systems fail to adapt
- Approach: Contextual bandits with salience scoring
- Results: Sublinear regret, interpretable emergence
- Impact: Applicable to any AI assistant rule system

## 1. Introduction
- Motivation: AI coding assistants surface rules/skills
- Problem: Fixed hierarchies don't reflect actual usage
- Contribution: Formal RL framework for adaptive rule selection

## 2. Related Work
- Multi-armed bandits
- Thompson Sampling
- Safe RL / constrained MDPs
- Prior work on coding assistant design

## 3. Problem Formulation
- Contextual bandit setup
- Constitutional constraints
- Salience definition

## 4. Approach
- 4.1 Salience Scoring
- 4.2 Thompson Sampling with Beta Priors
- 4.3 Constitutional Rules (Safe RL)
- 4.4 Learning Dynamics
- 4.5 Context-Aware Selection (LinUCB)

## 5. Experiments
- 5.1 Regret Comparison
- 5.2 Tier Emergence
- 5.3 Context-Aware vs Global
- 5.4 Ablation: α weight sensitivity

## 6. Discussion
- When does emergence work?
- Failure modes
- Computational considerations

## 7. Conclusion
- Summary of contributions
- Future work (hierarchical bandits, user-specific models)

## References
```

### Writing the Abstract

Template:

> Rule systems in AI coding assistants typically rely on fixed hierarchies
> that fail to reflect actual developer usage patterns. We present
> **Emergent Salience**, a contextual bandit framework where rules *earn*
> their tier through reinforcement learning. Our approach combines
> Thompson Sampling for exploration-exploitation, semantic centrality
> for foundational importance, and constitutional constraints for safe RL.
> Experiments show [specific results]. The framework achieves O(√(KT log K))
> regret while maintaining interpretable rule hierarchies. We release an
> open-source implementation integrated with [buildlog-template].

---

## Week 6 Milestone: Ship It

Deliverables:

1. **Code**: `buildlog-template/src/buildlog/salience.py` (working, tested)
2. **Figures**: Publication-quality plots for regret, tier evolution, context comparison
3. **Paper**: Complete outline + abstract + introduction draft

---

## Publishing Path

### Option 1: arXiv + Blog

1. Write up as arXiv preprint
2. Companion blog post with interactive demos
3. Share on Twitter/HN for visibility

### Option 2: Workshop Paper

Target venues:
- NeurIPS workshops (ML for Code, Safe RL)
- ICML workshops
- ACL workshops (if emphasizing the linguistics angle)

### Option 3: Full Conference

If experiments are strong:
- ICML, NeurIPS, ICLR (main track)
- AAAI, IJCAI (broader AI)

---

## Reflection: The Journey

You started with:
- Systems programming (strong)
- Probabilistic methods (gap)
- "I'll never sit through a traditional course"

You now have:
- Thompson Sampling intuition + implementation
- Beta distributions as learning representations
- Constitutional constraints as safe RL
- Learning dynamics (velocity, acceleration, phase)
- Context-aware selection with LinUCB
- A publishable implementation + paper draft

The gap is closed. The artifact is shipped.

---

## What's Next?

After this curriculum:

1. **Iterate on the paper** — get feedback, run more experiments
2. **Deploy in buildlog** — real usage data
3. **Extend** — hierarchical bandits? user-specific models? multi-agent?
4. **Teach** — publish the notebooks as educational content

You're now "the kind of person who can plan, write, and publish deep learning research seriously."

---

## Final Reflection Questions

1. What was the hardest concept? Did it click eventually?
2. Which linguistics anchor was most useful?
3. What would you add/remove from this curriculum?
4. What's your next research question?

---

**Curriculum Complete.**
