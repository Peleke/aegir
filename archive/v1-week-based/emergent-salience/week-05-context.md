# Week 5: Context-Aware Selection

**Goal**: Learn which rules apply *where* — context-dependent rule selection with LinUCB.

**Time**: 4-5 sessions × 30 min = 2-2.5 hours

**Milestone**: Working LinUCB implementation for context-aware rule surfacing.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 5.1 | `05a-contextual-bandits.ipynb` | 30 min | From bandits to contextual bandits |
| 5.2 | `05b-linucb.ipynb` | 30 min | Linear Upper Confidence Bound |
| 5.3 | `05c-task-context.ipynb` | 30 min | Embedding task/file context |
| 5.4 | `05d-context-aware-selection.ipynb` | 30 min | Full implementation |
| 5.5 | `05e-evaluation.ipynb` | 30 min | [EXTEND] Measuring context-aware regret |

---

## Day 5.1: Contextual Bandits

### Learning Objectives
- [ ] Understand the contextual bandit formulation
- [ ] See why standard Thompson Sampling ignores context
- [ ] Motivate context-dependent selection

### Linguistics Anchor: Pragmatics

> "It's cold in here" can be a statement (reporting temperature) or a request (close the window). The *context* determines the interpretation: who's speaking, where, to whom.
>
> Pragmatics is the study of context-dependent meaning. A sentence's meaning isn't fixed — it depends on situation.
>
> Rules are the same. "Write tests first" is critical for core business logic, less so for one-off scripts. Context determines relevance.

### The Limitation of Basic Thompson Sampling

```python
# Basic Thompson: same distribution regardless of context
theta = np.random.beta(rule.alpha, rule.beta)
```

Problem: A rule might be great for Python files but irrelevant for Markdown. Basic Thompson can't learn this.

### The Contextual Bandit Setup

```
Context x → select arm a → receive reward r

Goal: Learn a policy π(x) → a that maximizes expected reward
      given context x
```

---

## Day 5.2: LinUCB

### Learning Objectives
- [ ] Understand the LinUCB algorithm
- [ ] Implement the linear reward model
- [ ] See how context flows through selection

### The Algorithm

LinUCB models reward as linear in context:

```
E[r | x, a] = θ_a^T · x
```

Where:
- x = context vector (e.g., file type, task embedding)
- θ_a = learned weight vector for arm a
- r = reward (reinforce/contradict)

### Implementation

```python
@dataclass
class LinUCBArm:
    A: np.ndarray  # d×d matrix
    b: np.ndarray  # d×1 vector
    alpha: float = 1.0  # exploration parameter

    def __init__(self, d: int, alpha: float = 1.0):
        self.A = np.eye(d)
        self.b = np.zeros(d)
        self.alpha = alpha

    def get_ucb(self, context: np.ndarray) -> float:
        """Compute upper confidence bound for this arm given context."""
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        # Mean reward estimate
        mean = theta @ context

        # Confidence bound (uncertainty)
        std = self.alpha * np.sqrt(context @ A_inv @ context)

        return mean + std

    def update(self, context: np.ndarray, reward: float) -> None:
        """Update arm parameters given observation."""
        self.A += np.outer(context, context)
        self.b += reward * context

def linucb_select(arms: list[LinUCBArm], context: np.ndarray) -> int:
    """Select arm with highest UCB."""
    ucbs = [arm.get_ucb(context) for arm in arms]
    return np.argmax(ucbs)
```

### Key Insight

LinUCB learns *when* each rule is relevant, not just *if* it's good overall. A rule can have low global confidence but high context-specific relevance.

---

## Day 5.3: Task Context Embedding

### Learning Objectives
- [ ] Design the context vector
- [ ] Embed file type, task description, recent history
- [ ] Normalize and combine context features

### What Goes in the Context?

| Feature | Type | Dimension | Example |
|---------|------|-----------|---------|
| File type | one-hot | ~10 | [0, 1, 0, ...] for .py |
| Task embedding | dense | 384 | sentence-transformers output |
| Recent rules | sparse | n_rules | which rules were surfaced recently |
| Time features | scalar | 2-3 | time of day, day of week |

### Implementation

```python
def build_context(
    file_path: Path | None,
    task_description: str,
    recent_rules: list[str],
    embedder: SentenceTransformer
) -> np.ndarray:
    """
    Build context vector for LinUCB.
    """
    features = []

    # File type (one-hot)
    file_type = get_file_type(file_path)  # .py, .ts, .md, etc.
    file_vec = one_hot_encode(file_type, FILE_TYPES)
    features.append(file_vec)

    # Task embedding (dense)
    task_vec = embedder.encode(task_description)
    task_vec = task_vec / np.linalg.norm(task_vec)  # normalize
    features.append(task_vec)

    # Recent rules (sparse)
    recent_vec = encode_recent_rules(recent_rules, RULE_IDS)
    features.append(recent_vec)

    return np.concatenate(features)
```

### Dimensionality Concerns

Context dimension d matters for LinUCB:
- Too low: can't capture relevant features
- Too high: slow updates, needs more data

Typical: d = 50-200 after dimensionality reduction.

---

## Day 5.4: Full Context-Aware Selection

### Learning Objectives
- [ ] Integrate LinUCB with existing salience system
- [ ] Handle cold start (new contexts)
- [ ] Combine global and context-specific signals

### Hybrid Approach

```python
def context_aware_select(
    rules: list[Rule],
    context: np.ndarray,
    linucb_arms: dict[str, LinUCBArm],
    global_weight: float = 0.3
) -> Rule:
    """
    Select rule using both global salience and contextual relevance.

    score = global_weight * salience + (1 - global_weight) * context_score
    """
    scores = []
    for rule in rules:
        global_score = calculate_salience(rule, ...)

        if rule.id in linucb_arms:
            context_score = linucb_arms[rule.id].get_ucb(context)
        else:
            context_score = 0.5  # cold start prior

        combined = global_weight * global_score + (1 - global_weight) * context_score
        scores.append((combined, rule))

    return max(scores, key=lambda x: x[0])[1]
```

### Cold Start Handling

New contexts haven't been seen before. Options:
1. Fall back to global salience
2. Use context similarity to known contexts
3. Explore aggressively (high α in LinUCB)

---

## Day 5.5: Evaluation [EXTEND]

### Learning Objectives
- [ ] Define contextual regret
- [ ] Evaluate context-aware vs context-blind
- [ ] Generate plots for the paper

### Contextual Regret

```python
def contextual_regret(
    history: list[tuple[Context, Rule, Reward]],
    oracle_policy: Callable[[Context], Rule]
) -> float:
    """
    Compute regret against an oracle that knows the best rule per context.
    """
    regret = 0.0
    for context, chosen_rule, reward in history:
        best_rule = oracle_policy(context)
        best_reward = expected_reward(best_rule, context)
        regret += best_reward - reward
    return regret
```

### Experiment Design

For the paper:
1. Simulate sessions with varying contexts (file types, task types)
2. Compare: random, global Thompson, contextual LinUCB
3. Plot cumulative regret over sessions
4. Show that contextual learning outperforms global

---

## Week 5 Milestone: LinUCB + Context Embedding

Deliverable:

```python
# You've implemented:
class LinUCBArm: ...
def linucb_select(arms, context) -> int: ...
def build_context(file, task, recent, embedder) -> np.ndarray: ...
def context_aware_select(rules, context, arms, weight) -> Rule: ...
```

This is the advanced feature for buildlog — "which rules apply where."

---

## Why This Matters

Context-aware selection is what makes the system *smart*:

| Scenario | Context-Blind | Context-Aware |
|----------|---------------|---------------|
| Python file | Surfaces all rules | Surfaces Python-specific rules |
| Quick fix | Same rules as refactor | Fewer, more targeted rules |
| Test file | Same rules as prod | Testing-focused rules |

The system learns *when* rules are relevant, not just *if*.

---

## Reflection Questions

1. Why use LinUCB instead of Thompson Sampling with context features?
2. What happens if the context embedding is too high-dimensional?
3. How would you handle a completely new file type the system has never seen?

---

## [OPTIONAL DEPTH] Thompson Sampling with Linear Rewards

There's a Thompson Sampling variant for contextual bandits using linear reward models. Instead of sampling θ ~ Beta, you sample θ ~ N(μ, Σ) from the posterior over the weight vector.

This combines the benefits of Thompson (probability matching) with the context-awareness of LinUCB.

**Time estimate**: 45-60 min rabbit hole. Skip freely.

---

→ [Week 6: Integration & Paper](week-06-ship.md)
