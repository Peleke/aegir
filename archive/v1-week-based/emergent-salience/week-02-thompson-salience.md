# Week 2: Thompson Sampling & Salience

**Goal**: Implement Thompson Sampling for rule selection and the full salience formula.

**Time**: 4-5 sessions × 30 min = 2-2.5 hours

**Milestone**: Working `calculate_salience()` and `sample_rule()` functions.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 2.1 | `02a-thompson-sampling.ipynb` | 30 min | Sample from posteriors to select arms |
| 2.2 | `02b-exploration-exploitation.ipynb` | 30 min | Why Thompson beats ε-greedy and UCB |
| 2.3 | `02c-salience-formula.ipynb` | 30 min | `salience = confidence × (1 + α × centrality)` |
| 2.4 | `02d-emergent-tiers.ipynb` | 30 min | Percentile-based hierarchy |
| 2.5 | `02e-integration.ipynb` | 30 min | [EXTEND] Full rule selector |

---

## Day 2.1: Thompson Sampling

### Learning Objectives
- [ ] Understand the multi-armed bandit problem
- [ ] Implement Thompson Sampling with Beta priors
- [ ] See how sampling induces exploration

### Linguistics Anchor: Allophonic Variation

> In English, the [t] in "top" is aspirated [tʰ], but in "stop" it's unaspirated [t]. A learner doesn't memorize this rule explicitly — they sample from their hypothesis space, get feedback, and the correct allophonic distribution *emerges*.
>
> Thompson Sampling is the same: sample a hypothesis (which rule is best?), test it, update. The right answer emerges from exploration.

### The Algorithm

```python
def thompson_sample(rules: list[Rule]) -> Rule:
    """
    Select a rule by sampling from each rule's Beta posterior.

    1. For each rule, sample θ ~ Beta(α, β)
    2. Return the rule with highest sampled θ
    """
    samples = []
    for rule in rules:
        theta = np.random.beta(rule.alpha, rule.beta)
        samples.append((theta, rule))
    return max(samples, key=lambda x: x[0])[1]
```

### Key Insight

Sampling automatically balances exploration and exploitation:
- High-confidence rules (sharp Beta) → samples clustered near mean
- Low-confidence rules (wide Beta) → samples spread out → occasionally win → get explored

No ε parameter. No upper confidence bounds. Just sampling.

---

## Day 2.2: Why Thompson Beats the Alternatives

### Learning Objectives
- [ ] Compare Thompson vs ε-greedy vs UCB
- [ ] Understand regret curves
- [ ] See why Thompson excels in non-stationary environments

### The Competitors

| Algorithm | Exploration Strategy | Weakness |
|-----------|---------------------|----------|
| **ε-greedy** | Random exploration (ε% of time) | Wastes exploration on known-bad arms |
| **UCB** | Optimism under uncertainty | Deterministic, can get stuck |
| **Thompson** | Probability matching | (None significant for our use case) |

### Simulation Exercise

```python
def compare_algorithms(n_rounds: int = 1000) -> pd.DataFrame:
    """
    Run all three algorithms on same bandit problem.
    Plot cumulative regret over time.
    """
    # TODO: Implement
```

### Key Insight

Thompson Sampling is **probability matching**: the probability of selecting an arm equals the probability that arm is optimal. This is Bayes-optimal exploration.

---

## Day 2.3: The Salience Formula

### Learning Objectives
- [ ] Implement `calculate_salience()`
- [ ] Understand the role of α (centrality weight)
- [ ] Tune α for your use case

### The Formula

```
salience = confidence × (1 + α × centrality)
```

Where:
- **confidence** = `α / (α + β)` (expected value of Beta)
- **centrality** = mean cosine similarity to k-nearest neighbors
- **α** (the weight) = hyperparameter, typically 0.5-2.0

### Implementation

```python
@dataclass
class Rule:
    text: str
    embedding: np.ndarray
    alpha: float = 1.0  # reinforcements + 1
    beta: float = 1.0   # contradictions + 1

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

def calculate_salience(
    rule: Rule,
    corpus_embeddings: np.ndarray,
    k: int = 5,
    alpha_weight: float = 1.0
) -> float:
    """
    Calculate emergent salience for a rule.
    """
    centrality = get_centrality(rule.embedding, corpus_embeddings, k)
    return rule.confidence * (1 + alpha_weight * centrality)
```

### Why This Formula?

- **Confidence alone**: Misses semantic structure. A niche rule with 100% success isn't foundational.
- **Centrality alone**: Misses feedback. A central rule that always fails isn't useful.
- **Multiplicative**: Both must be present. Zero confidence → zero salience.
- **Additive centrality boost**: Centrality enhances, doesn't dominate.

---

## Day 2.4: Emergent Tiers

### Learning Objectives
- [ ] Implement percentile-based tier assignment
- [ ] Understand why fixed thresholds fail
- [ ] Visualize tier distribution

### The Problem with Fixed Thresholds

```python
# Bad: fixed thresholds
if salience > 0.9:
    tier = "FOUNDATIONAL"
elif salience > 0.7:
    tier = "CORE"
# ...
```

What if all rules have salience 0.3-0.5? No FOUNDATIONAL rules, ever.

### Percentile-Based Tiers

```python
def assign_tiers(rules: list[Rule], corpus: np.ndarray) -> dict[Rule, Tier]:
    """
    Assign tiers based on salience percentiles.
    """
    saliences = [calculate_salience(r, corpus) for r in rules]
    percentiles = scipy.stats.rankdata(saliences, method='average') / len(saliences)

    tiers = {}
    for rule, pct in zip(rules, percentiles):
        if pct < 0.25:
            tiers[rule] = Tier.PERIPHERAL
        elif pct < 0.50:
            tiers[rule] = Tier.EMERGING
        elif pct < 0.90:
            tiers[rule] = Tier.CORE
        else:
            tiers[rule] = Tier.FOUNDATIONAL
    return tiers
```

### Key Insight

The hierarchy is always populated. There's always a top 10%, always a bottom 25%. The tiers are *relative*, not absolute.

---

## Day 2.5: Full Rule Selector [EXTEND]

### Learning Objectives
- [ ] Combine Thompson Sampling with salience weighting
- [ ] Implement the full selection algorithm
- [ ] Run on simulated sessions

### The Algorithm

```python
def select_rule(
    rules: list[Rule],
    corpus: np.ndarray,
    alpha_weight: float = 1.0
) -> Rule:
    """
    Select a rule using Thompson Sampling weighted by salience.
    """
    samples = []
    for rule in rules:
        # Sample from posterior
        theta = np.random.beta(rule.alpha, rule.beta)
        # Weight by centrality
        centrality = get_centrality(rule.embedding, corpus)
        score = theta * (1 + alpha_weight * centrality)
        samples.append((score, rule))
    return max(samples, key=lambda x: x[0])[1]
```

---

## Week 2 Milestone: `calculate_salience()` + `select_rule()`

Deliverable:

```python
# You've implemented:
def calculate_salience(rule, corpus, k, alpha_weight) -> float: ...
def assign_tiers(rules, corpus) -> dict[Rule, Tier]: ...
def select_rule(rules, corpus, alpha_weight) -> Rule: ...
```

This is the core of the buildlog implementation. You're ~40% done with `salience.py`.

---

## Regret Intuition (Preview)

We'll formalize this in Week 6, but intuitively:

**Regret** = how much reward you lost by not always picking the best arm.

Thompson Sampling achieves regret O(√(KT log K)) where:
- K = number of arms (rules)
- T = number of rounds (sessions)

This is sublinear — regret grows slower than time. You converge.

---

## Reflection Questions

1. If you always sample the highest-confidence rule, what happens to exploration?
2. Why multiply confidence × centrality instead of adding them?
3. What happens to tier assignments as more rules are added to the corpus?

---

## [OPTIONAL DEPTH] Regret Bound Proof

The proof that Thompson Sampling achieves O(√(KT log K)) regret is beautiful but involved. It uses properties of Beta-Binomial conjugacy and a "frequentist regret, Bayesian analysis" trick.

If you want the full proof: [link to appendix]

**Time estimate**: 45-60 min rabbit hole. Skip freely.

---

→ [Week 3: Constitutional Rules](week-03-constitutional.md)
