# Week 1: Probabilistic Primitives

**Goal**: Build intuition for Beta distributions, Bayesian updating, and similarity — the building blocks of Thompson Sampling.

**Time**: 3-4 sessions × 30 min = 1.5-2 hours

**Milestone**: Implement `update_belief()` that takes (prior, observation) → posterior, visualized.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 1.1 | `01a-beta-distributions.ipynb` | 30 min | Plot, sample, update Beta distributions |
| 1.2 | `01b-bayesian-updating.ipynb` | 30 min | From prior → posterior with feedback |
| 1.3 | `01c-similarity-refresh.ipynb` | 30 min | Cosine similarity and k-NN |
| 1.4 | `01d-integration.ipynb` | 30 min | [EXTEND] Combine: belief + similarity = salience preview |

---

## Day 1.1: Beta Distributions

### Learning Objectives
- [ ] Plot Beta(α, β) for various α, β values
- [ ] Sample from Beta distributions
- [ ] Interpret α as "successes + 1", β as "failures + 1"
- [ ] Understand why Beta is conjugate to Bernoulli

### Linguistics Anchor: Language Acquisition

> When a child learns the word "dog," they don't memorize a definition. They accumulate evidence: "that's a dog" (+1 success), "no, that's a cat" (+1 failure). Their internal probability "is this a dog?" is a Beta distribution.
>
> α = 1 + times they correctly identified a dog
> β = 1 + times they incorrectly called something a dog
>
> Early on, the distribution is wide (uncertain). After hundreds of observations, it sharpens (confident).

### Core Exercise

```python
def plot_beta_evolution(reinforcements: list[bool]) -> None:
    """
    Given a sequence of True (success) / False (failure),
    plot how Beta(α, β) evolves.
    """
    # TODO: Implement
```

### Key Insight

The Beta distribution doesn't just store a probability — it stores *uncertainty about that probability*. Wide = unsure. Sharp = confident.

---

## Day 1.2: Bayesian Updating

### Learning Objectives
- [ ] Implement prior → likelihood → posterior update
- [ ] Visualize belief shift after each observation
- [ ] Understand why this is "learning"

### Linguistics Anchor: Phoneme Categories

> English speakers learn that [p] and [b] are different phonemes (minimal pair: "pat" vs "bat"). But a speaker of a language without this distinction starts with a uniform prior. Each time they hear feedback ("no, that was a /b/"), their posterior shifts.
>
> This is exactly Bayesian updating. Prior belief + evidence → updated belief.

### Core Exercise

```python
def update_belief(alpha: float, beta: float, success: bool) -> tuple[float, float]:
    """
    Update Beta parameters given an observation.
    """
    # TODO: Implement (it's trivially simple, but see it work)

def animate_learning(observations: list[bool]) -> Animation:
    """
    Create an animation showing the Beta distribution evolving.
    """
    # TODO: Implement
```

### Key Insight

The update rule is embarrassingly simple: success → α += 1, failure → β += 1. But the *implications* are profound: you're encoding all past evidence in two numbers.

---

## Day 1.3: Similarity Refresh

### Learning Objectives
- [ ] Compute cosine similarity between vectors
- [ ] Find k-nearest neighbors in embedding space
- [ ] Interpret "centrality" as mean similarity to neighbors

### Linguistics Anchor: Semantic Fields

> Words don't exist in isolation. "Dog" is close to "cat," "pet," "animal," "bark." Its meaning is defined by its *neighborhood*. A word that's close to many other words is semantically central. A word with few neighbors is peripheral (jargon, neologism).
>
> The same is true for rules. A rule semantically close to many others is "foundational." A rule far from the cluster is "niche."

### Core Exercise

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # TODO: Implement

def get_centrality(embedding: np.ndarray, corpus: np.ndarray, k: int = 5) -> float:
    """
    Compute centrality as mean similarity to k-nearest neighbors.
    """
    # TODO: Implement

def visualize_embedding_space(embeddings: np.ndarray, labels: list[str]) -> None:
    """
    2D projection showing centrality as node size.
    """
    # TODO: Implement with UMAP or t-SNE
```

### Key Insight

Centrality is emergent importance. You don't *declare* a rule foundational — you *observe* that it's close to many others in meaning space.

---

## Day 1.4: Integration [EXTEND]

### Learning Objectives
- [ ] Combine belief (Beta) with centrality (k-NN similarity)
- [ ] Preview the salience formula: `salience = confidence × (1 + α × centrality)`
- [ ] See why both components matter

### Why Both?

| Rule | Confidence | Centrality | Salience |
|------|------------|------------|----------|
| A | High (always reinforced) | Low (niche topic) | Medium |
| B | Medium (mixed feedback) | High (foundational) | Medium-High |
| C | High | High | **Very High** |
| D | Low | Low | Low |

Confidence alone misses semantic structure. Centrality alone ignores feedback. Together: emergent salience.

---

## Week 1 Milestone: `update_belief()` + Visualization

Deliverable:

```python
# You've implemented:
def update_belief(alpha, beta, success) -> tuple[float, float]: ...
def get_centrality(embedding, corpus, k) -> float: ...
def visualize_belief_evolution(observations) -> Animation: ...
```

You can now:
- Update a Beta distribution given feedback
- Compute centrality in embedding space
- Visualize how beliefs evolve

---

## Reflection Questions

1. Why is Beta(1, 1) uniform? What does this represent as a prior?
2. If a rule has α=100, β=2, how confident are we? What's the expected probability?
3. Why might a highly central rule still have low salience?

---

## [OPTIONAL DEPTH] Conjugate Priors

Why is Beta "conjugate" to Bernoulli? Short answer: the posterior is the same family as the prior. This means updates stay closed-form.

If you want the derivation: [link to appendix]

**Time estimate**: 15-20 min rabbit hole. Skip freely.

---

→ [Week 2: Thompson Sampling & Salience](week-02-thompson-salience.md)
