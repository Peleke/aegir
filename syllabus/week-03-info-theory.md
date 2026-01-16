# Week 3: Information Theory

**Goal**: Entropy as the language of uncertainty.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Compute mutual information between concept clusters.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 3.1 | 03a-entropy-intuition | 60 min | Entropy as surprise |
| 3.2 | 03b-mutual-information | 60 min | Shared structure |
| 3.3 | 03c-compression-as-understanding | 60 min | Good concepts compress |
| 3.4 | Exercises | 45 min | Compute MI on real data |
| 3.5 | Wiki deep dive | 45 min | Read information-theory.md |

---

## Day 3.1: Entropy as Surprise

### Learning Objectives
- [ ] Define entropy as expected surprise
- [ ] Compute entropy for various distributions
- [ ] Visualize entropy's behavior

### The Core Idea

**Information = Surprise**

- "The sun rose today" → Low information (expected)
- "It snowed in Miami in July" → High information (surprising!)

Mathematically:
```
Information of event x: I(x) = -log₂(p(x))
```

Rare events carry more information.

### Entropy: Average Surprise

```
H(X) = -Σ p(x) log₂ p(x) = E[I(X)]
```

**Intuition**: Entropy measures "how surprised you'll be on average."

### Code

```python
import numpy as np

def entropy(probs):
    """Compute entropy in bits."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

# Examples
print(f"Fair coin: {entropy([0.5, 0.5]):.2f} bits")       # 1.0
print(f"Loaded coin (90/10): {entropy([0.9, 0.1]):.2f} bits")  # 0.47
print(f"Certain: {entropy([1.0]):.2f} bits")              # 0.0
```

### Visualization

```python
import matplotlib.pyplot as plt

ps = np.linspace(0.001, 0.999, 100)
entropies = [entropy([p, 1-p]) for p in ps]

plt.figure(figsize=(8, 5))
plt.plot(ps, entropies)
plt.xlabel('P(heads)')
plt.ylabel('Entropy (bits)')
plt.title('Entropy of a Binary Variable')
plt.axhline(1, color='gray', linestyle='--', label='Maximum (fair)')
plt.legend()
plt.grid(True)
plt.show()
```

### Exercises

1. **Dice Entropy**: What's the entropy of a fair 6-sided die? A loaded die where 6 has probability 0.5?
2. **Maximum Entropy**: Prove that uniform distribution maximizes entropy for a given number of outcomes.
3. **Conditional Entropy**: Compute H(Y|X) for a simple joint distribution.

---

## Day 3.2: Mutual Information

### Learning Objectives
- [ ] Define mutual information as shared structure
- [ ] Compute MI from joint probabilities
- [ ] Understand MI ≥ 0 and MI = 0 iff independence

### The Core Idea

**Mutual Information = How much knowing X tells you about Y**

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X,Y)
```

**Intuition**:
- If X and Y are independent: I(X;Y) = 0
- If X determines Y completely: I(X;Y) = H(Y)

### Code

```python
def mutual_information(joint_probs):
    """
    Compute MI from joint probability matrix.
    joint_probs[i,j] = P(X=i, Y=j)
    """
    px = joint_probs.sum(axis=1)  # Marginal of X
    py = joint_probs.sum(axis=0)  # Marginal of Y

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(joint_probs.flatten())

    return hx + hy - hxy

# Independent variables
independent = np.array([[0.25, 0.25],
                        [0.25, 0.25]])
print(f"MI (independent): {mutual_information(independent):.4f}")  # ~0

# Correlated variables
correlated = np.array([[0.4, 0.1],
                       [0.1, 0.4]])
print(f"MI (correlated): {mutual_information(correlated):.4f}")  # > 0

# Perfectly correlated
perfect = np.array([[0.5, 0.0],
                    [0.0, 0.5]])
print(f"MI (perfect): {mutual_information(perfect):.4f}")  # = H(X) = H(Y)
```

### Why MI Matters for Concepts

Two concepts that **co-occur** (activate together) have high MI.

```
P(contract_testing, API) = high  → High MI
P(contract_testing, cooking) = low  → Low MI
```

This could define "relatedness" without explicit edges!

### Visualization: MI Matrix

```python
import seaborn as sns

# Compute MI between all pairs of concept indicators
n_concepts = len(concepts)
mi_matrix = np.zeros((n_concepts, n_concepts))

for i in range(n_concepts):
    for j in range(n_concepts):
        joint = compute_joint(concepts[i], concepts[j])
        mi_matrix[i, j] = mutual_information(joint)

plt.figure(figsize=(10, 8))
sns.heatmap(mi_matrix, xticklabels=concept_names, yticklabels=concept_names, cmap='viridis')
plt.title('Mutual Information Between Concepts')
plt.show()
```

### Exercises

1. **Data Processing Inequality**: Show that I(X;Y) ≥ I(X;f(Y)) for any function f.
2. **MI Estimation**: Compute MI between word co-occurrences in a corpus.
3. **Pointwise MI**: Implement PMI and compare to MI.

---

## Day 3.3: Compression = Understanding

### Learning Objectives
- [ ] Connect compression to understanding
- [ ] Use entropy as a quality metric for clusters
- [ ] Evaluate concept quality

### The Core Idea

**A good concept is a good compression.**

If you can describe many experiences with one concept, you've compressed.

```
Good: "contract testing" compresses {articles, code, errors, patterns}
Bad: "stuff" compresses nothing (too broad, high entropy)
Bad: "this specific test at 3pm" (too narrow, no generalization)
```

### Shannon's Source Coding Theorem

You can't compress below entropy.

→ **Concepts carve out low-entropy regions** of experience space.

### Cluster Entropy

```python
def cluster_entropy(cluster_embeddings, n_bins=50):
    """
    Estimate entropy of a cluster in embedding space.
    Lower entropy = more coherent cluster = better concept.
    """
    # Use histogram estimation for each dimension
    entropies = []
    for dim in range(cluster_embeddings.shape[1]):
        hist, _ = np.histogram(cluster_embeddings[:, dim], bins=n_bins, density=True)
        hist = hist[hist > 0]
        # Convert density to probability
        bin_width = (cluster_embeddings[:, dim].max() - cluster_embeddings[:, dim].min()) / n_bins
        probs = hist * bin_width
        probs = probs / probs.sum()  # Normalize
        entropies.append(entropy(probs))

    return np.mean(entropies)
```

### Concept Quality Metrics

```python
def concept_quality(cluster_embeddings, all_embeddings):
    """
    Evaluate how good a concept cluster is.

    Good concepts have:
    - Low internal entropy (coherent)
    - High separation from other points (distinct)
    """
    internal_entropy = cluster_entropy(cluster_embeddings)

    # Separation: average distance to non-cluster points
    from sklearn.metrics.pairwise import cosine_distances
    cluster_center = cluster_embeddings.mean(axis=0)
    distances = cosine_distances([cluster_center], all_embeddings)[0]
    separation = distances.mean()

    # Quality score (lower entropy, higher separation = better)
    return separation / (internal_entropy + 1e-6)
```

### Exercises

1. **Entropy vs Size**: Plot cluster entropy vs cluster size. Is there a correlation?
2. **Bad Concepts**: Create artificially bad concepts (random points). What's their entropy?
3. **Compression Ratio**: If each article is 1000 tokens and each concept is 1 token, what's the "compression ratio"?

---

## Day 3.4-3.5: Exercises & Deep Dive

### Compute MI on Real Data

```python
from collections import Counter

def estimate_mi_from_cooccurrence(texts, word1, word2, window=5):
    """
    Estimate MI between two words from co-occurrence in texts.
    """
    # Count occurrences
    count_w1 = 0
    count_w2 = 0
    count_both = 0
    count_total = 0

    for text in texts:
        words = text.lower().split()
        for i, word in enumerate(words):
            count_total += 1
            if word == word1:
                count_w1 += 1
                # Check window for word2
                window_words = words[max(0, i-window):i] + words[i+1:min(len(words), i+window+1)]
                if word2 in window_words:
                    count_both += 1
            if word == word2:
                count_w2 += 1

    # Compute probabilities
    p_w1 = count_w1 / count_total
    p_w2 = count_w2 / count_total
    p_both = count_both / count_total if count_both > 0 else 1e-10

    # PMI
    pmi = np.log2(p_both / (p_w1 * p_w2 + 1e-10))
    return pmi

# Test on some texts
texts = [...]  # Your corpus
print(estimate_mi_from_cooccurrence(texts, "contract", "testing"))
print(estimate_mi_from_cooccurrence(texts, "contract", "banana"))
```

### Wiki Deep Dive

Read `wiki/information-theory.md` thoroughly. Take notes on:
- How entropy relates to concept "sharpness"
- How MI could define concept relationships
- How KL divergence measures learning

---

## Week 3 Milestone

By the end of Week 3, you should have:

1. **Entropy intuition**: Understand it as "expected surprise"
2. **MI computation**: From joint probabilities and co-occurrence
3. **Cluster quality metrics**: Using entropy to evaluate concepts
4. **Research log entry**: Observations about information in your data

### Success Criteria

```python
# You can compute:
H(fair_coin) = 1.0 bit
H(loaded_coin_90_10) ≈ 0.47 bits
MI(independent) ≈ 0
MI(perfectly_correlated) = H(X)

# You understand:
# - Lower cluster entropy = better concept
# - High MI between concepts = related
# - Compression quality = concept quality
```

---

## Research Log Entry

```markdown
## Week 3 Observations

### Entropy Measurements
- Cluster A entropy: [X] bits
- Cluster B entropy: [Y] bits
- Interpretation: [Which is more coherent?]

### Mutual Information
- MI(A, B) = [Z]
- Interpretation: [How related are these concepts?]

### Questions
- What's the "right" entropy for a good concept?
- How does MI compare to cosine similarity?
- Can we use entropy to detect bad clusters?
```

---

## Reflection Questions

1. Why is entropy measured in "bits"?
2. How does mutual information differ from correlation?
3. If concepts are compressions, what are we compressing?

---

→ [Week 4: Clustering](week-04-clustering.md)
