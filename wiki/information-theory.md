# Information Theory Primer

## TL;DR

Information theory quantifies **surprise** and **structure**. Entropy measures uncertainty, mutual information measures shared structure, and good concepts are good compressions.

---

## Core Intuition

Information theory answers: **How much "surprise" is in a message?**

If I tell you the sun rose this morning, that's not surprising (low information). If I tell you it snowed in July in Miami, that's very surprising (high information).

---

## Entropy: Expected Surprise

For a random variable X with possible values {x₁, x₂, ...} and probabilities {p₁, p₂, ...}:

```
H(X) = -∑ᵢ pᵢ log₂(pᵢ)
```

**Intuition**:
- Fair coin flip (50/50): H = 1 bit (maximum uncertainty for 2 outcomes)
- Loaded coin (99/1): H ≈ 0.08 bits (almost certain, low surprise)
- Constant value: H = 0 bits (no uncertainty at all)

**For concept spaces**: A "sharp" concept cluster has low entropy (you know what's in it). A diffuse region has high entropy (could be anything).

### Python Example

```python
import numpy as np

def entropy(probs):
    """Compute entropy in bits."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

# Fair coin
print(entropy([0.5, 0.5]))  # 1.0 bit

# Loaded coin
print(entropy([0.99, 0.01]))  # ~0.08 bits

# Certain outcome
print(entropy([1.0]))  # 0.0 bits
```

---

## Mutual Information: Shared Structure

How much does knowing X tell you about Y?

```
I(X; Y) = H(X) + H(Y) - H(X,Y)
        = H(X) - H(X|Y)
```

**Intuition**:
- If X and Y are independent: I(X;Y) = 0 (knowing X tells you nothing about Y)
- If X determines Y completely: I(X;Y) = H(Y) (knowing X tells you everything about Y)

**For concept spaces**: Two concept clusters that co-activate have high mutual information. This could define "relatedness" without explicit edges.

### Python Example

```python
def mutual_information(joint_probs):
    """
    Compute MI from joint probability table.
    joint_probs[i,j] = P(X=i, Y=j)
    """
    px = joint_probs.sum(axis=1)  # Marginal of X
    py = joint_probs.sum(axis=0)  # Marginal of Y

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(joint_probs.flatten())

    return hx + hy - hxy

# Independent variables
independent = np.array([[0.25, 0.25], [0.25, 0.25]])
print(mutual_information(independent))  # ~0.0

# Perfectly correlated
correlated = np.array([[0.5, 0.0], [0.0, 0.5]])
print(mutual_information(correlated))  # 1.0 bit
```

---

## KL Divergence: Distance Between Distributions

How different are two probability distributions P and Q?

```
D_KL(P || Q) = ∑ᵢ P(xᵢ) log(P(xᵢ)/Q(xᵢ))
```

**Intuition**: The "cost" of using Q to encode data that actually comes from P.

**Warning**: Not symmetric! D_KL(P||Q) ≠ D_KL(Q||P)

**For concept spaces**: How different is the concept space now vs. before this experience? This could measure learning.

---

## Compression = Understanding

**Key insight**: A good concept is a good compression.

If you can describe many experiences with one concept, you've compressed. Shannon's source coding theorem says you can't compress below entropy — so concepts carve out low-entropy regions.

```
Good concept: "contract testing" compresses {article, code, errors, outcomes}
    → Many specific things, one label

Bad concept: "stuff" compresses nothing (too broad, high entropy)
    → Everything is "stuff", so the label tells you nothing

Bad concept: "this specific test run at 3pm" compresses nothing (too narrow)
    → Only one thing, not useful for generalization
```

**The sweet spot**: Concepts that capture regularities — patterns that repeat.

---

## Relevance for Aegir

| Concept | How We'll Use It |
|---------|------------------|
| Entropy | Measure cluster "sharpness" (is this a coherent concept?) |
| Mutual Information | Measure concept relatedness without explicit edges |
| KL Divergence | Measure how much new experience changes the space |
| Compression | Evaluate concept quality (good concepts = good compression) |

---

## Going Deeper

- **Book**: Cover & Thomas, "Elements of Information Theory" (Ch 1-3)
- **Visual**: [Colah's Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
- **Video**: 3Blue1Brown doesn't have one yet, but Khan Academy covers basics

---

## Key Formulas

```
Entropy:           H(X) = -∑ p(x) log p(x)
Conditional:       H(X|Y) = -∑ p(x,y) log p(x|y)
Mutual Info:       I(X;Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y)
KL Divergence:     D_KL(P||Q) = ∑ p(x) log(p(x)/q(x))
```
