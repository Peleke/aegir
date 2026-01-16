# Embeddings Primer

## TL;DR

Embeddings map discrete things (words, sentences, concepts) to continuous vectors. **Meaning emerges from co-occurrence**: things that appear together become similar. This is the foundation of concept space.

---

## Core Intuition

How do you represent "cat" to a computer?

**One-hot encoding**: cat = [1, 0, 0, 0, ...], dog = [0, 1, 0, 0, ...]
- Problem: Every word is equally distant from every other
- "cat" and "dog" are as different as "cat" and "refrigerator"

**Embeddings**: cat = [0.2, -0.5, 0.8, ...], dog = [0.3, -0.4, 0.7, ...]
- Meaning is in the geometry
- Similar things have similar vectors
- "cat" and "dog" are close; both are far from "refrigerator"

---

## The Distributional Hypothesis

**"You shall know a word by the company it keeps."** — J.R. Firth, 1957

Words that appear in similar contexts have similar meanings:
- "dog" and "cat" both appear near "pet", "fur", "vet"
- Therefore dog ≈ cat (in embedding space)

This is remarkably powerful. No one told the model what "dog" means — it learned from co-occurrence.

---

## Word2Vec

The breakthrough algorithm (Mikolov et al., 2013).

### Skip-gram: Predict Context from Word

Given a center word, predict surrounding words:

```
Sentence: "The cat sat on the mat"
Center word: "sat"
Context: ["The", "cat", "on", "the"]

Task: Given "sat", predict that "cat" is nearby
```

### Training

1. Initialize random vectors for each word
2. For each word in corpus:
   - Get center word vector w_c
   - Get context word vectors w_o
   - Push w_c toward w_o (if they co-occur)
   - Push w_c away from random negatives
3. After training: vectors encode co-occurrence patterns

### The Magic: Analogies

Famous result: king - man + woman ≈ queen

```python
import numpy as np

# Hypothetical embeddings
king = np.array([0.8, 0.2, 0.9])
man = np.array([0.7, 0.1, 0.3])
woman = np.array([0.6, 0.9, 0.3])
queen = np.array([0.7, 1.0, 0.9])

result = king - man + woman
print(f"king - man + woman = {result}")
print(f"queen = {queen}")
print(f"Cosine similarity: {np.dot(result, queen) / (np.linalg.norm(result) * np.linalg.norm(queen)):.3f}")
```

**Why this works**: "king" and "queen" differ by a "gender direction." So does "man" and "woman." Vector arithmetic reveals semantic structure.

---

## Word2Vec Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    """Simple Skip-gram Word2Vec."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words, negative_words):
        """
        center_words: [batch_size]
        context_words: [batch_size]
        negative_words: [batch_size, num_negatives]
        """
        # Get embeddings
        center = self.center_embeddings(center_words)  # [batch, dim]
        context = self.context_embeddings(context_words)  # [batch, dim]
        negatives = self.context_embeddings(negative_words)  # [batch, neg, dim]

        # Positive score: center · context
        pos_score = (center * context).sum(dim=1)  # [batch]

        # Negative scores: center · negatives
        neg_score = torch.bmm(negatives, center.unsqueeze(2)).squeeze()  # [batch, neg]

        # Loss: maximize positive, minimize negative
        pos_loss = F.logsigmoid(pos_score).mean()
        neg_loss = F.logsigmoid(-neg_score).mean()

        return -(pos_loss + neg_loss)
```

---

## Sentence Transformers

Word embeddings are great, but how do you embed a sentence?

### Naive: Average Word Vectors

```python
def naive_sentence_embedding(sentence, word_vectors):
    words = sentence.lower().split()
    vectors = [word_vectors[w] for w in words if w in word_vectors]
    return np.mean(vectors, axis=0)
```

Problem: "The dog bit the man" ≈ "The man bit the dog" (same words, same average).

### Better: Sentence Transformers

Transformers that are trained to produce good sentence embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Stock prices rose today"
]

embeddings = model.encode(sentences)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)

print("Similarity matrix:")
print(sims)
# sentences[0] and [1] will be similar (both about cat on mat)
# sentence[2] will be different (about stocks)
```

### How They're Trained

1. **Contrastive learning**: Similar sentences should have similar embeddings
2. **Training data**: Pairs like (question, answer), (premise, entailment)
3. **Architecture**: BERT-based, with pooling over tokens

---

## Visualizing Embeddings

High-dimensional embeddings are hard to see. Dimensionality reduction helps.

### UMAP (Uniform Manifold Approximation and Projection)

```python
import umap
import matplotlib.pyplot as plt

# Assume embeddings is [num_samples, embedding_dim]
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5)

# Add labels
for i, label in enumerate(labels):
    plt.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]))

plt.title("Embedding Space (UMAP projection)")
plt.show()
```

### t-SNE vs UMAP

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| t-SNE | Good local structure | Slow, distances not meaningful |
| UMAP | Faster, preserves global structure | Parameter-sensitive |

For exploration, UMAP is usually the better choice.

---

## Embedding Quality

How do you know if embeddings are good?

### Intrinsic Evaluation

- **Analogy tasks**: king - man + woman = queen?
- **Similarity correlation**: Does embedding similarity match human similarity ratings?
- **Clustering**: Do similar things cluster together?

### Extrinsic Evaluation

- **Downstream tasks**: Do embeddings improve classification, retrieval, etc.?
- **Transfer**: Do embeddings trained on one task help with another?

---

## Embedding Spaces for Concepts

For Aegir, we'll embed:
- **Articles/documents**: What is this about?
- **Chunks**: What does this passage say?
- **Queries**: What is the user asking?
- **Concepts**: What abstract idea does this represent?

The key insight: **similar embeddings = related concepts**.

```
Document: "Contract testing verifies API agreements"
          ↓ embed
Embedding: [0.2, -0.1, 0.8, ...]
          ↓ cluster
Concept: "Contract Testing" (emergent!)
```

---

## Relevance for Aegir

| Concept | Application |
|---------|-------------|
| Word embeddings | Basic building block |
| Sentence embeddings | Embed articles, chunks, queries |
| Co-occurrence → similarity | Meaning emerges from data |
| Analogy structure | Relationships encoded in geometry |
| Clustering | Concepts emerge from embedding clusters |

**Key insight**: We don't predefine concepts. We embed content, cluster embeddings, and concepts emerge. The embedding space is the foundation of concept space.

---

## Key Formulas

```
Cosine similarity:    sim(a, b) = (a · b) / (||a|| ||b||)
Word2Vec objective:   max Σ log σ(w_c · w_o) + Σ log σ(-w_c · w_neg)
Analogy:              a - b + c ≈ d  if  (a:b :: c:d)
```

---

## Going Deeper

- **Paper**: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- **Paper**: Reimers & Gurevych, "Sentence-BERT" (2019)
- **Tutorial**: [Jay Alammar's Word2Vec Visual Guide](https://jalammar.github.io/illustrated-word2vec/)
- **Library**: [sentence-transformers](https://www.sbert.net/)
