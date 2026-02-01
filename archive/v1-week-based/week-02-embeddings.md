# Week 2: Embeddings

**Goal**: Understand how meaning emerges from co-occurrence.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Build mini Word2Vec, see king - man + woman = queen.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 2.1 | 02a-word2vec-from-scratch | 60 min | Skip-gram intuition |
| 2.2 | 02a continued | 60 min | Implement and train |
| 2.3 | 02b-sentence-transformers | 45 min | Using pretrained models |
| 2.4 | 02c-embedding-visualization | 60 min | UMAP, t-SNE, interactive plots |
| 2.5 | Exercises + exploration | 45 min | Analogies, clustering preview |

---

## Day 2.1-2.2: Word2Vec from Scratch

### Learning Objectives
- [ ] Explain the distributional hypothesis
- [ ] Implement skip-gram negative sampling
- [ ] Train on a small corpus
- [ ] Discover word analogies

### The Core Insight

**Words that appear together become similar.**

```
"The cat sat on the mat"
"The dog sat on the rug"

→ cat ≈ dog (both appear with "sat", "the", "on")
→ mat ≈ rug (both appear after "on the")
```

This is the **distributional hypothesis**: meaning from context.

### Skip-Gram Architecture

```
Input: center word "sat"
      ↓
[Embedding Layer]  →  128-dim vector
      ↓
[Dot product with context word embeddings]
      ↓
Output: probability that "cat" is nearby
```

### Key Code

```python
import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context, negatives):
        # center: [batch]
        # context: [batch]
        # negatives: [batch, n_neg]

        center_emb = self.center_embeddings(center)      # [batch, dim]
        context_emb = self.context_embeddings(context)   # [batch, dim]
        neg_emb = self.context_embeddings(negatives)     # [batch, n_neg, dim]

        # Positive score: center · context
        pos_score = (center_emb * context_emb).sum(dim=1)  # [batch]

        # Negative scores: center · negatives
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()  # [batch, n_neg]

        return pos_score, neg_score
```

### Exercises

1. **Corpus Preparation**: Tokenize a text file, build vocabulary, create training pairs.
2. **Training Loop**: Implement negative sampling loss, train for 10 epochs.
3. **Analogy Test**: After training, test `king - man + woman ≈ ?`

---

## Day 2.3: Sentence Transformers

### Learning Objectives
- [ ] Use pretrained sentence embeddings
- [ ] Compare different models
- [ ] Understand when to use which model

### Why Sentence Embeddings?

Word embeddings are great, but:
- "The dog bit the man" ≈ "The man bit the dog" (same words!)
- Need to capture **order** and **meaning**

Sentence transformers solve this with transformers trained on semantic similarity.

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed sentences
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock prices rose sharply today."
]
embeddings = model.encode(sentences)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)
print(sims)
# sentences[0] and [1] are similar (~0.7)
# sentences[2] is different (~0.1)
```

### Model Comparison

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General |
| all-mpnet-base-v2 | 768 | Medium | Better | When quality matters |
| e5-large-v2 | 1024 | Slow | Best | Retrieval tasks |

### Exercises

1. **Semantic Search**: Embed a set of documents, then find most similar to a query.
2. **Model Comparison**: Compare embeddings from 2-3 models on the same sentences.
3. **Batch Processing**: Embed 1000 sentences efficiently with batching.

---

## Day 2.4: Embedding Visualization

### Learning Objectives
- [ ] Use UMAP for dimensionality reduction
- [ ] Create interactive visualizations with Plotly
- [ ] Identify clusters visually

### UMAP Basics

```python
import umap
import numpy as np

# Reduce from 384D to 2D
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,      # Local neighborhood size
    min_dist=0.1,        # How tight clusters are
    metric='cosine',     # Distance metric
    random_state=42
)

embeddings_2d = reducer.fit_transform(embeddings)
```

### Interactive Visualization

```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'text': sentences,
    'label': labels  # Optional cluster labels
})

fig = px.scatter(
    df, x='x', y='y',
    color='label',
    hover_data=['text'],
    title='Sentence Embedding Space'
)
fig.show()
```

### UMAP vs t-SNE

| Feature | UMAP | t-SNE |
|---------|------|-------|
| Speed | Faster | Slower |
| Global structure | Preserved | Less preserved |
| Reproducibility | Yes (with seed) | Less stable |
| Scalability | Good | Poor for large N |

**Recommendation**: Use UMAP for exploration, t-SNE for publication-quality plots.

### Exercises

1. **Parameter Exploration**: Try different `n_neighbors` and `min_dist`. What changes?
2. **3D Visualization**: Create a 3D scatter plot. Is it more informative?
3. **Cluster Coloring**: Color points by a category (topic, source, etc.).

---

## Day 2.5: Exercises & Exploration

### Analogy Arithmetic

```python
def analogy(word_a, word_b, word_c, embeddings, word2idx, idx2word):
    """
    Solve: word_a is to word_b as word_c is to ???

    Using vector arithmetic: word_b - word_a + word_c ≈ ???
    """
    vec = embeddings[word2idx[word_b]] - embeddings[word2idx[word_a]] + embeddings[word2idx[word_c]]

    # Find nearest neighbor (excluding input words)
    similarities = cosine_similarity([vec], embeddings)[0]
    for word in [word_a, word_b, word_c]:
        similarities[word2idx[word]] = -1

    best_idx = similarities.argmax()
    return idx2word[best_idx]

# Test
print(analogy("king", "man", "woman", ...))  # Should print "queen" (or close)
```

### Clustering Preview

```python
import hdbscan

# Cluster embeddings
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(embeddings)

# Visualize
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    color=labels.astype(str),
    hover_data=[sentences]
)
fig.show()
```

This is a preview of Week 4!

---

## Week 2 Milestone

By the end of Week 2, you should have:

1. **A working Word2Vec** trained on a small corpus
2. **Analogy tests** showing semantic structure
3. **Sentence embeddings** using pretrained models
4. **Interactive visualizations** of embedding space

### Success Criteria

```python
# Your Word2Vec should produce something like:
king - man + woman ≈ queen

# Your visualizations should show:
# - Related sentences clustering together
# - Clear separation between topics
# - Interactive hover showing text
```

---

## Research Log Entry

Start your research log with observations:

```markdown
## Week 2 Observations

### Embedding Quality
- MiniLM produces [X]-dimensional vectors
- Similar sentences have cosine similarity ~[Y]
- Different sentences have similarity ~[Z]

### Interesting Findings
- [What clustered unexpectedly?]
- [What analogies worked/failed?]

### Questions
- How does embedding dimension affect quality?
- What determines which sentences cluster together?
- Could we use these clusters as "concepts"?
```

---

## Reflection Questions

1. Why does "meaning from co-occurrence" work so well?
2. What information is lost in averaging word vectors?
3. How might the structure of embedding space relate to "concepts"?

---

→ [Week 3: Information Theory](week-03-info-theory.md)
