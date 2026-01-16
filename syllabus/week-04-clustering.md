# Week 4: Clustering & Emergent Concepts

**Goal**: See concepts emerge from data.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Cluster article embeddings, name the clusters.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 4.1 | 04a-density-clustering | 60 min | Why density beats k-means |
| 4.2 | 04a continued | 45 min | HDBSCAN deep dive |
| 4.3 | 04b-emergent-concepts | 60 min | Concepts from clusters |
| 4.4 | 04c-concept-quality-metrics | 60 min | Evaluating cluster quality |
| 4.5 | Integration | 45 min | Build concept extraction pipeline |

---

## Day 4.1-4.2: Density-Based Clustering

### Learning Objectives
- [ ] Explain why k-means fails for concepts
- [ ] Use HDBSCAN for density-based clustering
- [ ] Tune HDBSCAN parameters

### Why Not K-Means?

K-means assumes:
- You know k (number of clusters) in advance
- Clusters are spherical and equal-sized
- Every point belongs to a cluster

**For concepts, all three are wrong:**
- We don't know how many concepts exist
- Concepts have varying sizes and shapes
- Some points are noise (not every sentence is a clear concept)

### HDBSCAN: The Right Tool

**Hierarchical Density-Based Spatial Clustering of Applications with Noise**

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,      # Minimum points for a cluster
    min_samples=3,           # Core point threshold
    metric='euclidean',      # Or 'cosine' for embeddings
    cluster_selection_epsilon=0.0
)

labels = clusterer.fit_predict(embeddings)

# labels[i] = -1 means noise (no cluster)
print(f"Found {len(set(labels)) - 1} clusters")
print(f"Noise points: {(labels == -1).sum()}")
```

### Key Parameters

| Parameter | Effect | Guidance |
|-----------|--------|----------|
| `min_cluster_size` | Minimum cluster size | Start with 5-10 |
| `min_samples` | Core point definition | ≤ min_cluster_size |
| `cluster_selection_epsilon` | Merge threshold | 0 for strict |
| `metric` | Distance metric | 'cosine' for embeddings |

### Visualization

```python
import matplotlib.pyplot as plt
import umap

# Reduce to 2D for visualization
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot with cluster colors
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=labels,
    cmap='tab20',
    alpha=0.6,
    s=10
)
plt.colorbar(scatter)
plt.title(f'HDBSCAN Clustering ({len(set(labels))-1} clusters)')
plt.show()
```

### Exercises

1. **Parameter Sweep**: Try min_cluster_size = [3, 5, 10, 20]. How does cluster count change?
2. **Noise Analysis**: What fraction of points are noise? Are they truly noise or small concepts?
3. **Metric Comparison**: Compare 'euclidean' vs 'cosine'. Which gives more intuitive clusters?

---

## Day 4.3: Emergent Concepts

### Learning Objectives
- [ ] Extract representative samples from clusters
- [ ] Auto-generate cluster names
- [ ] Build a concept taxonomy

### From Clusters to Concepts

Each cluster is a **potential concept**. To make it useful:

1. **Find exemplars**: Most representative points
2. **Name it**: Use LLM or keywords
3. **Describe it**: What makes this cluster coherent?

### Finding Exemplars

```python
def get_cluster_exemplars(embeddings, labels, cluster_id, n=5):
    """Get the most central points in a cluster."""
    mask = labels == cluster_id
    cluster_embeddings = embeddings[mask]

    # Compute centroid
    centroid = cluster_embeddings.mean(axis=0)

    # Find closest to centroid
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([centroid], cluster_embeddings)[0]
    top_indices = similarities.argsort()[-n:][::-1]

    # Map back to original indices
    original_indices = np.where(mask)[0][top_indices]
    return original_indices
```

### Auto-Naming with Keywords

```python
from collections import Counter
import re

def extract_keywords(texts, n=5):
    """Extract most common meaningful words."""
    # Simple approach: word frequency
    words = []
    for text in texts:
        words.extend(re.findall(r'\b[a-z]{4,}\b', text.lower()))

    # Remove stopwords
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', ...}
    words = [w for w in words if w not in stopwords]

    return [word for word, count in Counter(words).most_common(n)]

def name_cluster(texts, exemplar_texts):
    """Generate a name for a cluster."""
    keywords = extract_keywords(texts)
    return " / ".join(keywords[:3])
```

### Using an LLM for Naming

```python
def name_cluster_with_llm(exemplar_texts, model="gpt-4"):
    """Use LLM to generate a descriptive name."""
    prompt = f"""Given these example texts from a cluster, suggest a short (2-4 word) concept name:

Examples:
{chr(10).join(f'- {t[:200]}...' for t in exemplar_texts[:5])}

Concept name:"""

    # Call your LLM API here
    response = call_llm(prompt, model=model)
    return response.strip()
```

### Exercises

1. **Manual Naming**: Look at 5 clusters manually. Can you name them? Do they make sense?
2. **Keyword Extraction**: Compare keyword-based names to your intuition.
3. **Hierarchy**: Some clusters seem related. Can you build a two-level taxonomy?

---

## Day 4.4: Concept Quality Metrics

### Learning Objectives
- [ ] Measure cluster coherence
- [ ] Measure cluster separation
- [ ] Combine into a quality score

### Coherence: Internal Consistency

```python
def cluster_coherence(embeddings, labels, cluster_id):
    """
    Measure how similar points within a cluster are.
    Higher = more coherent.
    """
    mask = labels == cluster_id
    cluster_embeddings = embeddings[mask]

    if len(cluster_embeddings) < 2:
        return 0.0

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(cluster_embeddings)

    # Average pairwise similarity (excluding self)
    n = len(cluster_embeddings)
    total = sims.sum() - n  # Subtract diagonal
    return total / (n * (n - 1))
```

### Separation: Distinctness

```python
def cluster_separation(embeddings, labels, cluster_id):
    """
    Measure how different this cluster is from others.
    Higher = more distinct.
    """
    mask = labels == cluster_id
    other_mask = (labels != cluster_id) & (labels != -1)

    cluster_center = embeddings[mask].mean(axis=0)
    other_centers = []

    for other_id in set(labels):
        if other_id != cluster_id and other_id != -1:
            other_centers.append(embeddings[labels == other_id].mean(axis=0))

    if not other_centers:
        return 1.0

    other_centers = np.array(other_centers)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([cluster_center], other_centers)[0]

    return 1 - sims.mean()  # Convert similarity to distance
```

### Combined Quality Score

```python
def concept_quality_score(embeddings, labels, cluster_id):
    """
    Combined quality score for a cluster-as-concept.

    Good concepts: high coherence + high separation
    """
    coherence = cluster_coherence(embeddings, labels, cluster_id)
    separation = cluster_separation(embeddings, labels, cluster_id)

    # Geometric mean (both matter equally)
    return np.sqrt(coherence * separation)

# Evaluate all clusters
quality_scores = {}
for cluster_id in set(labels):
    if cluster_id != -1:
        quality_scores[cluster_id] = concept_quality_score(embeddings, labels, cluster_id)

# Rank by quality
ranked = sorted(quality_scores.items(), key=lambda x: -x[1])
print("Best concepts:")
for cluster_id, score in ranked[:5]:
    print(f"  Cluster {cluster_id}: {score:.3f}")
```

### Exercises

1. **Quality Distribution**: Plot histogram of quality scores. Are most clusters good?
2. **Low Quality Investigation**: Examine the worst clusters. Why are they bad?
3. **Threshold Selection**: What quality threshold gives useful concepts?

---

## Day 4.5: Integration

### Build a Concept Extraction Pipeline

```python
class ConceptExtractor:
    """Extract concepts from a corpus of texts."""

    def __init__(self, embedding_model='all-MiniLM-L6-v2', min_cluster_size=5):
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedding_model)
        self.min_cluster_size = min_cluster_size

    def extract(self, texts):
        """Extract concepts from texts."""
        # 1. Embed
        print("Embedding texts...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # 2. Cluster
        print("Clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)

        # 3. Extract concepts
        concepts = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            mask = labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
            cluster_embeddings = embeddings[mask]

            concept = {
                'id': cluster_id,
                'size': mask.sum(),
                'centroid': cluster_embeddings.mean(axis=0),
                'exemplars': self.get_exemplars(texts, embeddings, labels, cluster_id),
                'keywords': extract_keywords(cluster_texts),
                'quality': concept_quality_score(embeddings, labels, cluster_id)
            }
            concepts.append(concept)

        return concepts, embeddings, labels

    def get_exemplars(self, texts, embeddings, labels, cluster_id, n=5):
        indices = get_cluster_exemplars(embeddings, labels, cluster_id, n)
        return [texts[i] for i in indices]

# Usage
extractor = ConceptExtractor()
concepts, embeddings, labels = extractor.extract(my_texts)

print(f"Extracted {len(concepts)} concepts")
for c in sorted(concepts, key=lambda x: -x['quality'])[:5]:
    print(f"  {c['keywords'][:3]} (n={c['size']}, q={c['quality']:.3f})")
```

---

## Week 4 Milestone

By the end of Week 4, you should have:

1. **Working HDBSCAN pipeline** on your own data
2. **Named concepts** with exemplars and keywords
3. **Quality metrics** to rank concepts
4. **Visualization** of concept space

### Success Criteria

```
- Clusters that make intuitive sense
- Quality scores that match your intuition
- At least one "surprising" concept you didn't expect
- A reusable ConceptExtractor class
```

---

## Research Log Entry

```markdown
## Week 4 Observations

### Clustering Results
- Total texts: [N]
- Clusters found: [M]
- Noise fraction: [X]%

### Best Concepts (by quality)
1. [Name] - [size] points, quality [Q]
2. [Name] - [size] points, quality [Q]
3. ...

### Surprising Findings
- [Unexpected cluster that emerged]
- [Texts that clustered together unexpectedly]

### Quality Metric Insights
- Quality threshold for "good" concept: ~[X]
- Correlation between size and quality: [observation]

### Questions for Later
- How do these clusters relate to each other? (→ Week 5-6: dynamics)
- Can we learn better metrics? (→ Week 8: GDL)
```

---

→ [Week 5: Dynamical Systems](week-05-dynamical.md)
