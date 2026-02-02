# Week 9: Integration

**Goal**: Put it all together. Build concept space prototype.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Working concept space that embeds, clusters, and reasons.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 9.1 | 09a-concept-space-prototype | 75 min | Core architecture |
| 9.2 | 09b-salience-field | 60 min | Attention/relevance field |
| 9.3 | 09c-reasoning-dynamics | 60 min | Movement through concept space |
| 9.4 | Testing & refinement | 45 min | Debug and improve |
| 9.5 | Documentation | 30 min | Write up design |

---

## Day 9.1: Concept Space Prototype

### Learning Objectives
- [ ] Design the core data structures
- [ ] Implement the main ConceptSpace class
- [ ] Integrate embeddings, clustering, and graph

### Architecture Overview

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Concept:
    """A single concept in the space."""
    id: int
    name: str
    centroid: np.ndarray
    exemplars: List[str]
    size: int
    quality: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class ConceptSpaceConfig:
    """Configuration for concept space."""
    embedding_model: str = 'all-MiniLM-L6-v2'
    min_cluster_size: int = 5
    similarity_threshold: float = 0.5
    salience_decay: float = 0.1
```

### The ConceptSpace Class

```python
from sentence_transformers import SentenceTransformer
import hdbscan
import networkx as nx
import torch
from torch_geometric.data import Data

class ConceptSpace:
    """
    Physics-inspired concept space.

    Combines:
    - Embeddings (Week 2)
    - Clustering (Week 4)
    - Graph structure (Week 8)
    - Dynamics (Week 5-6)
    """

    def __init__(self, config: ConceptSpaceConfig = None):
        self.config = config or ConceptSpaceConfig()
        self.embedder = SentenceTransformer(self.config.embedding_model)

        # Core state
        self.texts: List[str] = []
        self.embeddings: np.ndarray = None
        self.concepts: List[Concept] = []
        self.graph: nx.Graph = None
        self.pyg_data: Data = None

        # Salience field
        self.salience: np.ndarray = None

    def add_texts(self, texts: List[str]):
        """Add texts to the concept space."""
        new_embeddings = self.embedder.encode(texts, show_progress_bar=True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.texts = texts
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.texts.extend(texts)

        # Recompute structure
        self._update_structure()

    def _update_structure(self):
        """Recompute concepts, graph, and salience."""
        self._cluster_concepts()
        self._build_graph()
        self._initialize_salience()

    def _cluster_concepts(self):
        """Extract concepts via clustering."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size
        )
        labels = clusterer.fit_predict(self.embeddings)

        self.concepts = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            mask = labels == cluster_id
            cluster_embeddings = self.embeddings[mask]
            cluster_texts = [self.texts[i] for i in range(len(self.texts)) if mask[i]]

            concept = Concept(
                id=cluster_id,
                name=self._auto_name(cluster_texts),
                centroid=cluster_embeddings.mean(axis=0),
                exemplars=cluster_texts[:5],
                size=mask.sum(),
                quality=self._compute_quality(cluster_embeddings)
            )
            self.concepts.append(concept)

    def _auto_name(self, texts: List[str]) -> str:
        """Auto-generate concept name from texts."""
        from collections import Counter
        import re

        words = []
        for text in texts[:20]:
            words.extend(re.findall(r'\b[a-z]{4,}\b', text.lower()))

        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'there', 'their', 'which', 'would', 'could'}
        words = [w for w in words if w not in stopwords]

        common = Counter(words).most_common(3)
        return '/'.join([w for w, _ in common])

    def _compute_quality(self, cluster_embeddings: np.ndarray) -> float:
        """Compute concept quality score."""
        from sklearn.metrics.pairwise import cosine_similarity

        if len(cluster_embeddings) < 2:
            return 0.0

        sims = cosine_similarity(cluster_embeddings)
        n = len(cluster_embeddings)
        coherence = (sims.sum() - n) / (n * (n - 1))
        return coherence

    def _build_graph(self):
        """Build concept graph."""
        from sklearn.metrics.pairwise import cosine_similarity

        self.graph = nx.Graph()

        # Add nodes
        for concept in self.concepts:
            self.graph.add_node(concept.id, concept=concept)

        # Add edges based on similarity
        centroids = np.array([c.centroid for c in self.concepts])
        if len(centroids) > 1:
            sims = cosine_similarity(centroids)

            for i, c1 in enumerate(self.concepts):
                for j, c2 in enumerate(self.concepts):
                    if i < j and sims[i, j] > self.config.similarity_threshold:
                        self.graph.add_edge(c1.id, c2.id, weight=sims[i, j])

        # Convert to PyG
        self._build_pyg_data()

    def _build_pyg_data(self):
        """Convert graph to PyTorch Geometric format."""
        if len(self.concepts) == 0:
            return

        # Node features: concept centroids
        x = torch.tensor([c.centroid for c in self.concepts], dtype=torch.float)

        # Edge index
        edges = list(self.graph.edges())
        if edges:
            # Create bidirectional edges
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        self.pyg_data = Data(x=x, edge_index=edge_index)

    def _initialize_salience(self):
        """Initialize salience field (uniform)."""
        self.salience = np.ones(len(self.concepts)) / len(self.concepts)
```

### Exercises

1. **Test the Basics**: Create a ConceptSpace, add some texts, check concepts.
2. **Persistence**: Add save/load methods for the concept space.
3. **Incremental Updates**: Handle adding new texts without full recomputation.

---

## Day 9.2: Salience Field

### Learning Objectives
- [ ] Implement salience as a dynamic field
- [ ] Update salience based on queries
- [ ] Visualize salience on concept space

### Salience as Energy

```python
class ConceptSpace:
    # ... (previous code)

    def update_salience(self, query: str, decay: float = None):
        """
        Update salience based on a query.

        Salience decays from its current state and
        gets boosted for relevant concepts.
        """
        if decay is None:
            decay = self.config.salience_decay

        # Decay current salience
        self.salience *= (1 - decay)

        # Compute relevance to query
        query_embedding = self.embedder.encode([query])[0]
        relevance = self._compute_relevance(query_embedding)

        # Boost salience for relevant concepts
        self.salience += relevance

        # Normalize
        self.salience = self.salience / (self.salience.sum() + 1e-10)

        return self.salience

    def _compute_relevance(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute relevance of each concept to query."""
        centroids = np.array([c.centroid for c in self.concepts])
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity([query_embedding], centroids)[0]

        # Softmax for normalization
        exp_sims = np.exp(sims - sims.max())
        return exp_sims / exp_sims.sum()

    def propagate_salience(self, steps: int = 3, diffusion: float = 0.3):
        """
        Propagate salience through concept graph.

        Neighbors share salience (diffusion process).
        """
        for _ in range(steps):
            new_salience = np.zeros_like(self.salience)

            for node in self.graph.nodes():
                # Keep some of own salience
                new_salience[node] = (1 - diffusion) * self.salience[node]

                # Get salience from neighbors
                neighbors = list(self.graph.neighbors(node))
                if neighbors:
                    neighbor_salience = self.salience[neighbors].sum() / len(neighbors)
                    new_salience[node] += diffusion * neighbor_salience

            self.salience = new_salience / (new_salience.sum() + 1e-10)

        return self.salience

    def get_salient_concepts(self, top_k: int = 5) -> List[Concept]:
        """Get most salient concepts."""
        indices = self.salience.argsort()[-top_k:][::-1]
        return [self.concepts[i] for i in indices]
```

### Visualization

```python
def visualize_salience(space: ConceptSpace, query: str = None):
    """Visualize concept space with salience coloring."""
    import umap
    import matplotlib.pyplot as plt

    if query:
        space.update_salience(query)
        space.propagate_salience()

    # Project concept centroids to 2D
    centroids = np.array([c.centroid for c in space.concepts])
    reducer = umap.UMAP(n_components=2, random_state=42)
    centroids_2d = reducer.fit_transform(centroids)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Concepts as points, colored by salience
    scatter = ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c=space.salience,
        cmap='YlOrRd',
        s=[c.size * 10 for c in space.concepts],
        alpha=0.7
    )

    # Labels for top concepts
    top_indices = space.salience.argsort()[-5:][::-1]
    for idx in top_indices:
        ax.annotate(
            space.concepts[idx].name,
            (centroids_2d[idx, 0], centroids_2d[idx, 1]),
            fontsize=8
        )

    # Edges
    for edge in space.graph.edges():
        i, j = edge
        ax.plot(
            [centroids_2d[i, 0], centroids_2d[j, 0]],
            [centroids_2d[i, 1], centroids_2d[j, 1]],
            'gray', alpha=0.2, linewidth=0.5
        )

    plt.colorbar(scatter, label='Salience')
    plt.title(f'Concept Space' + (f' - Query: "{query}"' if query else ''))
    plt.show()

# Usage
space = ConceptSpace()
space.add_texts(my_texts)
visualize_salience(space, query="What is contract testing?")
```

---

## Day 9.3: Reasoning Dynamics

### Learning Objectives
- [ ] Implement reasoning as movement through concept space
- [ ] Trace reasoning paths
- [ ] Connect to variational framework

### Reasoning Trajectory

```python
class ConceptSpace:
    # ... (previous code)

    def reason(self, start_query: str, goal_query: str, max_steps: int = 10):
        """
        Reason from start to goal concept.

        Returns a trajectory through concept space.
        """
        # Find start and goal concepts
        start_embedding = self.embedder.encode([start_query])[0]
        goal_embedding = self.embedder.encode([goal_query])[0]

        start_concept = self._nearest_concept(start_embedding)
        goal_concept = self._nearest_concept(goal_embedding)

        print(f"Start: {start_concept.name}")
        print(f"Goal: {goal_concept.name}")

        # Find path through graph
        try:
            path = nx.shortest_path(self.graph, start_concept.id, goal_concept.id)
            trajectory = [self.concepts[i] for i in path]
            print(f"Path: {' -> '.join([c.name for c in trajectory])}")
            return trajectory
        except nx.NetworkXNoPath:
            print("No path found in graph. Using embedding interpolation.")
            return self._interpolate_path(start_concept, goal_concept, max_steps)

    def _nearest_concept(self, embedding: np.ndarray) -> Concept:
        """Find nearest concept to an embedding."""
        from sklearn.metrics.pairwise import cosine_similarity

        centroids = np.array([c.centroid for c in self.concepts])
        sims = cosine_similarity([embedding], centroids)[0]
        return self.concepts[sims.argmax()]

    def _interpolate_path(self, start: Concept, goal: Concept, steps: int) -> List[Concept]:
        """Linear interpolation in embedding space."""
        trajectory = [start]

        for i in range(1, steps):
            t = i / steps
            interp = (1 - t) * start.centroid + t * goal.centroid
            nearest = self._nearest_concept(interp)
            if nearest.id != trajectory[-1].id:
                trajectory.append(nearest)

        if trajectory[-1].id != goal.id:
            trajectory.append(goal)

        return trajectory
```

### Variational Reasoning

```python
def variational_reasoning_path(self, start_query: str, goal_query: str,
                                effort_weight: float = 1.0):
    """
    Find reasoning path that minimizes action.

    Action = ∫(effort - salience) dt
    """
    start_emb = self.embedder.encode([start_query])[0]
    goal_emb = self.embedder.encode([goal_query])[0]

    # Define action
    def compute_action(path_embeddings):
        """Compute action for a path."""
        action = 0
        for i in range(len(path_embeddings) - 1):
            # Effort: distance traveled
            effort = np.linalg.norm(path_embeddings[i+1] - path_embeddings[i])

            # Salience: relevance at current point
            nearest = self._nearest_concept(path_embeddings[i])
            salience = self.salience[nearest.id]

            action += effort_weight * effort - salience

        return action

    # Simple optimization: gradient descent on waypoints
    # (Full variational would use BVP solver)

    num_waypoints = 5
    waypoints = np.linspace(start_emb, goal_emb, num_waypoints + 2)[1:-1]

    learning_rate = 0.1
    for iteration in range(50):
        # Compute gradient numerically
        gradients = []
        for i, wp in enumerate(waypoints):
            grad = np.zeros_like(wp)
            h = 0.01
            for d in range(len(wp)):
                wp_plus = wp.copy()
                wp_minus = wp.copy()
                wp_plus[d] += h
                wp_minus[d] -= h

                path_plus = np.vstack([start_emb, *waypoints[:i], wp_plus, *waypoints[i+1:], goal_emb])
                path_minus = np.vstack([start_emb, *waypoints[:i], wp_minus, *waypoints[i+1:], goal_emb])

                grad[d] = (compute_action(path_plus) - compute_action(path_minus)) / (2 * h)

            gradients.append(grad)

        # Update waypoints
        for i in range(len(waypoints)):
            waypoints[i] -= learning_rate * gradients[i]

    # Convert to concepts
    full_path = np.vstack([start_emb, *waypoints, goal_emb])
    trajectory = [self._nearest_concept(emb) for emb in full_path]

    return trajectory
```

### Visualize Reasoning

```python
def visualize_reasoning(space: ConceptSpace, trajectory: List[Concept]):
    """Visualize a reasoning trajectory."""
    import umap
    import matplotlib.pyplot as plt

    centroids = np.array([c.centroid for c in space.concepts])
    reducer = umap.UMAP(n_components=2, random_state=42)
    centroids_2d = reducer.fit_transform(centroids)

    fig, ax = plt.subplots(figsize=(12, 10))

    # All concepts
    ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c='lightgray', s=50, alpha=0.5
    )

    # Trajectory
    traj_ids = [c.id for c in trajectory]
    traj_2d = centroids_2d[traj_ids]

    ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'b-', linewidth=2, label='Reasoning path')
    ax.scatter(traj_2d[:, 0], traj_2d[:, 1], c='blue', s=100, zorder=5)

    # Labels
    for i, concept in enumerate(trajectory):
        ax.annotate(
            f"{i}: {concept.name}",
            (traj_2d[i, 0], traj_2d[i, 1]),
            fontsize=8
        )

    ax.scatter([traj_2d[0, 0]], [traj_2d[0, 1]], c='green', s=200, marker='o', label='Start')
    ax.scatter([traj_2d[-1, 0]], [traj_2d[-1, 1]], c='red', s=200, marker='*', label='Goal')

    plt.legend()
    plt.title('Reasoning Trajectory')
    plt.show()
```

---

## Day 9.4-9.5: Testing & Documentation

### Integration Test

```python
def test_full_pipeline():
    """Test the complete concept space pipeline."""
    # Sample texts (you'd use your real data)
    texts = [
        "Contract testing verifies API agreements between services",
        "Unit tests check individual function behavior",
        "Integration tests verify component interactions",
        "Docker containers package applications with dependencies",
        "Kubernetes orchestrates container deployments",
        "CI/CD pipelines automate software delivery",
        # ... more texts
    ]

    # Create concept space
    space = ConceptSpace()
    space.add_texts(texts)

    print(f"Created {len(space.concepts)} concepts")
    for c in space.concepts:
        print(f"  - {c.name}: {c.size} items, quality={c.quality:.2f}")

    # Test salience
    space.update_salience("How do I test microservices?")
    space.propagate_salience()

    print("\nTop salient concepts:")
    for c in space.get_salient_concepts(3):
        print(f"  - {c.name}: salience={space.salience[c.id]:.3f}")

    # Test reasoning
    trajectory = space.reason(
        "I want to containerize my application",
        "How do I deploy to production?"
    )

    print("\nReasoning trajectory:")
    for i, c in enumerate(trajectory):
        print(f"  {i}: {c.name}")

    return space

space = test_full_pipeline()
```

### Write Design Document

Create a design document covering:

1. **Architecture**: ConceptSpace class structure
2. **Data Flow**: Texts → Embeddings → Concepts → Graph → Salience
3. **Key Algorithms**: Clustering, graph construction, salience propagation
4. **API**: Main methods and their parameters
5. **Future Work**: What's not implemented yet

---

## Week 9 Milestone

By the end of Week 9, you should have:

1. **ConceptSpace class**: Complete implementation
2. **Salience field**: Working update and propagation
3. **Reasoning**: Path finding through concept space
4. **Visualization**: See your concept space

### Success Criteria

```python
# Your ConceptSpace can:
# 1. Ingest texts and extract concepts
# 2. Build a concept graph
# 3. Update salience for queries
# 4. Find reasoning paths
# 5. Visualize the whole thing

# This is your working prototype!
```

---

→ [Week 10: Experiments](week-10-experiments.md)
