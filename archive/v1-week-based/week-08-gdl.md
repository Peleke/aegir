# Week 8: Geometric Deep Learning

**Goal**: Learn on structured data. Build a simple GNN.

**Time**: 5 days × 90 min = 7.5 hours

**Milestone**: Working GNN on concept graph.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 8.1 | 08a-graph-basics | 60 min | Graphs as data structures |
| 8.2 | 08b-message-passing | 60 min | The core GNN operation |
| 8.3 | 08b continued | 45 min | GCN, GAT architectures |
| 8.4 | 08c-equivariance | 60 min | Symmetry and structure |
| 8.5 | Application | 45 min | GNN on concept space |

---

## Day 8.1: Graph Basics

### Learning Objectives
- [ ] Represent graphs in code
- [ ] Understand adjacency and edge lists
- [ ] Visualize graphs with NetworkX

### Graphs as Data

A **graph** G = (V, E) has:
- **Vertices (nodes)** V: entities
- **Edges** E: relationships

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()

# Add nodes with features
G.add_node("cat", embedding=[0.2, 0.5])
G.add_node("dog", embedding=[0.3, 0.4])
G.add_node("pet", embedding=[0.25, 0.45])
G.add_node("animal", embedding=[0.1, 0.3])

# Add edges (relationships)
G.add_edge("cat", "pet")
G.add_edge("dog", "pet")
G.add_edge("cat", "animal")
G.add_edge("dog", "animal")
G.add_edge("pet", "animal")

# Visualize
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=12, font_weight='bold')
plt.title("Simple Concept Graph")
plt.show()
```

### Representations

```python
# Adjacency matrix
A = nx.adjacency_matrix(G).todense()
print("Adjacency matrix:")
print(A)

# Edge list
edge_list = list(G.edges())
print("\nEdge list:", edge_list)

# For PyTorch Geometric
import torch
edge_index = torch.tensor([
    [0, 0, 1, 1, 2],  # Source nodes
    [2, 3, 2, 3, 3]   # Target nodes
], dtype=torch.long)

# Node features
x = torch.tensor([
    [0.2, 0.5],  # cat
    [0.3, 0.4],  # dog
    [0.25, 0.45],  # pet
    [0.1, 0.3]   # animal
], dtype=torch.float)
```

### Concept Graphs from Clustering

```python
def build_concept_graph(concepts, embeddings, similarity_threshold=0.5):
    """
    Build a graph where concepts are nodes and
    edges connect similar concepts.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    G = nx.Graph()

    # Add nodes
    for i, concept in enumerate(concepts):
        G.add_node(i,
            name=concept.get('name', f'Concept_{i}'),
            embedding=concept['centroid'],
            size=concept['size']
        )

    # Add edges based on similarity
    centroids = np.array([c['centroid'] for c in concepts])
    sims = cosine_similarity(centroids)

    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            if sims[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=sims[i, j])

    return G

# Build and visualize
concept_graph = build_concept_graph(concepts, embeddings)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(concept_graph, k=2, iterations=50)

# Node sizes based on concept size
sizes = [concept_graph.nodes[n]['size'] * 50 for n in concept_graph.nodes()]

nx.draw(concept_graph, pos,
        node_size=sizes,
        node_color='lightgreen',
        alpha=0.7,
        with_labels=True,
        font_size=8)
plt.title("Concept Graph")
plt.show()
```

### Exercises

1. **Graph Statistics**: Compute degree distribution, clustering coefficient, diameter.
2. **Different Thresholds**: How does the graph change with different similarity thresholds?
3. **Directed Graphs**: Create a directed concept graph where edges point toward more general concepts.

---

## Day 8.2-8.3: Message Passing

### Learning Objectives
- [ ] Understand message passing as information flow
- [ ] Implement a basic GNN layer
- [ ] Use PyTorch Geometric

### The Core Idea

**Message passing** = nodes aggregate information from neighbors.

```
For each node i:
    1. Collect messages from neighbors j ∈ N(i)
    2. Aggregate (sum, mean, max)
    3. Update node representation
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMessagePassing(nn.Module):
    """Basic message passing layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, in_dim]
        edge_index: [2, num_edges]
        """
        src, dst = edge_index
        num_nodes = x.size(0)

        # Gather source node features
        messages = x[src]  # [num_edges, in_dim]

        # Aggregate by destination (mean)
        aggregated = torch.zeros(num_nodes, x.size(1))
        count = torch.zeros(num_nodes)

        for i, (s, d) in enumerate(zip(src, dst)):
            aggregated[d] += messages[i]
            count[d] += 1

        count[count == 0] = 1
        aggregated = aggregated / count.unsqueeze(1)

        # Update: concatenate self and aggregated, then transform
        combined = torch.cat([x, aggregated], dim=1)
        return F.relu(self.linear(combined))
```

### GCN (Graph Convolutional Network)

```python
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Create PyG data object
data = Data(x=x, edge_index=edge_index)

# GCN layer
conv = GCNConv(2, 8)  # 2 input features → 8 output features

# Forward pass
out = conv(data.x, data.edge_index)
print(f"Input shape: {data.x.shape}")
print(f"Output shape: {out.shape}")
```

### GAT (Graph Attention Network)

```python
from torch_geometric.nn import GATConv

# GAT learns which neighbors to attend to
gat = GATConv(2, 8, heads=4, concat=True)
out = gat(data.x, data.edge_index)
print(f"GAT output shape: {out.shape}")  # [num_nodes, 8*4]
```

### Multi-Layer GNN

```python
class ConceptGNN(nn.Module):
    """Multi-layer GNN for concept processing."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

# Create and test
model = ConceptGNN(2, 16, 8)
out = model(data.x, data.edge_index)
print(f"Final output shape: {out.shape}")
```

### Exercises

1. **Message Visualization**: Print what messages each node receives.
2. **Layer Depth**: Compare 2, 4, 8 layers. What's the receptive field?
3. **Aggregation Functions**: Compare sum vs mean vs max aggregation.

---

## Day 8.4: Equivariance

### Learning Objectives
- [ ] Understand permutation equivariance
- [ ] See why GNNs respect graph structure
- [ ] Connect to group theory (brief)

### Permutation Equivariance

GNNs are **equivariant** to node permutation:

```
If you reorder the nodes, the output reorders the same way.
```

```python
def test_permutation_equivariance(model, data):
    """Verify GNN is permutation equivariant."""
    # Original output
    out1 = model(data.x, data.edge_index)

    # Permute nodes
    perm = torch.randperm(data.x.size(0))
    x_perm = data.x[perm]

    # Permute edge_index accordingly
    inv_perm = torch.argsort(perm)
    edge_index_perm = inv_perm[data.edge_index]

    # Output on permuted graph
    out2 = model(x_perm, edge_index_perm)

    # Unpermute output
    out2_unperm = out2[inv_perm]

    # Should be equal (up to numerical precision)
    diff = (out1 - out2_unperm).abs().max().item()
    print(f"Max difference: {diff:.6f}")
    assert diff < 1e-5, "Not permutation equivariant!"

test_permutation_equivariance(model, data)
```

### Why Equivariance Matters

- **Generalization**: Network doesn't memorize node order
- **Efficiency**: Can share parameters across nodes
- **Correctness**: Node order is arbitrary; shouldn't affect result

### The GDL Blueprint

From Bronstein et al.'s framework:

| Domain | Symmetry Group | Network Type |
|--------|----------------|--------------|
| Sets | Permutation Sₙ | DeepSets, Transformers |
| Grids | Translation ℤⁿ | CNNs |
| Graphs | Node permutation Sₙ | GNNs |
| Manifolds | Diffeomorphisms | Gauge CNNs |

### Exercises

1. **Break Equivariance**: Modify the GNN to depend on node order. What happens?
2. **Other Symmetries**: What symmetry does a CNN respect?
3. **GDL Paper**: Read Section 3 of the Geometric Deep Learning paper.

---

## Day 8.5: GNN on Concept Space

### Learning Objectives
- [ ] Apply GNN to concept graph
- [ ] Learn concept representations
- [ ] Use for downstream tasks

### Task: Concept Similarity Prediction

```python
from torch_geometric.data import Data, DataLoader
import torch.optim as optim

class ConceptSimilarityModel(nn.Module):
    """Predict whether two concepts are similar."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gnn = ConceptGNN(in_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, pairs):
        """
        pairs: [num_pairs, 2] - indices of node pairs to compare
        """
        # Get node embeddings from GNN
        node_emb = self.gnn(x, edge_index)

        # Concatenate pairs
        pair_emb = torch.cat([
            node_emb[pairs[:, 0]],
            node_emb[pairs[:, 1]]
        ], dim=1)

        return self.classifier(pair_emb)

# Training loop
def train_concept_model(model, data, pairs, labels, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(data.x, data.edge_index, pairs).squeeze()
        loss = criterion(pred, labels.float())

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
```

### Task: Concept Propagation

```python
def propagate_relevance(gnn, data, query_embedding, num_steps=3):
    """
    Given a query, propagate relevance through concept graph.

    Returns: relevance score for each concept
    """
    # Initialize relevance based on similarity to query
    sims = F.cosine_similarity(data.x, query_embedding.unsqueeze(0))
    relevance = F.softmax(sims, dim=0)

    # Propagate through graph
    for _ in range(num_steps):
        # Message passing: neighbors share relevance
        src, dst = data.edge_index
        messages = relevance[src]

        # Aggregate
        new_relevance = torch.zeros_like(relevance)
        new_relevance.index_add_(0, dst, messages)

        # Normalize
        degree = torch.zeros_like(relevance)
        degree.index_add_(0, dst, torch.ones(src.size(0)))
        degree[degree == 0] = 1
        new_relevance = new_relevance / degree

        # Mix with original (residual)
        relevance = 0.5 * relevance + 0.5 * new_relevance

    return relevance

# Example usage
query = torch.randn(data.x.size(1))  # Random query embedding
relevance = propagate_relevance(model.gnn, data, query)
top_concepts = relevance.topk(5)
print("Top relevant concepts:", top_concepts)
```

### Putting It Together

```python
def full_pipeline(texts, query):
    """
    End-to-end: texts → concepts → graph → GNN → relevant concepts
    """
    # 1. Embed texts (Week 2)
    embeddings = embed(texts)

    # 2. Cluster into concepts (Week 4)
    concepts, labels = cluster(embeddings)

    # 3. Build concept graph (Today)
    graph = build_concept_graph(concepts, embeddings)

    # 4. Convert to PyG
    data = graph_to_pyg(graph)

    # 5. GNN forward pass
    model = ConceptGNN(...)
    concept_embeddings = model(data.x, data.edge_index)

    # 6. Find relevant concepts
    query_emb = embed([query])[0]
    relevance = propagate_relevance(model, data, query_emb)

    return relevance
```

---

## Week 8 Milestone

By the end of Week 8, you should have:

1. **Graph representation**: Build concept graphs from clusters
2. **Message passing**: Understand and implement GNN layers
3. **PyTorch Geometric**: Use GCN, GAT on your data
4. **Working GNN**: Trained model on concept tasks

### Success Criteria

```python
# You can:
# 1. Build a concept graph from embeddings
# 2. Implement basic message passing
# 3. Use PyTorch Geometric GNNs
# 4. Train a GNN for concept tasks

# You understand:
# - Message passing aggregates neighbor information
# - GNNs are permutation equivariant
# - Multiple layers = larger receptive field
# - GNNs learn on structure
```

---

## Research Log Entry

```markdown
## Week 8 Observations

### Concept Graph
- Built graph with [N] concepts, [E] edges
- Average degree: [X]
- Graph is [connected/disconnected]

### GNN Training
- Architecture: [layers, hidden dims]
- Training loss converged to: [X]
- Validation accuracy: [Y]%

### Key Insights
- GNN propagates relevance [faster/slower] than direct similarity
- [Which concepts became more similar after GNN?]
- [Did GNN discover implicit relationships?]

### Ideas for Week 9
- Use GNN to refine concept boundaries
- GNN + dynamics for reasoning
- Learn the metric from GNN
```

---

→ [Week 9: Integration](week-09-integration.md)
