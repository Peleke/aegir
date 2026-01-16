# Geometric Deep Learning Primer

## TL;DR

Geometric deep learning is ML on structured data: graphs, manifolds, groups. **Message passing** lets nodes aggregate information from neighbors. **Equivariance** means the network respects symmetries. This is how we'll learn on concept spaces.

---

## Core Intuition

Standard neural networks assume data is a flat vector or grid. But much data has structure:
- Social networks (graphs)
- Molecules (3D point clouds)
- Concept spaces (??? — that's what we're figuring out)

GDL builds networks that respect this structure.

---

## Why Graphs?

A **graph** G = (V, E) has:
- **Vertices/Nodes** V: entities (concepts, words, articles)
- **Edges** E: relationships (co-occurrence, similarity, citation)

### Why Not Just Flatten?

You could embed a graph as a big adjacency matrix. But:
- Loses locality (neighbor structure)
- Doesn't generalize to different graph sizes
- Doesn't respect permutation symmetry (node order is arbitrary)

GNNs solve these problems.

---

## Message Passing

The core operation in graph neural networks:

```
For each node i:
    1. Gather messages from neighbors j ∈ N(i)
    2. Aggregate messages (sum, mean, max)
    3. Update node representation
```

```python
import torch
import torch.nn as nn

class SimpleMessagePassing(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.message_fn = nn.Linear(in_dim, out_dim)
        self.update_fn = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, in_dim]
        edge_index: Edge list [2, num_edges]
        """
        src, dst = edge_index

        # Create messages
        messages = self.message_fn(x[src])  # [num_edges, out_dim]

        # Aggregate by destination (sum)
        aggregated = torch.zeros(x.size(0), messages.size(1))
        aggregated.index_add_(0, dst, messages)

        # Update
        out = self.update_fn(torch.cat([x, aggregated], dim=1))
        return out
```

### Visual

```
Step 1: Message          Step 2: Aggregate       Step 3: Update

   B                        B                        B
   │                        │                        │
   ↓                        ↓                        ↓
   A ← C                    A ←─ sum ─→ A'          A' (new embedding)
   ↑                        ↑
   │                        │
   D                        D

Each neighbor sends       All messages              Node updates its
a message to A           combined at A              representation
```

---

## Graph Neural Network Architectures

### GCN (Graph Convolutional Network)

```python
# Simplified GCN layer
h_i = σ(Σ_j (1/√(d_i d_j)) W h_j)
```

Where d_i, d_j are node degrees. This is "normalized aggregation."

### GAT (Graph Attention Network)

Learns attention weights for which neighbors matter more:

```python
α_ij = softmax_j(attention(h_i, h_j))
h_i = σ(Σ_j α_ij W h_j)
```

### GraphSAGE

Samples neighbors and uses learnable aggregators:

```python
h_i = σ(W · concat(h_i, AGG({h_j : j ∈ N(i)})))
```

Where AGG could be mean, max, LSTM over neighbors.

---

## Equivariance

A function f is **equivariant** to a transformation g if:

```
f(g·x) = g·f(x)
```

Transform the input, then apply f = Apply f, then transform the output.

### Example: Rotation Equivariance

If you rotate the input, the output rotates the same way:
- Good for molecules (rotate the molecule → predictions rotate)
- Good for images (rotate the image → feature maps rotate)

### Permutation Equivariance (Graphs)

Graph neural networks are equivariant to node permutation:
- Reorder the nodes → output reorders the same way
- This is why GNNs work regardless of node labeling

```
        Original                    Permuted

    1 ─── 2                     3 ─── 1
    │     │          →          │     │
    4 ─── 3                     2 ─── 4

    GNN output for node 1   =   GNN output for node 3
    (same neighbors, same result)
```

---

## Symmetry Groups

**Group theory** formalizes symmetries:
- **Rotation group SO(3)**: 3D rotations
- **Permutation group S_n**: Reordering n objects
- **Translation group**: Shifting in space

A network is **equivariant to group G** if it respects all transformations in G.

### Lie Groups

Continuous symmetry groups (rotations, translations) are **Lie groups**. They have:
- Smooth structure (you can take derivatives)
- Lie algebra (infinitesimal transformations)

**For concept spaces**: If the concept space has symmetries (e.g., analogies like A:B::C:D), we might want equivariant networks.

---

## Practical GNN with PyTorch Geometric

```python
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Create a simple graph
edge_index = torch.tensor([
    [0, 1, 1, 2],  # source nodes
    [1, 0, 2, 1]   # target nodes
], dtype=torch.long)

x = torch.randn(3, 16)  # 3 nodes, 16 features each

data = Data(x=x, edge_index=edge_index)

# Define a simple GNN
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = SimpleGNN()
out = model(data)
print(out.shape)  # [3, 16] - updated node embeddings
```

---

## Message Passing as Diffusion

Message passing is like diffusion on a graph:
- Information flows from neighbors to node
- Multiple layers = information from farther away
- k layers = k-hop neighborhood

```
Layer 1: Node sees 1-hop neighbors
Layer 2: Node sees 2-hop neighbors (neighbors of neighbors)
Layer 3: Node sees 3-hop neighbors
...
```

**Caution**: Too many layers → over-smoothing (all nodes become similar).

---

## Relevance for Aegir

| Concept | Application |
|---------|-------------|
| Graph structure | Concept space as a graph (concepts = nodes, relations = edges) |
| Message passing | Information flows between related concepts |
| Attention | Some relations matter more than others |
| Equivariance | Network respects concept space symmetries |
| Multi-hop | Reasoning chains as multi-hop paths |

**Key insight**: If concepts form a graph (via co-occurrence, similarity, or learned edges), GNNs can learn to propagate relevance, discover clusters, and support reasoning.

---

## The GDL Blueprint

Bronstein et al.'s "Geometric Deep Learning: Grids, Groups, Graphs" unifies:

| Domain | Symmetry | Network Type |
|--------|----------|--------------|
| Sets | Permutation | DeepSets, Transformers |
| Grids | Translation | CNNs |
| Graphs | Node permutation | GNNs |
| Manifolds | Diffeomorphisms | Gauge CNNs |
| Groups | Group action | Equivariant networks |

**For us**: Concept spaces might be manifolds with graph structure. We might need hybrid approaches.

---

## Key Formulas

```
Message passing:     h_i^(l+1) = UPDATE(h_i^(l), AGG({MSG(h_i^(l), h_j^(l)) : j ∈ N(i)}))
GCN:                 H^(l+1) = σ(D̃^(-½) Ã D̃^(-½) H^(l) W^(l))
Attention:           α_ij = softmax_j(a^T [Wh_i || Wh_j])
Equivariance:        f(g·x) = g·f(x)
```

---

## Going Deeper

- **Paper**: Bronstein et al., "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (2021)
- **Course**: [Stanford CS224W - Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
- **Library**: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- **Video**: [Petar Veličković's talks on GNNs](https://www.youtube.com/watch?v=uF53xsT7mjc)
