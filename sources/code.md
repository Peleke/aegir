# Code

Reference implementations and libraries. Use these to understand algorithms and as starting points for experiments.

---

## Core Libraries

### sentence-transformers
[GitHub](https://github.com/UKPLab/sentence-transformers) | [Docs](https://www.sbert.net/)

`W2`

Production-ready sentence embeddings.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['Hello world', 'How are you?'])
```

**Use for**: All embedding needs in the curriculum.

---

### PyTorch Geometric (PyG)
[GitHub](https://github.com/pyg-team/pytorch_geometric) | [Docs](https://pytorch-geometric.readthedocs.io/)

`W8`

The graph neural network library.

```python
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Create graph data
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
x = torch.randn(2, 16)  # 2 nodes, 16 features
data = Data(x=x, edge_index=edge_index)

# Build GNN
conv = GCNConv(16, 32)
out = conv(data.x, data.edge_index)
```

**Use for**: Week 8 GNN experiments.

---

### UMAP
[GitHub](https://github.com/lmcinnes/umap) | [Docs](https://umap-learn.readthedocs.io/)

`W2` `W4`

Dimensionality reduction that preserves structure.

```python
import umap
reducer = umap.UMAP(n_components=2)
embedding_2d = reducer.fit_transform(high_dim_embeddings)
```

**Use for**: Visualizing embedding spaces throughout curriculum.

---

### HDBSCAN
[GitHub](https://github.com/scikit-learn-contrib/hdbscan) | [Docs](https://hdbscan.readthedocs.io/)

`W4`

Density-based clustering without specifying k.

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(embeddings)
```

**Use for**: Concept emergence in Week 4.

---

### NetworkX
[GitHub](https://github.com/networkx/networkx) | [Docs](https://networkx.org/)

`W8`

Graph manipulation and analysis.

```python
import networkx as nx
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3)])
nx.draw(G, with_labels=True)
```

**Use for**: Building concept graphs, visualization.

---

## Reference Implementations

### Word2Vec from Scratch

Multiple implementations to study:

| Repo | Language | Notes |
|------|----------|-------|
| [gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py) | Python | Production quality |
| [word2vec-numpy](https://github.com/nathanrooy/word2vec-from-scratch-with-python) | NumPy | Educational |
| [Raschka's LLM book](https://github.com/rasbt/LLMs-from-scratch) | PyTorch | Ch 2-3 |

**Exercise**: Implement skip-gram yourself, then compare to these.

---

### GNN Implementations

| Repo | What | Notes |
|------|------|-------|
| [PyG examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) | GCN, GAT, GraphSAGE | Official examples |
| [DGL](https://github.com/dmlc/dgl) | Alternative GNN library | Good tutorials |
| [Spektral](https://github.com/danielegrattarola/spektral) | Keras/TF GNNs | If you prefer TF |

---

### Dynamical Systems

| Repo | What | Notes |
|------|------|-------|
| [PySINDy](https://github.com/dynamicslab/pysindy) | Sparse dynamics discovery | Data-driven dynamics |
| [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) | Julia ODE solvers | Best ODE library |
| [scipy.integrate](https://docs.scipy.org/doc/scipy/reference/integrate.html) | Python ODE solvers | What we'll use |

**Code snippet for Week 5**:
```python
from scipy.integrate import solve_ivp
import numpy as np

def pendulum(t, y):
    theta, omega = y
    return [omega, -np.sin(theta)]

sol = solve_ivp(pendulum, [0, 10], [0.1, 0], dense_output=True)
```

---

### Variational Calculus

| Repo | What | Notes |
|------|------|-------|
| [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) | Numerical optimization | minimize, BVP solvers |
| [autograd](https://github.com/HIPS/autograd) | Automatic differentiation | Good for learning |
| [JAX](https://github.com/google/jax) | Modern autodiff | Week 6+ |

**Code snippet for geodesics**:
```python
from scipy.integrate import solve_bvp
import numpy as np

def geodesic_ode(t, y):
    # y = [x, dx/dt]
    # For flat space: d²x/dt² = 0
    return np.vstack([y[1], np.zeros_like(y[0])])

def boundary_conditions(ya, yb):
    return np.array([ya[0] - 0, yb[0] - 1])  # x(0)=0, x(1)=1
```

---

## Visualization Code

### Phase Portraits
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_portrait(f, xlim, ylim, density=20):
    """Plot vector field for 2D system dx/dt = f(x, y)."""
    x = np.linspace(*xlim, density)
    y = np.linspace(*ylim, density)
    X, Y = np.meshgrid(x, y)
    U, V = f(X, Y)

    # Normalize arrows
    M = np.sqrt(U**2 + V**2)
    M[M == 0] = 1
    U, V = U/M, V/M

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, M, cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Portrait')
```

### 3D Embedding Visualization
```python
import plotly.express as px
import pandas as pd

def plot_embeddings_3d(embeddings, labels, hover_text=None):
    """Interactive 3D scatter plot of embeddings."""
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'z': embeddings[:, 2],
        'label': labels,
        'text': hover_text or labels
    })
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='label', hover_data=['text'])
    fig.show()
```

### Graph Visualization
```python
import networkx as nx
import matplotlib.pyplot as plt

def plot_concept_graph(G, node_colors=None, node_sizes=None):
    """Plot a concept graph with optional styling."""
    pos = nx.spring_layout(G, k=2, iterations=50)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors or 'lightblue',
                           node_size=node_sizes or 500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
```

---

## Project Templates

### Experiment Structure
```
experiments/
├── 001-concept-clustering/
│   ├── README.md           # Hypothesis, methods, results
│   ├── config.yaml         # Parameters
│   ├── run.py              # Main script
│   ├── analysis.ipynb      # Analysis notebook
│   └── figures/            # Generated figures
```

### Notebook Structure
```python
# %% [markdown]
# # Experiment: [Title]
# **Hypothesis**: [What you're testing]
# **Date**: [Date]

# %% Setup
import numpy as np
import matplotlib.pyplot as plt
# ...

# %% Data Loading
# ...

# %% Experiment
# ...

# %% Analysis
# ...

# %% Conclusions
# - Finding 1
# - Finding 2
```

---

## Useful Snippets

### Cosine Similarity Matrix
```python
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)
```

### Progress Bars
```python
from tqdm.auto import tqdm
for item in tqdm(items, desc="Processing"):
    process(item)
```

### Reproducibility
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Save/Load Models
```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pt'))
```
