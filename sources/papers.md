# Papers

Key research papers, organized by topic. Read the `[CORE]` papers; others are for deeper dives.

---

## Embeddings (Week 2)

### Efficient Estimation of Word Representations in Vector Space
**Mikolov et al., 2013** | [arXiv](https://arxiv.org/abs/1301.3781)

`[CORE]` `W2`

The Word2Vec paper. Introduced skip-gram and CBOW.

**Key insight**: Distributed representations from co-occurrence. King - man + woman = queen.

---

### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
**Reimers & Gurevych, 2019** | [arXiv](https://arxiv.org/abs/1908.10084)

`[CORE]` `W2`

How to get good sentence embeddings from transformers.

**Key insight**: Siamese/triplet training produces embeddings where cosine similarity is meaningful.

---

### Attention Is All You Need
**Vaswani et al., 2017** | [arXiv](https://arxiv.org/abs/1706.03762)

`[CORE]` `W2`

The transformer paper. Foundation of modern embeddings.

**Key insight**: Self-attention replaces recurrence. Parallelizable, captures long-range dependencies.

---

## Information Theory (Week 3)

### A Mathematical Theory of Communication
**Shannon, 1948** | [Bell Labs](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

`[CORE]` `W3`

The paper that started it all. Surprisingly readable.

**Key insight**: Information can be quantified. Entropy is the fundamental limit.

---

### Deep Learning and the Information Bottleneck Principle
**Tishby & Zaslavsky, 2015** | [arXiv](https://arxiv.org/abs/1503.02406)

`[ADVANCED]` `W3`

Information-theoretic view of deep learning.

**Key insight**: Networks compress inputs while preserving task-relevant information.

---

## Clustering (Week 4)

### HDBSCAN: Hierarchical Density-Based Clustering
**Campello, Moulavi, Sander, 2013** | [Springer](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)

`[CORE]` `W4`

The HDBSCAN algorithm we'll use for concept emergence.

**Key insight**: Density-based clustering without specifying k. Handles varying densities.

---

### UMAP: Uniform Manifold Approximation and Projection
**McInnes, Healy, Melville, 2018** | [arXiv](https://arxiv.org/abs/1802.03426)

`[CORE]` `W4`

The UMAP dimensionality reduction algorithm.

**Key insight**: Preserves both local and global structure. Based on Riemannian geometry.

---

## Dynamical Systems (Week 5)

### Computation in Attractor Networks
**Hopfield & Tank, 1985** | [Biological Cybernetics](https://link.springer.com/article/10.1007/BF00339943)

`[ADVANCED]` `W5`

Hopfield networks as energy-minimizing systems.

**Key insight**: Memory as attractors. Retrieval as dynamics toward fixed points.

---

### Energy-Based Models
**LeCun et al., 2006** | [MIT Press](https://dl.acm.org/doi/10.5555/2976456.2976483)

`[ADVANCED]` `W5`

Tutorial on energy-based learning.

**Key insight**: Learning as shaping an energy landscape. Inference as energy minimization.

---

## Variational Methods (Week 6)

### Auto-Encoding Variational Bayes
**Kingma & Welling, 2013** | [arXiv](https://arxiv.org/abs/1312.6114)

`[CORE]` `W6`

The VAE paper. Variational inference meets deep learning.

**Key insight**: Latent space structure from variational lower bound optimization.

---

### An Introduction to Variational Methods for Graphical Models
**Jordan et al., 1999** | [Machine Learning](https://link.springer.com/article/10.1023/A:1007665907178)

`[ADVANCED]` `W6`

Classical variational inference.

**Key insight**: Approximating intractable distributions by optimization.

---

## Geometric Deep Learning (Week 8)

### Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
**Bronstein et al., 2021** | [arXiv](https://arxiv.org/abs/2104.13478)

`[CORE]` `W8`

The unifying framework paper. 150+ pages of insight.

**Key insight**: Symmetry and structure as the organizing principles of deep learning.

---

### Semi-Supervised Classification with Graph Convolutional Networks
**Kipf & Welling, 2016** | [arXiv](https://arxiv.org/abs/1609.02907)

`[CORE]` `W8`

The GCN paper that popularized graph neural networks.

**Key insight**: Spectral convolutions on graphs, simplified to message passing.

---

### Graph Attention Networks
**Veličković et al., 2017** | [arXiv](https://arxiv.org/abs/1710.10903)

`[CORE]` `W8`

Attention mechanism for graphs.

**Key insight**: Learn which neighbors to attend to, not just aggregate uniformly.

---

### Neural Message Passing for Quantum Chemistry
**Gilmer et al., 2017** | [arXiv](https://arxiv.org/abs/1704.01212)

`[ADVANCED]` `W8`

Unified view of GNNs as message passing.

**Key insight**: Most GNN variants are instances of message passing.

---

## Concept Spaces & Reasoning (Week 9-10)

### Reasoning About Knowledge
**Fagin et al., 1995** | [MIT Press](https://mitpress.mit.edu/9780262562003/reasoning-about-knowledge/)

`[ADVANCED]` `W9`

Modal logic for knowledge and belief.

**Key insight**: Formal frameworks for "agent knows X" — useful for multi-source reasoning.

---

### The Free Energy Principle
**Friston, 2010** | [Nature Reviews Neuroscience](https://www.nature.com/articles/nrn2787)

`[RESEARCH]` `W10`

Brain as inference engine minimizing surprise.

**Key insight**: Perception and action as variational inference. Biology uses variational principles.

---

### Transformers Learn In-Context by Gradient Descent
**von Oswald et al., 2023** | [arXiv](https://arxiv.org/abs/2212.07677)

`[RESEARCH]` `W10`

Transformers implicitly implement optimization.

**Key insight**: In-context learning is a form of gradient descent. Connects to our "reasoning as optimization" theme.

---

## Reading Strategy

### Weeks 1-4 (Foundations)
Focus on `[CORE]` papers. Skim for intuition, don't get stuck on math.

### Weeks 5-8 (Core Concepts)
Read papers alongside notebooks. Implement key ideas.

### Weeks 9-10 (Research)
Deep dive into papers relevant to your experiments. Take notes for your publishable artifact.

---

## Paper Reading Tips

1. **First pass** (10 min): Title, abstract, figures, conclusion. Do I need this?
2. **Second pass** (30 min): Intro, skim methods, study results. What's the key insight?
3. **Third pass** (1-2 hr): Full read with pen and paper. Can I reproduce the key result?

**For implementation**: Look for the "Algorithm" box or pseudocode. Often on page 3-4.
