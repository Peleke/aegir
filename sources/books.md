# Books

Curated textbooks with chapter guides. Read what you need, when you need it.

---

## Foundations (Week 1)

### Build a Large Language Model (From Scratch)
**Sebastian Raschka** | [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)

`[CORE]` `W1` `W2`

Essential companion for understanding transformers and embeddings.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1-2 | Tokenization, embeddings | Week 1-2 |
| Ch 3 | Attention mechanism | Week 2 |
| Ch 4-5 | Training, pretraining | Reference |

**Why**: Builds intuition for how meaning gets encoded. Hands-on implementation.

---

### Deep Learning with PyTorch
**Eli Stevens, Luca Antiga** | [Manning](https://www.manning.com/books/deep-learning-with-pytorch)

`[INTRO]` `W1`

Solid PyTorch foundation if you need more than the primer.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 3-4 | Tensors, autograd | Week 1 |
| Ch 5-6 | Neural networks | Week 1 |
| Ch 8 | Convolutions | Optional |

---

## Information Theory (Week 3)

### Elements of Information Theory
**Cover & Thomas** | [Wiley](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959)

`[CORE]` `W3`

The definitive textbook. Dense but rewarding.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1 | Introduction | Skim |
| Ch 2 | Entropy, relative entropy, MI | Week 3 (essential) |
| Ch 3 | Asymptotic equipartition | Reference |
| Ch 5 | Data compression | Week 3 (compression = understanding) |

**Alternative**: David MacKay's "Information Theory, Inference, and Learning Algorithms" (free online) — more accessible, Bayesian perspective.

### Going Deeper: Quantum Information

**Quantum Computation and Quantum Information**
**Nielsen & Chuang** | [Cambridge](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)

`[ADVANCED]` `W3`

The "Mike and Ike" book. Chapter 11 (Entropy and Information) provides deep foundations for Shannon entropy, von Neumann entropy, and information-theoretic quantities from first principles.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 2.1-2.2 | Linear algebra, postulates | Reference (if curious) |
| Ch 11.1-11.3 | Classical/quantum entropy | Week 3 extension |
| Ch 11.4 | Strong subadditivity | Reference |

**Why**: Rigorous foundations of information theory. Even without quantum interest, Ch 11 is one of the clearest treatments of entropy.

---

## Dynamical Systems (Week 5)

### Nonlinear Dynamics and Chaos
**Steven Strogatz** | [Westview Press](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering)

`[CORE]` `W5`

Beautifully written. The standard introduction.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1-2 | 1D flows, bifurcations | Week 5 |
| Ch 4-5 | 2D flows, limit cycles | Week 5 |
| Ch 6 | Phase plane analysis | Week 5 |
| Ch 7-8 | Chaos, attractors | Optional (Week 5 extension) |

**Why**: Builds geometric intuition for dynamics. Many pictures.

---

## Variational Calculus (Week 6)

### Calculus of Variations
**Gelfand & Fomin** | [Dover](https://store.doverpublications.com/products/9780486414485)

`[CORE]` `W6`

Classic, concise, inexpensive.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1 | Functionals, variations | Week 6 (essential) |
| Ch 2 | Euler-Lagrange equations | Week 6 (essential) |
| Ch 3 | Applications | Week 6 |
| Ch 4+ | Advanced | Reference |

---

### Mechanics
**Landau & Lifshitz** | [Butterworth-Heinemann](https://www.elsevier.com/books/mechanics/landau/978-0-08-050347-9)

`[ADVANCED]` `W6`

The physics perspective. How least action underlies all of mechanics.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1 | Lagrangian mechanics | Week 6 |
| Ch 2 | Conservation laws | Week 6 (Noether's theorem) |

**Why**: Shows why physicists believe in variational principles.

---

## Differential Geometry (Week 7)

### Riemannian Geometry
**Do Carmo** | [Birkhäuser](https://link.springer.com/book/10.1007/978-1-4757-2201-7)

`[CORE]` `W7`

Standard graduate text. First 3 chapters are accessible.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 0-1 | Review, manifolds | Week 7 |
| Ch 2 | Metrics, connections | Week 7 |
| Ch 3 | Geodesics | Week 7 |
| Ch 4+ | Curvature | Reference |

---

### Visual Differential Geometry and Forms
**Tristan Needham** | [Princeton](https://press.princeton.edu/books/hardcover/9780691203690/visual-differential-geometry-and-forms)

`[INTRO]` `W7`

Stunning visualizations. Geometric intuition first.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1-3 | Curvature intuition | Week 7 |
| Ch 4-6 | Geodesics, parallel transport | Week 7 |

**Why**: Makes geometry visual and intuitive.

---

## Geometric Deep Learning (Week 8)

### Graph Representation Learning
**William Hamilton** | [Morgan & Claypool](https://www.cs.mcgill.ca/~wlh/grl_book/)

`[CORE]` `W8`

Comprehensive coverage of GNNs and graph learning.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1-2 | Background, node embeddings | Week 8 |
| Ch 5 | Graph neural networks | Week 8 (essential) |
| Ch 6 | Message passing | Week 8 |

**Free online**: Author provides free PDF.

---

## Research & Integration (Week 9-10)

### Build a Reasoning Model (From Scratch)
**Sebastian Raschka** | [Manning](https://www.manning.com/books/build-a-reasoning-model-from-scratch) (early access)

`[RESEARCH]` `W9-10`

Companion to the LLM book. Chain-of-thought, verification, reinforcement learning for reasoning.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1-2 | Reasoning fundamentals | Week 9 |
| Ch 3+ | CoT, verification | Week 10 |

**Why**: Our variational reasoning could connect to these techniques. Good for grounding the theory in practice.

---

### Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
**Bronstein, Bruna, Cohen, Veličković** | [arXiv](https://arxiv.org/abs/2104.13478)

`[RESEARCH]` `W8-10`

The "GDL textbook." Unifying framework.

Read after Week 8 notebooks. Essential for understanding how geometry + learning connect.

---

### Build a Text-to-Image Generator (From Scratch)
**Sebastian Raschka** | [Manning](https://www.manning.com/books/build-a-text-to-image-generator-from-scratch) (early access)

`[OPTIONAL]` `W7`

Diffusion models operate in latent space. Could build intuition for how meaning traverses high-dimensional spaces.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 2-3 | VAEs, latent spaces | Week 7 (optional) |
| Ch 4+ | Diffusion | Reference |

**Why**: Latent space geometry in action. Not core curriculum, but a nice rabbit hole.

---

### Information Geometry and Its Applications
**Shun-ichi Amari** | [Springer](https://link.springer.com/book/10.1007/978-4-431-55978-8)

`[ADVANCED]` `W9-10`

Where information theory meets differential geometry.

| Chapter | Topic | When to Read |
|---------|-------|--------------|
| Ch 1-2 | Statistical manifolds | Week 9 |
| Ch 3-4 | Fisher information metric | Week 9 |

**Why**: Could inform how we define metrics on concept space.

---

## Quick Reference

| Topic | Primary Book | Alternative |
|-------|--------------|-------------|
| PyTorch | Stevens & Antiga | Raschka Ch 1-2 |
| Embeddings | Raschka Ch 2-3 | Jurafsky NLP book |
| Information Theory | Cover & Thomas | MacKay (free) |
| Dynamical Systems | Strogatz | — |
| Variational | Gelfand & Fomin | Landau (physics) |
| Geometry | Do Carmo | Needham (visual) |
| GDL | Hamilton | Bronstein et al. |

---

## Post-Curriculum: Physics Track

After completing the core curriculum, these books offer a deeper physics foundation for variational methods and dynamical systems.

### Learn Physics with Functional Programming
**Scott Walck** | [No Starch Press](https://nostarch.com/learn-physics-functional-programming)

`[POST]`

Physics from first principles using Haskell. Unusual approach that builds Newtonian mechanics, electromagnetism, and more through functional composition.

**Why read it**: The functional decomposition of physical laws mirrors how we're thinking about concept dynamics. Also: building physics simulations from scratch reinforces variational intuition.

**Approach**: Work through at your own pace after core curriculum. The Haskell is accessible; the physics is the point.

---

### The Theoretical Minimum Series
**Leonard Susskind** | [Stanford Continuing Studies](https://theoreticalminimum.com/)

`[POST]`

Susskind's "Theoretical Minimum" series (Classical Mechanics, Quantum Mechanics, etc.) teaches physics the way physicists actually think about it. Free lectures online.

| Book | Topic | Relevance |
|------|-------|-----------|
| Classical Mechanics | Lagrangian/Hamiltonian | Deepens Week 6 |
| Statistical Mechanics | Entropy, partition functions | Connects to Week 3 |
| Quantum Mechanics | State space, measurement | Enriches geometry intuition |

**Why read it**: If you want the "real" physics behind variational principles and phase space. The lectures are excellent — book optional.

---

*These are enrichment tracks, not prerequisites. The core curriculum is self-contained.*
