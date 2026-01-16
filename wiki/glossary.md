# Glossary

Quick reference for key terms used throughout the curriculum.

---

## A

**Action (S)**
: Integral of the Lagrangian over time: S = ∫L dt. Nature chooses paths that make the action stationary.

**Attractor**
: A state or set of states that trajectories converge to. Concepts could be attractors in thought-space.

**Atlas**
: A collection of overlapping charts that cover a manifold.

---

## B

**Basin of Attraction**
: The set of all initial conditions that eventually flow to a given attractor.

**Bifurcation**
: A qualitative change in system behavior as a parameter varies (e.g., new fixed points appear).

---

## C

**Chart**
: A local coordinate system on a manifold.

**Christoffel Symbols (Γ)**
: Connection coefficients that describe how coordinates change as you move on a curved manifold.

**Concept Space**
: The geometric space where concepts live. In our framework: embedding space + salience field + metric structure.

**Contrastive Learning**
: Training embeddings by pushing similar items together and dissimilar items apart.

**Co-occurrence**
: When two things appear together (in text, events, etc.). Foundation of distributional semantics.

**Curvature**
: How much a manifold deviates from flat space. Gaussian curvature K > 0 is sphere-like, K < 0 is saddle-like.

---

## D

**Distributional Hypothesis**
: "You shall know a word by the company it keeps." Words in similar contexts have similar meanings.

**Dynamical System**
: A rule dx/dt = f(x) that determines how state evolves over time.

---

## E

**Eigenvalue**
: For a matrix A and vector v, if Av = λv, then λ is an eigenvalue and v is an eigenvector. Determines stability at fixed points.

**Embedding**
: A map from discrete objects (words, concepts) to continuous vectors where geometry encodes meaning.

**Entropy (H)**
: Measure of uncertainty/surprise: H(X) = -Σ p(x) log p(x). Higher entropy = more uncertainty.

**Equivariance**
: A function f is equivariant to transform g if f(g·x) = g·f(x). The output transforms the same way as the input.

**Euler-Lagrange Equation**
: The condition for a path to extremize a functional: d/dt(∂L/∂q̇) = ∂L/∂q.

---

## F

**Fixed Point**
: A state x* where f(x*) = 0 — the system doesn't move.

**Flow**
: The set of all trajectories of a dynamical system.

**Functional**
: A function that takes a function and returns a number. S[q] = ∫L(q, q̇, t) dt is a functional.

---

## G

**Geodesic**
: The shortest path between two points on a manifold. Straight lines in flat space, great circles on a sphere.

**GNN (Graph Neural Network)**
: A neural network that operates on graph-structured data via message passing.

**Gradient System**
: A dynamical system where dx/dt = -∇V(x). Always flows "downhill" on the potential V.

**Graph**
: A structure G = (V, E) with vertices V and edges E.

---

## H

**HDBSCAN**
: Hierarchical Density-Based Spatial Clustering. Finds clusters of varying density without prespecifying k.

---

## I

**Information**
: Quantified surprise. An event with probability p has information -log₂(p) bits.

---

## J

**Jacobian**
: Matrix of partial derivatives. At a fixed point, eigenvalues of the Jacobian determine stability.

---

## K

**KL Divergence (D_KL)**
: Asymmetric measure of difference between distributions: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)).

---

## L

**Lagrangian (L)**
: L = T - V (kinetic minus potential energy). The function being integrated to get the action.

**Lie Group**
: A continuous symmetry group (like rotations) with smooth structure.

**Limit Cycle**
: A closed periodic orbit that trajectories converge to.

**Line Element**
: Infinitesimal distance ds² = Σ g_ij dx^i dx^j. Defines the metric.

---

## M

**Manifold**
: A space that locally looks like ℝⁿ. Globally can be curved or have interesting topology.

**Message Passing**
: GNN operation where nodes aggregate information from their neighbors.

**Metric (Riemannian)**
: A positive-definite matrix g_ij at each point that defines distances and angles.

**Mutual Information (I)**
: How much knowing X reduces uncertainty about Y: I(X;Y) = H(X) - H(X|Y).

---

## N

**Negative Sampling**
: Training technique for embeddings: contrast positive examples with random negative examples.

---

## P

**Phase Space**
: The space of all possible states of a system. For a pendulum: (angle, angular velocity).

**Potential (V)**
: A scalar field where the force is F = -∇V. Systems flow toward potential minima.

**Principal of Least Action**
: Physical paths extremize (usually minimize) the action integral.

---

## R

**Riemann Tensor**
: Measures curvature in n dimensions. Describes how vectors rotate when parallel transported around loops.

---

## S

**Saddle Point**
: A fixed point that is stable in some directions and unstable in others.

**Salience**
: How "attention-grabbing" something is. In our framework: a field that shapes which concepts are active.

**Sentence Transformer**
: A transformer model trained to produce meaningful sentence embeddings.

**Skip-gram**
: Word2Vec variant that predicts context words from a center word.

**Stability**
: A fixed point is stable if nearby trajectories converge to it.

**Strange Attractor**
: A fractal attractor associated with chaotic systems.

---

## T

**Tangent Space (T_pM)**
: The space of all possible velocities/directions at a point p on a manifold.

**t-SNE**
: t-distributed Stochastic Neighbor Embedding. Dimensionality reduction preserving local structure.

**Trajectory**
: The path x(t) traced by a system evolving according to its dynamics.

---

## U

**UMAP**
: Uniform Manifold Approximation and Projection. Dimensionality reduction preserving global and local structure.

---

## V

**Variational Calculus**
: Calculus for finding functions that extremize functionals.

**Vector Field**
: Assignment of a vector to each point in space. f(x) defines a vector field.

---

## W

**Word2Vec**
: Algorithm for learning word embeddings from co-occurrence. "King - man + woman = queen."

---

## Symbols

| Symbol | Meaning |
|--------|---------|
| H(X) | Entropy of X |
| I(X;Y) | Mutual information between X and Y |
| D_KL(P\|\|Q) | KL divergence from Q to P |
| S[q] | Action functional |
| L | Lagrangian |
| ∇ | Gradient operator |
| g_ij | Metric tensor components |
| Γ^k_ij | Christoffel symbols |
| R^ρ_σμν | Riemann curvature tensor |
| T_pM | Tangent space at point p |
| δS | Variation of the action |
