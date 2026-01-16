# Aegir

*The jötunn who personified the sea and hosted feasts for the Æsir*

A self-paced curriculum for learning the mathematical foundations of **concept spaces** and **variational reasoning** — toward building physics-inspired AI memory and reasoning systems.

## The Vision

What if reasoning paths through concept space followed the **principle of least action**? What if concepts **emerged** from experience rather than being predefined? What if justification was **intrinsic to the geometry** rather than external metadata?

This curriculum builds the mathematical foundation to explore these ideas.

## Quick Start

```bash
# Clone and enter
cd ~/Documents/Projects/aegir

# Install dependencies (requires uv)
uv sync

# Launch JupyterLab
jupyter lab

# Start with notebooks/00-setup/00-environment-test.ipynb
```

## Structure

```
aegir/
├── wiki/           # Primers on core concepts
├── sources/        # Curated reading list
├── syllabus/       # Week-by-week curriculum
├── notebooks/      # Hands-on learning (Jupyter)
├── lib/            # Reusable Python code
├── data/           # Sample data
└── experiments/    # Research logs
```

## Curriculum Overview

| Week | Topic | Milestone |
|------|-------|-----------|
| 0 | Onboarding | First embedding visualization |
| 1 | Foundations | Gradient descent from scratch |
| 2 | Embeddings | Mini Word2Vec working |
| 3 | Information Theory | Compute MI between clusters |
| 4 | Clustering | Name emergent concept clusters |
| 5 | Dynamical Systems | Visualize attractor landscape |
| 6 | Variational Calculus | Derive shortest path |
| 7 | Differential Geometry | Compute geodesics on sphere |
| 8 | Geometric Deep Learning | Build simple GNN |
| 9 | Integration | Working concept space prototype |
| 10 | Experiments | Research results |

## Design Principles

**ADHD-Optimized**
- Micro-modules (30-45 min each)
- Visual progress tracking
- Quick wins early
- Interleaved theory/practice

**Hands-On First**
- Every concept → immediate code
- Never more than 10 min reading before doing
- Visualize everything

**Build Toward Something Real**
- Not academic exercises
- Culminates in working prototype
- Designed for multi-source RAG integration

## Prerequisites

- Python fluency
- Basic familiarity with linear algebra/calculus (we'll refresh)
- Curiosity about the nature of concepts and reasoning

## The Big Idea

We're building toward a system where:

1. **Concepts emerge** from density in embedding space
2. **Salience** creates a potential field that shapes attention
3. **Reasoning** is movement through this space
4. **Paths** minimize an action functional (least effort + maximum relevance)
5. **Justification** is intrinsic — the path exists because it's optimal

Think of it as **Lagrangian mechanics for thought**.

## Resources

See `sources/` for the full reading list. Key resources:

- **3Blue1Brown** - Linear algebra, calculus intuition
- **Strogatz** - Nonlinear Dynamics and Chaos
- **Bronstein et al.** - Geometric Deep Learning
- **Raschka** - Build an LLM from Scratch

## License

MIT
