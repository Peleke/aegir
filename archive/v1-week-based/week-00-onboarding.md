# Week 0: Onboarding

**Goal**: Environment works, first visualization complete.

**Time**: 1-2 hours total

**Vibe**: Quick win. Get something cool on screen fast.

---

## Day 0.1: Setup (~30 min)

### Install Dependencies

```bash
cd ~/Documents/Projects/aegir
uv sync
```

This installs all Python dependencies in a virtual environment.

### Verify Installation

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import sentence_transformers; print('sentence-transformers OK')"
uv run python -c "import umap; print('UMAP OK')"
```

All three should print without errors.

### Launch JupyterLab

```bash
uv run jupyter lab
```

This opens JupyterLab in your browser.

---

## Day 0.2: First Notebook (~45 min)

### Complete the Environment Test

Open `notebooks/00-setup/00-environment-test.ipynb` and run all cells.

### What You'll Do

1. **Embed some sentences** using sentence-transformers
2. **Project to 2D** using UMAP
3. **Visualize** with matplotlib
4. **See clusters emerge** — similar sentences cluster together!

### Expected Output

A 2D scatter plot where:
- Related sentences are near each other
- Unrelated sentences are far apart
- You can hover/click to see labels (with plotly)

---

## Checklist

- [ ] `uv sync` completes without errors
- [ ] JupyterLab launches
- [ ] Environment test notebook runs completely
- [ ] You see your first embedding visualization
- [ ] You feel the dopamine hit

---

## Troubleshooting

### "uv: command not found"

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "No module named torch"

Make sure you're using uv to run:
```bash
uv run jupyter lab  # NOT just "jupyter lab"
```

### CUDA/GPU Issues

The curriculum works fine on CPU. If you want GPU:
```bash
# Check if CUDA available
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### "Out of memory" on embeddings

Reduce batch size in the notebook or use a smaller model:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast
# Instead of 'all-mpnet-base-v2'  # Larger, better but more memory
```

---

## What's Next

**Week 1** starts with foundations:
- NumPy/PyTorch primer (tensor operations, broadcasting, autograd)
- Linear algebra refresh (vectors, matrices, eigenthings)
- Calculus refresh (derivatives, gradients, chain rule)

But first, enjoy your visualization! You just:
1. Turned sentences into vectors
2. Preserved their semantic relationships
3. Made the invisible visible

This is the foundation of everything we'll build.

---

## Optional: Explore

If you have extra time and curiosity:

### Experiment 1: Your Own Sentences
Change the sentences in the notebook. What clusters together?

### Experiment 2: Different Models
Try different embedding models:
```python
# Fast, small
SentenceTransformer('all-MiniLM-L6-v2')

# Better quality
SentenceTransformer('all-mpnet-base-v2')

# Multilingual
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### Experiment 3: More Dimensions
Instead of 2D UMAP, try 3D:
```python
reducer = umap.UMAP(n_components=3)
```

Then visualize with plotly's `scatter_3d`.

---

## Week 0 Complete!

You've verified your environment and seen your first embedding visualization.

**Key insight**: Meaning is geometric. Similar things are near; different things are far. This simple idea underlies everything we'll build.

→ [Week 1: Foundations](week-01-foundations.md)
