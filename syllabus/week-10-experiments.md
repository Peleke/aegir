# Week 10: Experiments & Publishable Artifact

**Goal**: Run real experiments. Create something publishable.

**Time**: 5+ days × 90 min = 7.5+ hours

**Milestone**: Blog post, paper draft, or open-source release.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 10.1 | 10a-multi-source-rag | 90 min | Concept space for RAG |
| 10.2 | 10b-concept-emergence | 90 min | Study how concepts emerge |
| 10.3 | 10c-variational-reasoning | 90 min | Test least-action hypothesis |
| 10.4 | Analysis & writing | 60 min | Compile results |
| 10.5 | Publishable artifact | 60 min | Finalize output |

---

## Experiment Ideas

### Experiment A: Multi-Source RAG

**Hypothesis**: Concept space improves retrieval across heterogeneous sources.

```python
class MultiSourceRAGExperiment:
    """
    Test concept space for retrieval across multiple data sources.

    Sources:
    - Articles/documentation
    - Code snippets
    - Conversation logs
    - (Your own data)
    """

    def __init__(self):
        self.space = ConceptSpace()
        self.sources = {}

    def add_source(self, name: str, texts: List[str]):
        """Add a data source."""
        self.sources[name] = {
            'texts': texts,
            'start_idx': len(self.space.texts),
            'end_idx': len(self.space.texts) + len(texts)
        }
        self.space.add_texts(texts)

    def query(self, question: str, top_k: int = 5):
        """
        Query across all sources using concept space.

        Returns passages from relevant concepts.
        """
        # Update salience
        self.space.update_salience(question)
        self.space.propagate_salience()

        # Get top concepts
        top_concepts = self.space.get_salient_concepts(top_k)

        # Get passages from these concepts
        results = []
        for concept in top_concepts:
            # Find texts belonging to this concept
            for exemplar in concept.exemplars:
                idx = self.space.texts.index(exemplar)
                source = self._find_source(idx)
                results.append({
                    'text': exemplar,
                    'source': source,
                    'concept': concept.name,
                    'salience': self.space.salience[concept.id]
                })

        return results

    def _find_source(self, idx: int) -> str:
        for name, info in self.sources.items():
            if info['start_idx'] <= idx < info['end_idx']:
                return name
        return 'unknown'

    def evaluate(self, test_queries: List[Dict]):
        """
        Evaluate retrieval quality.

        test_queries: [{'query': ..., 'relevant': [...]}]
        """
        results = {
            'precision': [],
            'recall': [],
            'cross_source': []  # Did we retrieve from multiple sources?
        }

        for tq in test_queries:
            retrieved = self.query(tq['query'], top_k=5)
            retrieved_texts = [r['text'] for r in retrieved]
            retrieved_sources = set(r['source'] for r in retrieved)

            # Precision/Recall
            relevant_retrieved = set(retrieved_texts) & set(tq['relevant'])
            precision = len(relevant_retrieved) / len(retrieved_texts) if retrieved_texts else 0
            recall = len(relevant_retrieved) / len(tq['relevant']) if tq['relevant'] else 0

            results['precision'].append(precision)
            results['recall'].append(recall)
            results['cross_source'].append(len(retrieved_sources) > 1)

        return {
            'mean_precision': np.mean(results['precision']),
            'mean_recall': np.mean(results['recall']),
            'cross_source_rate': np.mean(results['cross_source'])
        }
```

### Experiment B: Concept Emergence

**Hypothesis**: Concepts emerge naturally from embedding clusters and are semantically meaningful.

```python
class ConceptEmergenceExperiment:
    """
    Study how concepts emerge from clustering.

    Questions:
    - Do cluster names match human intuition?
    - How stable are concepts across random seeds?
    - Do concepts capture semantic categories?
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def run_emergence_study(self, n_trials: int = 5):
        """Run clustering multiple times with different seeds."""
        all_concepts = []

        for trial in range(n_trials):
            space = ConceptSpace()
            # Set seed in embedder if possible, or accept variability
            space.add_texts(self.texts)
            all_concepts.append(space.concepts)

        return self.analyze_stability(all_concepts)

    def analyze_stability(self, all_concepts: List[List[Concept]]):
        """Analyze stability of concepts across trials."""
        # Find concepts that appear consistently
        concept_names = [set(c.name for c in concepts) for concepts in all_concepts]

        # Jaccard similarity between trials
        similarities = []
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                intersection = len(concept_names[i] & concept_names[j])
                union = len(concept_names[i] | concept_names[j])
                similarities.append(intersection / union if union > 0 else 0)

        return {
            'mean_jaccard': np.mean(similarities),
            'std_jaccard': np.std(similarities),
            'consistent_concepts': self._find_consistent(concept_names)
        }

    def _find_consistent(self, concept_names: List[set]) -> List[str]:
        """Find concepts that appear in majority of trials."""
        from collections import Counter
        all_names = [name for names in concept_names for name in names]
        counts = Counter(all_names)
        threshold = len(concept_names) / 2
        return [name for name, count in counts.items() if count >= threshold]

    def human_evaluation(self, space: ConceptSpace, n_samples: int = 10):
        """
        Generate evaluation form for human judgment.

        For each concept, show exemplars and ask:
        - Does this concept make sense? (1-5)
        - Is the name appropriate? (1-5)
        - What would you name it?
        """
        form = []
        for concept in np.random.choice(space.concepts, min(n_samples, len(space.concepts)), replace=False):
            form.append({
                'concept_id': concept.id,
                'auto_name': concept.name,
                'exemplars': concept.exemplars[:5],
                'coherence_rating': None,  # 1-5
                'name_rating': None,  # 1-5
                'suggested_name': None
            })

        return form
```

### Experiment C: Variational Reasoning

**Hypothesis**: Reasoning paths that minimize action are more natural/useful.

```python
class VariationalReasoningExperiment:
    """
    Test whether variational (least-action) paths are better.

    Compare:
    - Shortest graph path
    - Straight-line embedding interpolation
    - Variational (action-minimizing) path
    """

    def __init__(self, space: ConceptSpace):
        self.space = space

    def compare_paths(self, start_query: str, goal_query: str):
        """Compare different path-finding strategies."""
        results = {}

        # Graph shortest path
        start_concept = self.space._nearest_concept(
            self.space.embedder.encode([start_query])[0]
        )
        goal_concept = self.space._nearest_concept(
            self.space.embedder.encode([goal_query])[0]
        )

        try:
            graph_path = nx.shortest_path(
                self.space.graph, start_concept.id, goal_concept.id
            )
            results['graph'] = {
                'path': [self.space.concepts[i].name for i in graph_path],
                'length': len(graph_path),
                'action': self._compute_action_for_path(graph_path)
            }
        except nx.NetworkXNoPath:
            results['graph'] = None

        # Interpolation path
        interp_path = self.space._interpolate_path(start_concept, goal_concept, 5)
        results['interpolation'] = {
            'path': [c.name for c in interp_path],
            'length': len(interp_path),
            'action': self._compute_action_for_concepts(interp_path)
        }

        # Variational path
        var_path = self.space.variational_reasoning_path(start_query, goal_query)
        results['variational'] = {
            'path': [c.name for c in var_path],
            'length': len(var_path),
            'action': self._compute_action_for_concepts(var_path)
        }

        return results

    def _compute_action_for_path(self, concept_ids: List[int]) -> float:
        """Compute action along a path of concept IDs."""
        action = 0
        for i in range(len(concept_ids) - 1):
            c1 = self.space.concepts[concept_ids[i]]
            c2 = self.space.concepts[concept_ids[i+1]]

            effort = np.linalg.norm(c2.centroid - c1.centroid)
            salience = self.space.salience[c1.id]
            action += effort - salience

        return action

    def _compute_action_for_concepts(self, concepts: List[Concept]) -> float:
        """Compute action along a path of Concepts."""
        return self._compute_action_for_path([c.id for c in concepts])

    def evaluate_path_quality(self, paths: Dict, human_ratings: Dict = None):
        """Evaluate path quality using metrics and optional human ratings."""
        metrics = {}

        for name, path_info in paths.items():
            if path_info is None:
                metrics[name] = None
                continue

            metrics[name] = {
                'length': path_info['length'],
                'action': path_info['action'],
                'concepts_visited': path_info['path']
            }

            if human_ratings and name in human_ratings:
                metrics[name]['human_rating'] = human_ratings[name]

        return metrics
```

---

## Creating Your Publishable Artifact

### Option 1: Blog Post

**Structure**:
1. **Hook**: Why concept spaces matter for AI memory
2. **Background**: Brief on embeddings, clustering, dynamics
3. **Method**: Your ConceptSpace implementation
4. **Experiments**: Results from above
5. **Code**: Link to repo
6. **Future Work**: What's next

**Target**: ~2000-3000 words, 4-6 figures

```python
def generate_blog_figures(space: ConceptSpace, experiment_results: Dict):
    """Generate figures for blog post."""
    figures = {}

    # Figure 1: Concept space overview
    figures['concept_space'] = visualize_salience(space)

    # Figure 2: Reasoning trajectory example
    trajectory = space.reason("API testing", "deployment")
    figures['reasoning'] = visualize_reasoning(space, trajectory)

    # Figure 3: Experiment results
    # ... (bar charts, comparisons)

    return figures
```

### Option 2: Technical Report / Paper

**Structure**:
1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiments
6. Results
7. Discussion
8. Conclusion

**Target**: 8-12 pages, LaTeX format

### Option 3: Open Source Library

**Structure**:
```
aegir/
├── README.md           # Quick start, examples
├── docs/               # Full documentation
│   ├── getting-started.md
│   ├── api-reference.md
│   └── tutorials/
├── aegir/              # Package
│   ├── __init__.py
│   ├── concept_space.py
│   ├── clustering.py
│   ├── dynamics.py
│   └── visualization.py
├── examples/           # Example notebooks
├── tests/              # Unit tests
└── pyproject.toml
```

---

## Research Log Template

```markdown
# Experiment: [Name]

## Hypothesis
[What you're testing]

## Method
[How you're testing it]

## Results

### Quantitative
| Metric | Value |
|--------|-------|
| ... | ... |

### Qualitative
[Observations, examples, edge cases]

## Figures
[Include key visualizations]

## Discussion
[What do results mean?]

## Limitations
[What might be wrong?]

## Next Steps
[What to try next]
```

---

## Week 10 Milestones

### Day 10.1-10.3: Run Experiments

- [ ] Multi-source RAG experiment complete
- [ ] Concept emergence study done
- [ ] Variational reasoning comparison finished

### Day 10.4: Analysis

- [ ] Results compiled
- [ ] Key figures generated
- [ ] Insights documented

### Day 10.5: Publishable Artifact

- [ ] Draft complete
- [ ] Figures finalized
- [ ] Code cleaned up
- [ ] Ready for feedback

---

## What's Next?

Congratulations! You've completed the Aegir curriculum. You now have:

1. **Mathematical foundations**: Information theory, variational calculus, dynamical systems, differential geometry, GDL
2. **Practical skills**: Embeddings, clustering, GNNs, visualization
3. **Working prototype**: ConceptSpace implementation
4. **Research artifact**: Blog post, paper, or library

### Continuing the Journey

- **Depth**: Pick one area (e.g., variational methods) and go deeper
- **Breadth**: Connect to other domains (neuroscience, philosophy of mind)
- **Application**: Integrate with your multi-source RAG system
- **Community**: Share your work, get feedback, collaborate

### Big Questions to Explore

1. Can concepts be learned end-to-end?
2. How does salience relate to attention in transformers?
3. Is there a "correct" metric on concept space?
4. How do concepts compose hierarchically?
5. Can reasoning truly be variational?

---

## Final Reflection

```markdown
## My Aegir Journey

### What I Learned
- [Key concepts that clicked]
- [Skills I developed]
- [Surprises along the way]

### What I Built
- [Describe your ConceptSpace]
- [Key features]
- [What works well]

### What's Next
- [Immediate next steps]
- [Long-term research directions]
- [How this fits into larger goals]

### Acknowledgments
[People, resources, tools that helped]
```

---

**Thank you for taking this journey.** The intersection of physics, information theory, and machine learning is rich territory. Your concept space is just the beginning.

🌊 Aegir
