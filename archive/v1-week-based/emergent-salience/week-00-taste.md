# Week 0: Taste the System

**Goal**: See emergent salience working end-to-end before learning any theory.

**Time**: 1 session × 30 min

**Milestone**: Run a demo that shows rules earning their tier through usage.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 0.1 | `00a-taste-emergent-salience.ipynb` | 30 min | Watch rules compete and emerge |

---

## Day 0.1: Watch Rules Compete

### Learning Objectives
- [ ] See Thompson Sampling select rules in real-time
- [ ] Watch salience scores update after feedback
- [ ] Observe a rule "emerge" from PERIPHERAL to CORE tier
- [ ] Get excited about what we're building

### The Demo

You'll run a pre-built simulation where:
1. A pool of 20 rules starts with uniform priors
2. Simulated feedback reinforces/contradicts rules
3. Thompson Sampling explores while exploiting
4. Salience scores evolve, tiers shift
5. Visualization shows the "emergence" in real-time

**No theory yet.** Just watch it work.

### What You'll See

```
Session 1:  [▓▓░░░░░░░░] Rule "test-first" sampled → reinforced → α=2
Session 10: [▓▓▓▓▓░░░░░] Rule "test-first" now EMERGING tier
Session 50: [▓▓▓▓▓▓▓▓░░] Rule "test-first" now CORE tier
            Rule "spaces-not-tabs" got contradicted → β=4, dropping...
```

### Why Start Here

Traditional courses: "First, let's define a Beta distribution..."

This course: "First, watch your future system work. *Then* we'll explain why."

You can't fall off a curriculum you're already invested in.

---

## Week 0 Milestone: Emotional Buy-In

After this session, you should feel:
- [ ] "I want to understand how that works"
- [ ] "I can see where this is going"
- [ ] "This is worth 30 minutes a day"

If you don't feel this, something's wrong with the demo. Flag it.

---

## Linguistics Anchor Preview

In Week 1, we'll connect this to language acquisition:

> A child doesn't memorize grammar rules. They hear utterances, get feedback (comprehension, correction), and *emerge* a grammar. Rules that work get reinforced. Rules that fail get pruned. The grammar that survives is the one that earned its structure.

Your rule system works the same way.

---

→ [Week 1: Probabilistic Primitives](week-01-primitives.md)
