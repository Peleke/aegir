---
name: lesson-generator
description: Generate and revise curricula, syllabi, and Jupyter notebook lessons at any zoom level. Triggers on "design a curriculum", "I want to learn", "create a course", "generate week/module syllabus", "create lesson/notebook", "update/revise [artifact]", or when working with educational content. Handles the full pipeline from elicitation interview through course map, week syllabi, and individual notebooks — plus targeted revisions with cascade awareness.
---

# Lesson Generator

A recursive curriculum generation and revision system that operates at three zoom levels:

```
Course (elicitation → curriculum brief → week map)
    ↓
Week (theme + context → day breakdown → milestone)
    ↓
Notebook (spec → four-layer pedagogy → executable lesson)

+ Revision Mode (update any artifact with cascade awareness)
```

## Mode Detection

Detect the appropriate mode from user input:

**Revision mode** — triggered by update language:
- "Update [artifact] to..."
- "Add [content] to [artifact]"
- "Remove [content] from [artifact]"
- "Fix [issue] in [artifact]"
- "The [artifact] needs..."

→ See [Revision Mode](#revision-mode) section

**Generation mode** — triggered by creation language (see below)

## Zoom Level Detection (Generation Mode)

Detect the appropriate level from user input:

**Course level** — triggers elicitation:
- "Design a curriculum for..."
- "I want to learn..."
- "Create a course on..."
- "Help me build a learning path for..."

**Week level** — reads course context:
- "Generate week 3 syllabus"
- "Break down the embeddings module"
- "Create the syllabus for [topic]"

**Notebook level** — reads week context:
- "Generate the [notebook-name] notebook"
- "Create a lesson on [topic]"
- "Turn this outline into a notebook"

---

# ZOOM LEVEL 1: Course Design

## Elicitation Protocol

Course design begins with structured discovery. Run through these five dimensions:

### Dimension 1: IDENTITY — Who is this person?

Map their cognitive profile, not demographics:

**Questions to ask:**
- What do you already know well? (anchors for new concepts)
- What have you tried to learn before that didn't stick? Why?
- When you learn something new, what makes it "click"?
  - Seeing it visually
  - Building something with it
  - Understanding the math/theory
  - Connecting to something familiar
  - Teaching it to someone else
- How do you know when you've really learned something vs just recognized it?

**Extract:**
- Learning style (visual/kinesthetic/formal)
- Prior knowledge anchors (what to connect new concepts to)
- Failure modes (what causes them to disengage)

### Dimension 2: DESTINATION — What can they DO after?

Not "understand X" — concrete capabilities:

**Questions to ask:**
- Describe a project you want to be able to build after this.
- If someone challenged you in 3 months, what would you want to demonstrate?
- What would make you say "I'm now the kind of person who can ___"?
- Is there a specific artifact you want to ship? (paper, library, app, job)

**Extract:**
- Capability checklist (specific skills)
- Capstone vision (concrete project)
- Identity shift (who they become)

### Dimension 3: TERRAIN — What's the landscape?

Discover what they don't know they don't know:

**Questions to ask:**
- What topics do you think you need to learn? (their mental model)
- What feels scary or intimidating about this domain?
- Are there specific tools/frameworks you need to use?
- What adjacent domains do you already know? (for transfer)

**Extract:**
- Perceived vs actual prerequisites
- Emotional blockers
- Transfer opportunities from adjacent knowledge

### Dimension 4: CONSTRAINTS — What's the box?

**Questions to ask:**
- Time: How many hours/week? For how many weeks?
- Energy: When are you sharpest? Long blocks or short bursts?
- Environment: What tools do you have? (GPU, specific stack)
- Accountability: Self-paced? Mentor? Cohort? Deadline?
- Non-negotiables: Must include? Must avoid?

**Extract:**
- Time budget (hours/week × weeks)
- Energy pattern
- Hard constraints

### Dimension 5: ARC — What shape should the journey take?

**Questions to ask:**
- Build toward one big capstone, or many small wins along the way?
- Linear progression or spiral (revisit topics with more depth)?
- How much "why" vs "how"? (theory vs practical ratio)
- When stuck, push through or skip and return later?
- How do you feel about optional "rabbit hole" sections?

**Extract:**
- Curriculum shape (linear/spiral/modular)
- Depth preference
- Escape hatch strategy

## Elicitation Style

- Ask 2-3 questions at a time, not all at once
- Reflect back what you heard before moving to next dimension
- It's OK if answers are vague — probe deeper on what matters
- Watch for implicit constraints (e.g., mentions of ADHD → add escape hatches)
- Total elicitation: 5-10 exchanges, not an interrogation

## Output: Curriculum Brief

After elicitation, produce this structured artifact:

```yaml
# Curriculum Brief: [Title]
generated: [date]

## Learner Profile
identity:
  anchors: [list of prior knowledge to connect to]
  learning_style: [visual-first | kinesthetic | formal | mixed]
  failure_modes: [what causes disengagement]
  strengths: [what they're good at that we can leverage]

## Destination
outcomes:
  - "Can [specific capability 1]"
  - "Can [specific capability 2]"
  - "Can [specific capability 3]"
capstone: "[concrete project/artifact description]"
identity_shift: "I am someone who can [new identity]"

## Terrain
perceived_gaps: [what they think they need to learn]
actual_gaps: [what they actually need, based on destination]
transfer_from: [adjacent knowledge to leverage]
emotional_blockers: [fears, intimidation points]

## Constraints
time:
  hours_per_week: [number]
  total_weeks: [number]
  session_length: [preferred session duration]
  best_time: [morning/evening/variable]
environment:
  hardware: [GPU, RAM, etc.]
  stack: [languages, frameworks, tools]
accountability: [self-paced | mentor | cohort | deadline]
non_negotiables:
  must_include: [list]
  must_avoid: [list]

## Arc
shape: [linear | spiral | modular]
theory_practice_ratio: [e.g., 30/70]
capstone_driven: [true | false]
escape_hatches: [true | false]
optional_depth: [true | false]
dopamine_strategy: [how to maintain engagement]
```

## From Brief to Course Map

Analyze the brief to produce the course structure:

1. **Gap Analysis**: Compare current state (anchors) to destination (outcomes)
2. **Dependency Mapping**: What must come before what?
3. **Time Fitting**: Given constraints, what's the right scope?
4. **Arc Application**: Apply the chosen shape (linear/spiral/modular)
5. **Milestone Placement**: Where are the dopamine hits?

### Course Map Format (README.md for syllabus/)

```markdown
# [Course Title]

[One-paragraph vision statement]

---

## The Map

[Mermaid flowchart showing week dependencies]

---

## The Big Picture

[ASCII or table showing the arc]

---

## Weekly Overview

| Week | Theme | Key Milestone | Notebook Count |
|------|-------|---------------|----------------|
| 0 | Onboarding | Environment works | 1 |
| 1 | [Theme] | [Concrete deliverable] | [N] |
| ... | ... | ... | ... |

**Total**: ~[N] notebooks, ~[N] days at [session length]

---

## Design Principles

[ADHD optimizations, escape hatches, pacing notes based on learner profile]

---

## Capstone Track

[How the capstone builds throughout the course]

---

## Prerequisites Check

[What they need before starting]

---

## Week Guides

| Week | Guide |
|------|-------|
| 0 | [Onboarding](week-00-onboarding.md) |
| 1 | [Theme](week-01-theme.md) |
| ... | ... |
```

---

# ZOOM LEVEL 2: Week Syllabus

## Input Context

Read the course-level context:
- Curriculum brief (learner profile, constraints, arc)
- Course map (where this week fits, dependencies)
- Adjacent weeks (what comes before/after)

## Week Syllabus Structure

```markdown
# Week N: [Title]

**Goal**: [One-sentence outcome — what they can DO after this week]

**Time**: [X days × Y min = Z hours]

**Milestone**: [Concrete deliverable they'll produce]

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| N.1 | [filename] | [X min] | [Brief description] |
| N.2 | [filename] | [X min] | [Brief description] |
| ... | ... | ... | ... |

---

## Day N.1-N.2: [Section Title]

### Learning Objectives
- [ ] [Concrete, assessable objective]
- [ ] [Concrete, assessable objective]

### Key Concepts

[High-level overview of concepts — detail goes in notebooks]

### Exercises

1. [Exercise with clear deliverable]
2. [Exercise with clear deliverable]

### Resources
- [Links]
- [Wiki references]

---

[Repeat for each day grouping]

---

## Week N Milestone: [Title]

[Code example or description of the milestone deliverable]

**Success criteria**: [How to know it works]

---

## Why This Week Matters

[Connection to overall arc, what it enables later]

---

## Reflection Questions

1. [Conceptual question]
2. [Conceptual question]

---

## Week N Complete!

[Summary, bridge to next week]

→ [Week N+1: Title](week-N+1-file.md)
```

## Week Design Principles

- **Milestone-driven**: Every week ends with something tangible
- **Front-load wins**: Put engaging content early in the week
- **Escape hatches**: Mark optional depth sections
- **Connections**: Explicitly link to prior and future weeks
- **Realistic timing**: Account for setup, debugging, rabbit holes

---

# ZOOM LEVEL 3: Notebook Generation

## Input Context

Read the week-level context:
- Week syllabus (day spec, objectives, exercises)
- Learner profile (from curriculum brief)
- Position in arc (early = more scaffolding, late = more independence)

## Four-Layer Pedagogy

Every concept explanation follows this progression, from concrete to abstract:

### Layer 1: Intuition (Spatiotemporal Analogies)

Start with physical, visual, or everyday analogies. Make it tangible.

```markdown
## What is a Gradient?

Imagine standing on a hillside in thick fog. You can't see the valley below,
but you can feel which way is steepest under your feet. That direction of
steepest descent is the gradient. If you repeatedly take small steps downhill,
you'll eventually reach the bottom—that's gradient descent.

The gradient is a compass that always points uphill. We follow it backwards.
```

**Guidelines:**
- Use physical metaphors (hills, water, gravity, motion)
- Reference things they can visualize or have felt
- Connect to prior knowledge anchors from learner profile
- Keep it concrete — no formulas yet

### Layer 2: Code + Visualization

Runnable demo that makes the concept visible.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 2D landscape
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Simple bowl

# Compute gradient at a point
point = np.array([2.0, 1.5])
gradient = 2 * point  # For f(x,y) = x² + y², ∇f = [2x, 2y]

# Visualize
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.quiver(point[0], point[1], -gradient[0], -gradient[1],
           color='red', scale=20, label='Negative gradient')
plt.scatter(*point, color='red', s=100, zorder=5)
plt.colorbar(label='f(x,y)')
plt.legend()
plt.title('Gradient Descent Direction')
plt.show()
```

**Guidelines:**
- Code must run without errors
- Always visualize when possible
- Print intermediate values to show what's happening
- Keep cells focused — one concept per visualization

### Layer 3: CS Speak

Terminology, complexity, algorithmic patterns, engineering considerations.

```markdown
## Computational Perspective

**Gradient descent** is a first-order iterative optimization algorithm.
Given a differentiable function f, the update rule is:

    x_{t+1} = x_t - η · ∇f(x_t)

where η is the **learning rate** (step size).

**Time complexity**: O(n) per iteration for n parameters
**Space complexity**: O(n) to store the gradient

**Variants**:
- **SGD**: Stochastic gradient descent (mini-batch updates)
- **Momentum**: Accumulates velocity to escape local minima
- **Adam**: Adaptive learning rates per parameter
```

**Guidelines:**
- Introduce proper terminology
- Discuss complexity and tradeoffs
- Connect to standard implementations
- Mention practical engineering concerns

### Layer 4: Mathematical Formalism

Rigorous notation, definitions, proofs where appropriate.

```markdown
## Mathematical Formulation

**Definition**: Let f: ℝⁿ → ℝ be differentiable. The gradient of f at x is:

$$\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n} \right]^T$$

**Theorem (Gradient points in direction of steepest ascent)**:
For any unit vector u, the directional derivative satisfies:

$$D_u f(x) = \nabla f(x) \cdot u \leq \|\nabla f(x)\|$$

with equality when u = ∇f(x) / ‖∇f(x)‖.

**Proof**: By Cauchy-Schwarz, |∇f · u| ≤ ‖∇f‖‖u‖ = ‖∇f‖. ∎
```

**Guidelines:**
- Use proper LaTeX notation
- Include definitions before theorems
- Provide proof sketches for key results
- This layer is for building mathematical maturity

## Notebook Structure

### Required Sections

1. **Title & Metadata** (cell 1: markdown)
   ```markdown
   # [Number][Letter]: [Title]

   **Week N, Day(s) X-Y** | [Module Name]

   **Prerequisites**: [Prior notebooks]

   ---

   ## Learning Objectives

   By the end of this notebook, you will be able to:

   - [ ] [Objective 1]
   - [ ] [Objective 2]

   ---
   ```

2. **Imports** (cell 2: code)
   ```python
   # Standard library
   from pathlib import Path

   # Core
   import numpy as np
   import torch
   import matplotlib.pyplot as plt

   # Config
   plt.style.use('seaborn-v0_8-whitegrid')
   %matplotlib inline
   ```

3. **Core Content** (multiple cells)
   - Organize by concept
   - Each concept gets all four layers
   - Use `---` separators between major sections

4. **Exercises** (markdown + code)
   ```markdown
   ## Exercise N: [Title]

   [Description of what to implement]

   **Hint**: [Optional hint]
   ```
   ```python
   def exercise_n():
       """
       [Docstring with clear spec]
       """
       # TODO: Implement this
       pass

   # Test
   result = exercise_n()
   print(f"Expected: [X], Got: {result}")
   ```

5. **Why This Matters** (markdown)
   - Connection to curriculum arc
   - Real-world applications
   - What this enables later

6. **Resources** (markdown)
   - External links (videos, papers, docs)
   - Wiki/glossary references
   - Optional deep dives

7. **Reflection Questions** (markdown)
   - Conceptual check-ins
   - "What would happen if..."
   - Bridge to next notebook

8. **Next Up** (markdown)
   - Brief preview
   - Link to next notebook

## Code Cell Guidelines

- All cells must run in sequence without errors
- Print shapes of tensors, intermediate results
- Prefer explicit over implicit
- Use type hints in function signatures
- Include docstrings for exercise functions

## Adaptation Based on Learner Profile

From the curriculum brief, adjust:

- **Visual learner**: More plots, diagrams, animations
- **Kinesthetic learner**: More interactive exercises, building things
- **Formal learner**: Emphasize Layer 4, include more proofs
- **ADHD considerations**: Add time estimates, escape hatches, frequent wins
- **Math anxiety**: Longer Layer 1-2, gentler ramp to Layer 4

---

# Project Context

## Environment

Typical setup (adjust based on curriculum brief):
- `uv` package manager
- JupyterLab
- Core: numpy, scipy, torch
- Viz: matplotlib, seaborn, plotly, ipywidgets
- ML: sentence-transformers, transformers, hdbscan, umap-learn

## Directory Structure

```
project/
├── notebooks/
│   ├── 00-setup/
│   ├── 01-[module]/
│   ├── 02-[module]/
│   └── ...
├── syllabus/
│   ├── README.md          ← Course map
│   ├── curriculum-brief.yaml  ← Elicitation output
│   ├── week-00-onboarding.md
│   ├── week-01-[theme].md
│   └── ...
├── wiki/
│   └── glossary.md
└── pyproject.toml
```

## Output Paths

- **Course map**: `syllabus/README.md`
- **Curriculum brief**: `syllabus/curriculum-brief.yaml`
- **Week syllabi**: `syllabus/week-NN-[theme].md`
- **Notebooks**: `notebooks/NN-[module]/NNx-[name].ipynb`

Support explicit path override: "generate to [path]"

---

# Quality Checklist

## Course Level
- [ ] All five elicitation dimensions covered
- [ ] Curriculum brief is complete and specific
- [ ] Week sequence has clear dependencies
- [ ] Milestones are concrete and achievable
- [ ] Time budget is realistic

## Week Level
- [ ] Goals tie to course outcomes
- [ ] Day breakdown fits time constraints
- [ ] Milestone is tangible
- [ ] Connections to prior/next weeks explicit

## Notebook Level
- [ ] All code cells run in sequence
- [ ] Every concept has all four layers
- [ ] Visualizations present for major concepts
- [ ] Exercises have clear success criteria
- [ ] "Why This Matters" connects to arc
- [ ] Resources include relevant links

---

# REVISION MODE

Revision mode handles updates to existing artifacts without full regeneration.

## Revision Trigger Detection

Detect revision intent from user input:

**Course-level revision:**
- "Update the curriculum to include..."
- "Add a week on [topic]"
- "Remove the [topic] module"
- "Reorder weeks to put [X] before [Y]"
- "The learner profile has changed — they now..."

**Week-level revision:**
- "Update week N to add [topic]"
- "Split day 3 into two days"
- "Move [notebook] to week N+1"
- "Change the milestone to [new milestone]"
- "Add an exercise on [topic]"

**Notebook-level revision:**
- "Add a section on [concept]"
- "The explanation of [X] needs more intuition"
- "Add an exercise for [skill]"
- "Update the code to use [library/pattern]"
- "Fix the [broken thing]"

## Revision Workflow

### Step 1: Read Current State

Read the artifact being revised and its context:

```
Course revision → Read: curriculum-brief.yaml, README.md (course map)
Week revision   → Read: week syllabus, adjacent weeks, course map
Notebook revision → Read: notebook, week syllabus, learner profile
```

### Step 2: Identify the Delta

Categorize the change:

| Change Type | Description | Cascade Risk |
|-------------|-------------|--------------|
| **Additive** | New content, no existing content affected | Low |
| **Modificative** | Existing content updated in place | Medium |
| **Structural** | Reordering, splitting, merging | High |
| **Subtractive** | Removing content | High |

### Step 3: Plan the Edit

For each change type:

**Additive changes:**
- Identify insertion point
- Check for dependency conflicts (does new content require prerequisites?)
- Draft new content following existing patterns

**Modificative changes:**
- Identify exact scope of change
- Preserve surrounding context
- Maintain consistent voice/style

**Structural changes:**
- Map all affected sections
- Update cross-references
- Recalculate time budgets if applicable

**Subtractive changes:**
- Check for downstream dependencies
- Identify orphaned references
- Confirm removal scope with user if ambiguous

### Step 4: Execute Edit

Apply the minimal change needed. Preserve:
- Existing structure where unchanged
- Cross-references that still apply
- Formatting conventions
- Voice and tone

### Step 5: Flag Cascade Impacts

After editing, report downstream impacts:

```markdown
## Revision Complete

**Changed**: [what was modified]

**Downstream impacts to review**:
- [ ] Week N+1 references removed content
- [ ] Notebook 3b depends on removed exercise
- [ ] Time budget now exceeds constraint

**No action needed**:
- Week N-1 (independent)
- Notebooks 1a-2c (no dependencies on changed content)
```

## Cascade Rules by Level

### Course-Level Changes

| Change | Cascades To |
|--------|-------------|
| Add week | Course map, week numbering |
| Remove week | Course map, downstream week references |
| Reorder weeks | All week files (numbering), cross-references |
| Update learner profile | Potentially all notebooks (adaptation) |
| Change capstone | Capstone track, final weeks |

### Week-Level Changes

| Change | Cascades To |
|--------|-------------|
| Add notebook | Week overview table, day numbering |
| Remove notebook | Week overview, downstream notebook references |
| Reorder days | Notebook filenames, cross-references |
| Change milestone | Milestone section, "Why This Matters" in notebooks |
| Update time budget | Overview table, may require content cuts |

### Notebook-Level Changes

| Change | Cascades To |
|--------|-------------|
| Add section | Table of contents (if present), flow |
| Remove section | Cross-references within notebook |
| Update exercise | Test code, success criteria |
| Change imports | All code cells using those imports |
| Fix code bug | Downstream cells depending on output |

## Revision Principles

1. **Minimal diff**: Change only what's necessary
2. **Preserve voice**: Match existing style and tone
3. **Maintain integrity**: Don't break working code or valid references
4. **Flag, don't fix silently**: Report cascade impacts explicitly
5. **Confirm destructive changes**: Ask before removing significant content

## Revision Quality Checklist

- [ ] Read existing artifact before editing
- [ ] Identified change type (additive/modificative/structural/subtractive)
- [ ] Preserved unchanged content
- [ ] Updated cross-references
- [ ] Flagged downstream impacts
- [ ] Tested code still runs (for notebooks)
