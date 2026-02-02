---
name: lesson-generator
description: Generate and revise curricula, arc syllabi, and Jupyter notebook lessons at any zoom level. Triggers on "design a curriculum", "I want to learn", "create a course", "generate arc/module syllabus", "create lesson/notebook", "update/revise [artifact]", or when working with educational content. Handles the full pipeline from elicitation interview through course map, arc syllabi, and individual module notebooks — plus targeted revisions with cascade awareness.
---

# Lesson Generator

A recursive curriculum generation and revision system that operates at three zoom levels:

```
Course (elicitation → curriculum brief → arc map)
    ↓
Arc (theme + context → module breakdown → publication checkpoints)
    ↓
Module Lesson (spec → belt system → executable notebook)

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

**Arc level** — reads course context:
- "Generate arc 2 syllabus"
- "Break down the Bayesian inference arc"
- "Create the syllabus for [topic]"

**Module level** — reads arc context:
- "Generate the module 0.1 notebook"
- "Create a lesson on [topic]"
- "Turn this outline into a notebook"
- "Generate arc 0, module 1 lesson"

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
- Time: How many hours/week? Any hard deadlines?
- Energy: When are you sharpest? Long blocks or short bursts?
- Environment: What tools do you have? (GPU, specific stack)
- Accountability: Self-paced? Mentor? Cohort? Deadline?
- Non-negotiables: Must include? Must avoid?

**Extract:**
- Time budget (hours/week, session length)
- Energy pattern
- Hard constraints

### Dimension 5: ARC — What shape should the journey take?

**Questions to ask:**
- Build toward one big capstone, or many small wins along the way?
- Linear progression or spiral (revisit topics with more depth)?
- How much "why" vs "how"? (theory vs practical ratio)
- When stuck, push through or skip and return later?
- How do you feel about optional "rabbit hole" sections?
- What do you want to publish along the way? (forces articulation)

**Extract:**
- Curriculum shape (linear/spiral/modular)
- Depth preference
- Escape hatch strategy
- Publication goals

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

learner_profile:
  identity:
    anchors: [list of prior knowledge to connect to]
    learning_style: [visual-first | kinesthetic | formal | mixed]
    failure_modes: [what causes disengagement]
    strengths: [what they're good at that we can leverage]
    neurodivergence:
      type: [if disclosed]
      implications: [accommodations]

destination:
  outcomes:
    - "Can [specific capability 1]"
    - "Can [specific capability 2]"
    - "Can [specific capability 3]"
  capstone: "[concrete project/artifact description]"
  identity_shift: "I am someone who can [new identity]"
  publication:
    - "[what they want to publish, for whom]"

terrain:
  perceived_gaps: [what they think they need to learn]
  actual_gaps: [what they actually need, based on destination]
  transfer_from: [adjacent knowledge to leverage]
  emotional_blockers: [fears, intimidation points]

constraints:
  time:
    hours_per_week: [number]
    session_length: [preferred session duration]
    best_time: [morning/evening/variable]
  environment:
    hardware: [GPU, RAM, etc.]
    stack: [languages, frameworks, tools]
  accountability: [self-paced | mentor | cohort | deadline]
  non_negotiables:
    must_include: [list]
    must_avoid: [list]

arc:
  shape: [linear | spiral | modular]
  theory_practice_ratio: [e.g., 20/80]
  capstone_driven: [true | false]
  escape_hatches: [true | false]
  optional_depth: [true | false]
  dopamine_strategy: [how to maintain engagement]

structure: arcs
```

## From Brief to Course Map

Analyze the brief to produce the course structure:

1. **Gap Analysis**: Compare current state (anchors) to destination (outcomes)
2. **Dependency Mapping**: What must come before what? (arcs can run concurrently if independent)
3. **Scope Fitting**: Given constraints, what's the right number of arcs?
4. **Arc Application**: Apply the chosen shape — arcs are completion-gated, not time-gated
5. **Publication Placement**: Where are the publishable artifacts?
6. **Implementation Targets**: What real code ships from each arc?

### Course Map Format (README.md for syllabus/)

```markdown
# [Course Title]

> [Design principles, AuDHD optimizations if applicable]

## The Map

[Mermaid flowchart showing arc dependencies — arcs can be concurrent]

## Arc Overview

| Arc | Title | Prerequisites | Key Deliverables |
|-----|-------|--------------|-----------------|
| 0 | [Title] | None | [deliverables] |
| 1 | [Title] | None | [deliverables] |
| 2 | [Title] | 0, 1 | [deliverables] |
| ... | ... | ... | ... |

## Publication Cadence

[Types of publications, audiences, cadence]

## Lineage

[What prior curriculum versions this absorbs, if any]
```

---

# ZOOM LEVEL 2: Arc Syllabus

## Input Context

Read the course-level context:
- Curriculum brief (learner profile, constraints, arc)
- Course map (where this arc fits, dependencies)
- Adjacent arcs (what comes before/after, what can run concurrently)

## Arc Syllabus Structure

Each arc file follows this template:

```markdown
# Arc N: Title

**Destination**: [what you can do when this arc is complete]
**Prerequisites**: [which arcs must be complete]
**Estimated sessions**: [range, not calendar time]

## The Map (mermaid flowchart of modules within arc)

## Modules
### Module N.1: Title

> *[Absorbed from ... if consolidating prior content]*

- **Motivation** (visual/scenario/analogy — fires the start engine)
- **Implementation** (build it in code — what specifically)
- **Theory backfill** (read the math that explains what you built)
- **Exercises** (at least 1 designed to produce a publishable artifact, marked [PUBLISH])
- **[OPTIONAL DEPTH]** rabbit holes (for hyperfocus sessions)

### Module N.2: Title
[repeat pattern]

## Publication Checkpoints

| # | Artifact | Type | Audience | Template |
|---|----------|------|----------|----------|
| 1 | [title] | [tweet/tutorial/code/HN post/research] | [who] | [which exercise → edit → publish] |

## Implementation Targets
- **buildlog**: [specific files/features]
- **other**: [if applicable]

## Resources (books, videos, links — specific chapters/sections per module)
```

## Arc Design Principles

- **Completion-gated**: An arc is done when you can do the thing, not when time runs out
- **Modules are sequential within an arc**: Module N.2 depends on N.1
- **Build first, theory second**: Every module starts with implementation
- **Publication as forcing function**: At least 2 publication checkpoints per arc
- **Implementation targets are real**: Exercises produce code that ships
- **Motivation hooks**: Each module starts with something that fires the start engine
- **Concurrency**: Independent arcs (e.g., Arc 0 and Arc 1) can run in parallel

---

# ZOOM LEVEL 3: Module Lesson (Notebook Generation)

## The Belt System

Notebooks are organized into **belts** that represent depth levels. Each belt is additive — higher belts include everything from lower belts.

| Belt | Layers | Content | Self-Contained? |
|------|--------|---------|-----------------|
| **Core** | L0 + L1 + L2 | Problem + Intuition + Code | Yes |
| **Depth** | L3 + L4 | CS Speak + Math Formalism | Requires Core |

**File structure**:
```
notebooks/
  arc-0-probabilistic-foundations/
    module-0.1-taste-demo/
      0.1-taste-demo-core.ipynb       # L0 + L1 + L2 (main path)
      0.1-taste-demo-depth.ipynb      # L3 + L4 (extension)
      hero-intro.png
    module-0.2-probability-counting/
      0.2-probability-counting-core.ipynb
      0.2-probability-counting-depth.ipynb
  supplements/                         # Generated on-demand
    prereq-bayes-theorem.ipynb
    prereq-big-o-notation.ipynb
```

### Layer Definitions

| Layer | Name | Content | Voice |
|-------|------|---------|-------|
| **L0** | Problem/Motivation | Real scenario, why this matters | Narrative, engaging |
| **L1** | Intuition | Analogies, anchors, visual metaphors | Peer ("Let's...") |
| **L2** | Code + Viz | Runnable demos, plots, exercises | Peer ("Let's...") |
| **L3** | CS Speak | Terminology, complexity, patterns | Internal monologue |
| **L4** | Math Formalism | Definitions, theorems, proofs | Internal monologue |

## Input Context

Read the arc-level context:
- Arc syllabus (module spec, objectives, exercises, implementation targets)
- Learner profile (from curriculum brief)
- Position in arc (early modules = more scaffolding, late = more independence)
- Publication checkpoints (is this module's exercise a publication draft?)

## Cumulative Problem Thread

**Critical**: Problems should build on each other. Each problem uses output from the previous problem.

**Good**:
```
Problem 1: Load the data → creates `data` variable
Problem 2: Extract hotel info → uses `data`, creates `hotels`
Problem 3: Filter by amenity → uses `hotels`, creates `filtered`
Problem 4: Export to CSV → uses `filtered`
```

**Bad**:
```
Problem 1: Unrelated tensor exercise
Problem 2: Different unrelated exercise
Problem 3: Another standalone exercise
```

When designing notebooks, spend time crafting a compelling problem thread that:
1. Has a real, relatable scenario
2. Builds cumulatively
3. Ends with a tangible artifact
4. Connects to the arc's implementation targets where possible

## Voice & Tone

### Core Notebooks (L0-L2): Peer Voice
```markdown
Let's figure out how to track belief about each rule. We need something that
can represent "I'm 80% confident this rule helps" and update as we get feedback.

Here's the situation: you have 20 rules, and users keep reinforcing or
contradicting them. How do we keep score?
```

### Depth Notebooks (L3-L4): Internal Monologue Voice
```markdown
I need a distribution that updates cleanly with binary feedback. The Beta
distribution is conjugate to Bernoulli, so the posterior stays in the same
family. This means updates are O(1) — just increment α or β.
```

## Narrative Flow Principles

These principles govern how concepts are introduced in every notebook. The goal is cognitive tension before framework, code before formula, recognition before naming.

### 1. Story-first opening

Every notebook opens with a concrete problem or incident, not abstract theory. The reader should feel cognitive tension ("this is wrong" or "how would I fix this?") before any framework is introduced. The opening should be a specific, real scenario — not a hypothetical.

**Pattern**: Describe an incident → show the broken output → let the reader feel the wrongness → THEN begin building the fix.

### 2. Code-before-formula rule

Mathematical notation appears ONLY AFTER working code demonstrates the concept. The sequence is always:

```
compute it → verify it → name it
```

Never: "Here's the formula. Now let's implement it." Always: "Here's what we just computed. That has a name: [formula]."

### 3. Single-example-first

Introduce concepts with ONE concrete example before building the full dataset. The reader should understand the problem on one case before seeing ten. This prevents "wall of data" paralysis and lets the reader build intuition incrementally.

**Pattern**: Show one failing case → build the fix for that one case → THEN expand to the full dataset.

### 4. Discovery order over logical order

Problems can appear in the order the reader discovers concepts, not the order they'd appear in a textbook. For example, vocabulary matching before building the test suite — because you need to feel the problem before constructing systematic tests.

The reader's curiosity drives the sequence, not taxonomic completeness.

### 5. Formula emergence

Weighted combinations, probability rules, Bayes' theorem, etc. should feel *recognized*, not *introduced*. The reader has already been computing the thing; the formula just names what they built.

**Pattern**: Build all components → combine them informally ("weighted average — you've done this since GPA") → THEN show the formula → reader thinks "oh, that's just what I already did."

### 6. Companion text callouts

Reference companion texts (ThinkBayes2, ThinkStats, Blitzstein, etc.) AFTER the reader has built something, as "go deeper" pointers. Never as prerequisites. Place them in callout blocks after verification cells.

```markdown
> **Go deeper**: You just built a weighted linear combination. For the probability
> foundations underneath this: ThinkStats Ch 1-2 (exploratory data analysis),
> ThinkBayes Ch 1 (computational Bayesian thinking), Blitzstein Ch 1-2 (formal
> probability framework).
```

### 7. Transition pattern

Use RECAP → PROBLEM RESTATEMENT → NEW APPROACH between sections. Never jump to a new concept without bridging from the previous one.

**Pattern**:
```markdown
[What we just showed] works, but [specific failure case]. Entry N [describes the
failure]. We need to check [new dimension].
```

## Problem Framing Progression

| Stage | Framing Style | Example |
|-------|---------------|---------|
| Early modules | Explicit step-by-step | "Your task is to: 1. Create X, 2. Call Y, 3. Return Z" |
| Mid modules | Goal + hints | "Extract the hotel info. Hint: each entry has a `hotel` key" |
| Late modules | Goal only | "Create `pruned_data` with only the relevant fields" |

## Visual Markers (No Emoji)

### Section Headers
```python
# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 0: THE PROBLEM
# ═══════════════════════════════════════════════════════════════════════════════
```

### Problem Headers
```python
# -------------------------------------------------------------------------------
# Problem 1: Load and Explore the Data
# -------------------------------------------------------------------------------
```

### Code Cells
```python
# --- YOUR CODE BELOW ---

# >>> SOLUTION (collapsed by default)
```

### Progress Markers (Minimal)
```python
# Tests pass. Moving on.
```

## Narrative Bookends

### Intro Section

Every notebook starts with a **narrative intro** that is:
- Fun and engaging
- Accurate and insightful
- Humorous with personality
- Sets up the problem scenario
- **Story-first**: Opens with a concrete incident, not abstract framing

Two patterns, depending on whether the module has a real incident to anchor on:

#### Pattern A: War Story (preferred when a real incident exists)

```markdown
# ═══════════════════════════════════════════════════════════════════════════════
# INTRO
# ═══════════════════════════════════════════════════════════════════════════════

## The Setup

While configuring a RunPod GPU environment, you had one agent, one rule, and
one evaluation function. The agent followed the rule perfectly — wrote a clean
Protocol, then a concrete class implementing it. The evaluation function ran
`if rule_text in agent_output`, got `False`, and logged a negative reward.

The agent was punished for doing its job.

**By the end of this notebook**, you'll have a working scorer that evaluates
rule compliance through three signals — and can explain its scores in plain
English.

Let's see exactly what happened.
```

#### Pattern B: Scenario (when no war story is available)

```markdown
# ═══════════════════════════════════════════════════════════════════════════════
# INTRO
# ═══════════════════════════════════════════════════════════════════════════════

[HERO IMAGE: Generate with ComfyUI - something visually striking related to topic]

## The Setup

You're the lead data scientist at a hotel booking startup. Your CEO bursts in:
"We need to find family-friendly hotels. Yesterday."

You have a JSON dump from the Amadeus API. 200 hotels. Nested data. Your mission:
extract the ones with pools, babysitting, or kids' clubs, and get them into a
clean CSV before the 4pm investor demo.

No pressure.

**By the end of this notebook**, you'll have a working data pipeline that:
- Loads messy JSON
- Extracts relevant fields
- Filters by criteria
- Exports clean CSV

Let's go.
```

In both patterns, the intro ends with a concrete preview of what the reader will build and a forward-leaning transition ("Let's go." / "Let's see exactly what happened.").

### Outro Section

Every notebook ends with a **narrative outro** that:
- Summarizes what was learned
- Celebrates the achievement (without being corny)
- Bridges to the next module
- Flags if this module's output is a publication draft

```markdown
# ═══════════════════════════════════════════════════════════════════════════════
# OUTRO
# ═══════════════════════════════════════════════════════════════════════════════

## What Just Happened

You took a gnarly JSON blob and turned it into a clean CSV. Along the way, you:
- Navigated nested dictionaries without losing your mind
- Used list comprehensions like a civilized person
- Applied boolean filtering to extract exactly what you needed

## Publication Note

Exercise 3 from this module is a draft for the "contains check takedown" post.
Run an edit pass and it's ready to publish.

## What's Next

In Module 0.2, we build the probability foundations that make this scoring
rigorous instead of hand-wavy.

→ [Module 0.2: Probability & Counting](../module-0.2-probability-counting/0.2-probability-counting-core.ipynb)
```

### Hero Images

Use ComfyUI to generate hero images for intro/outro sections:

```python
# Example prompt for ComfyUI
mcp__comfyui__imagine(
    description="A determined data scientist surrounded by floating JSON brackets and hotel icons, dramatic lighting, digital art style",
    output_path="/path/to/notebooks/arc-0/module-0.1/hero-intro.png",
    style="digital_art",
    quality="standard"
)
```

## Core Notebook Structure

```python
# Cell 1: Metadata (markdown)
"""
# Module 0.1: [Title] — Core

**Arc 0: Probabilistic Foundations** | Module 1 of 8

**Prerequisites**: [Prior modules or "None"]

**Time**: ~[X] minutes

**Implementation target**: [what this builds toward in real code, if applicable]

---

## Learning Objectives

By the end of this notebook, you will be able to:

- [ ] [Objective 1]
- [ ] [Objective 2]
- [ ] [Objective 3]
"""

# Cell 2: Imports (code)
"""
# Provided Code - Do NOT Edit
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline
"""

# Cell 3+: Intro section (markdown + optional hero image)

# Cell N: Layer 0 - Problem/Motivation (markdown)

# Cell N+1: Layer 1 - Intuition (markdown)

# Cell N+2: Layer 2 - Code + Viz (code cells with problems)
# Each problem:
#   - Problem header (markdown)
#   - Provided code cell (if needed)
#   - Student code cell with TODO
#   - Solution cell (collapsed)
#   - Expected output / assertions

# Cell M: Exercises section
# Mark publication-track exercises with [PUBLISH]

# Cell M+1: Outro section (markdown)

# Cell M+2: Resources (markdown — from arc's Resources section)

# Cell M+3: Next Up (markdown — link to next module)
```

## Depth Notebook Structure

Depth notebooks assume the student has completed the Core notebook.

```python
# Cell 1: Metadata (markdown)
"""
# Module 0.1: [Title] — Depth

**Extends**: [0.1-topic-core.ipynb](0.1-topic-core.ipynb)

**Belt**: Depth (CS Speak + Math Formalism)

---

## Prerequisites

This notebook assumes familiarity with:

### For CS Speak (L3):
- [ ] Big-O notation (O(1), O(n), O(log n))
- [ ] Basic data structure tradeoffs

### For Math Formalism (L4):
- [ ] [Specific prereq 1]
- [ ] [Specific prereq 2]

**Gap detected?** Ask:
> "Generate prereq supplement: [topic]"

I'll create a focused notebook that fills just that gap.
"""

# Cell 2: Layer 3 - CS Speak
# Terminology, complexity, engineering considerations
# Internal monologue voice

# Cell 3: Layer 4 - Math Formalism
# Definitions, theorems, proofs
# Confident "let's fix that" energy for proofs

# Inline prereq callouts where needed:
"""
> **PREREQ CHECK: Bayes' Theorem**
> This section assumes you can apply: posterior ∝ prior × likelihood
> If shaky, ask: "Generate prereq supplement: Bayes' theorem"
"""
```

## Exercise Cell Pattern

```python
# -------------------------------------------------------------------------------
# Problem N: [Title]
# -------------------------------------------------------------------------------

"""
[Description of what to do - explicit for early modules, goal-only for late]

Your task:
- [Step 1]
- [Step 2]
- [Step 3]

Hint: [Optional hint for early modules]
"""

# Provided Code - Do NOT Edit
provided_variable = [1, 2, 3, 4, 5]

# --- YOUR CODE BELOW ---
def solve_problem_n():
    """
    [Clear docstring with spec]
    """
    # TODO: Implement
    pass


# >>> SOLUTION (collapsed by default)
# ┌─────────────────────────────────────────────────────────────────────────────
# │ def solve_problem_n():
# │     """Solution implementation"""
# │     return [x * 2 for x in provided_variable]
# └─────────────────────────────────────────────────────────────────────────────


# Test
result = solve_problem_n()
expected = [2, 4, 6, 8, 10]
assert result == expected, f"Expected {expected}, got {result}"
# Tests pass. Moving on.
```

## Prereq Supplement Generation

When a student requests a prereq supplement:

1. **Elicit current level**: "What do you know about X? Have you seen Y notation?"

2. **Generate focused supplement** at appropriate belt:
   - Core-level (code-focused) for practical understanding
   - Depth-level (math-focused) for rigorous understanding

3. **Link back**: End with "Now return to [notebook], you're ready for [section]"

Supplement structure:
```markdown
# SUPPLEMENT: [Topic] ([Belt] Belt)

**Generated for**: [Source notebook] → [Section] prereq
**Time**: ~[X] minutes

## Why You're Here

You hit a prereq gap. This supplement teaches [topic] at the [belt] level.

## [Content organized by layers appropriate to belt]

## You're Ready

You now understand:
- [Key point 1]
- [Key point 2]

→ Return to [source notebook]
```

## Code Cell Guidelines

- All cells must run in sequence without errors
- Print shapes of tensors, intermediate results
- Prefer explicit over implicit
- Use type hints in function signatures
- Include docstrings for exercise functions
- Solutions in collapsed format (visual box)

## Adaptation Based on Learner Profile

From the curriculum brief, adjust:

- **Visual learner**: More plots, diagrams, animations
- **Kinesthetic learner**: More interactive exercises, building things
- **Formal learner**: Emphasize Depth notebooks, include more proofs
- **ADHD considerations**:
  - Add time estimates
  - Clear stopping points
  - Frequent wins
  - Engaging narrative hooks
  - Short motivation hooks that fire the start engine
  - Mermaid maps so you always know where you are
  - Optional depth rabbit holes for hyperfocus sessions
- **Math anxiety**:
  - Longer L0-L2 in Core
  - Gentler ramp in Depth
  - "Let's fix that" confidence for proofs

---

# Project Context

## Environment

Typical setup (adjust based on curriculum brief):
- `uv` package manager
- JupyterLab
- Core: numpy, scipy, torch
- Viz: matplotlib, seaborn, plotly, ipywidgets
- ML: sentence-transformers, transformers, hdbscan, umap-learn
- Stats: pymc, numpyro, arviz (Arc 2+)

## Directory Structure

```
aegir/
├── notebooks/
│   ├── arc-0-probabilistic-foundations/
│   │   ├── module-0.1-[topic]/
│   │   │   ├── 0.1-[topic]-core.ipynb
│   │   │   ├── 0.1-[topic]-depth.ipynb
│   │   │   └── hero-intro.png
│   │   ├── module-0.2-[topic]/
│   │   │   └── ...
│   │   └── ...
│   ├── arc-1-linear-algebra-calculus/
│   │   └── ...
│   └── supplements/
│       ├── prereq-[topic].ipynb
│       └── ...
├── syllabus/
│   ├── README.md              ← Course map (arc overview)
│   ├── curriculum-brief.yaml  ← Elicitation output
│   └── arcs/
│       ├── README.md          ← Arc overview + design principles
│       ├── curriculum-brief.yaml
│       ├── arc-0-probabilistic-foundations.md
│       ├── arc-1-linear-algebra-calculus.md
│       └── ...
├── sources/
│   ├── books.md
│   ├── videos.md
│   └── ...
├── wiki/
│   └── ...
├── archive/
│   └── v1-week-based/        ← Preserved original content
└── pyproject.toml
```

## Output Paths

- **Course map**: `syllabus/README.md`
- **Curriculum brief**: `syllabus/curriculum-brief.yaml`
- **Arc overview**: `syllabus/arcs/README.md`
- **Arc syllabi**: `syllabus/arcs/arc-N-[theme].md`
- **Core notebooks**: `notebooks/arc-N-[theme]/module-N.M-[topic]/N.M-[topic]-core.ipynb`
- **Depth notebooks**: `notebooks/arc-N-[theme]/module-N.M-[topic]/N.M-[topic]-depth.ipynb`
- **Supplements**: `notebooks/supplements/prereq-[topic].ipynb`

Support explicit path override: "generate to [path]"

---

# Quality Checklist

## Course Level
- [ ] All five elicitation dimensions covered
- [ ] Curriculum brief is complete and specific
- [ ] Arc sequence has clear dependencies (and concurrency where possible)
- [ ] Each arc has a concrete destination (what you can DO)
- [ ] Publication checkpoints defined per arc

## Arc Level
- [ ] Destination ties to course outcomes
- [ ] Module breakdown is sequential within the arc
- [ ] Each module has motivation → implementation → theory backfill → exercises
- [ ] At least 2 publication checkpoints with type/audience/template
- [ ] Implementation targets are specific (real files, real features)
- [ ] Resources reference specific chapters/sections, not whole books
- [ ] Estimated sessions range is realistic

## Module Lesson (Notebook) Level
- [ ] Compelling problem thread (cumulative, real scenario)
- [ ] All code cells run in sequence
- [ ] Core notebook is self-contained (L0+L1+L2)
- [ ] Depth notebook declares prereqs
- [ ] Narrative intro is engaging and fun
- [ ] Narrative outro celebrates and bridges to next module
- [ ] Publication-track exercises marked with [PUBLISH]
- [ ] Exercises have clear success criteria
- [ ] Solutions provided in collapsed format
- [ ] Visual markers follow style guide (no emoji)
- [ ] Voice matches belt level (peer vs internal monologue)
- [ ] Links to arc's implementation targets where applicable

---

# REVISION MODE

Revision mode handles updates to existing artifacts without full regeneration.

## Revision Trigger Detection

Detect revision intent from user input:

**Course-level revision:**
- "Update the curriculum to include..."
- "Add an arc on [topic]"
- "Remove the [topic] arc"
- "Reorder arcs to put [X] before [Y]"
- "The learner profile has changed — they now..."

**Arc-level revision:**
- "Update arc N to add a module on [topic]"
- "Split module N.3 into two modules"
- "Move [module] to arc N+1"
- "Change the publication checkpoint to [new target]"
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
Course revision  → Read: curriculum-brief.yaml, syllabus/README.md, syllabus/arcs/README.md
Arc revision     → Read: arc syllabus, adjacent arcs, course map
Notebook revision → Read: notebook, arc syllabus, learner profile
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
- Recalculate module numbering if applicable

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
- [ ] Arc N+1 references removed content
- [ ] Module 0.3 depends on removed exercise
- [ ] Publication checkpoint no longer aligns

**No action needed**:
- Arc N-1 (independent)
- Modules 0.1-0.2 (no dependencies on changed content)
```

## Cascade Rules by Level

### Course-Level Changes

| Change | Cascades To |
|--------|-------------|
| Add arc | Course map, arc numbering, dependency graph |
| Remove arc | Course map, downstream arc prerequisites |
| Reorder arcs | All arc files (numbering), prerequisite chains |
| Update learner profile | Potentially all notebooks (adaptation) |
| Change publication goals | Publication checkpoints across arcs |

### Arc-Level Changes

| Change | Cascades To |
|--------|-------------|
| Add module | Arc module list, module numbering |
| Remove module | Arc module list, downstream module references |
| Reorder modules | Module numbering, cross-references |
| Change publication checkpoint | Arc publication table, notebook [PUBLISH] markers |
| Update implementation target | Module exercises, notebook code |

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
