# Lesson Generator

A recursive curriculum generation and revision system for Claude Code.

```
Person → Elicitation → Curriculum Brief → Course Map
                                              ↓
                                         Week Syllabi
                                              ↓
                                      Jupyter Notebooks
```

## What It Does

Generates educational content at three zoom levels:

| Level | Input | Output |
|-------|-------|--------|
| **Course** | Elicitation interview | Curriculum brief + week map |
| **Week** | Course context + theme | Day breakdown + milestone |
| **Notebook** | Week spec + objectives | Four-layer pedagogy lesson |

Plus **revision mode** for targeted updates with cascade awareness.

## Four-Layer Pedagogy

Every concept explanation follows this progression:

1. **Intuition** — Spatiotemporal analogies, physical metaphors
2. **Code + Visualization** — Runnable demos that make concepts visible
3. **CS Speak** — Terminology, complexity, engineering considerations
4. **Mathematical Formalism** — Rigorous notation, definitions, proofs

## Triggers

**Generation:**
- "I want to learn [topic]"
- "Design a curriculum for..."
- "Generate week N syllabus"
- "Create the [notebook-name] notebook"

**Revision:**
- "Update [artifact] to include..."
- "Add [topic] to week N"
- "Fix [issue] in [notebook]"

## Elicitation Dimensions

Course design starts with structured discovery across five dimensions:

1. **Identity** — Cognitive profile, learning style, failure modes
2. **Destination** — Concrete capabilities, capstone vision
3. **Terrain** — Knowledge gaps, emotional blockers, transfer opportunities
4. **Constraints** — Time, energy, environment, accountability
5. **Arc** — Curriculum shape, theory/practice ratio, escape hatches

## Output Structure

```
project/
├── syllabus/
│   ├── README.md              ← Course map
│   ├── curriculum-brief.yaml  ← Elicitation output
│   └── week-NN-[theme].md     ← Week syllabi
└── notebooks/
    └── NN-[module]/
        └── NNx-[name].ipynb   ← Lessons
```

## Usage

Install as a Claude Code skill:

```bash
# Skills live in .claude/skills/
cp -r lesson-generator /path/to/project/.claude/skills/
```

Then just ask Claude to generate curriculum content. The skill auto-detects the appropriate zoom level and mode.

## License

MIT
