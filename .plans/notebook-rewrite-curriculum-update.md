# Plan: Notebook Rewrite + Curriculum Update + Skill Update

Three parallel tracks. All can execute concurrently.

---

## Track 1: Rewrite Module 0.1 Notebook

**File**: `notebooks/arc-0-probabilistic-foundations/module-0.1-taste-demo/0.1-taste-demo-core.ipynb`

**Problem**: Current notebook dumps concepts without motivation. The contains check arrives out of nowhere, the linear combination is a fait accompli, and the 10 BuildlogEntry objects land before you care.

**Fix**: Restructure narrative scaffolding using a story-first pattern (STORY → CODE → NAME THE CONCEPT). All existing code/exercises preserved; only the narrative flow and cell ordering change.

### New Cell Structure

| Cell | Type | Content |
|------|------|---------|
| 0 | md | Module header + learning objectives (unchanged) |
| 1 | code | Imports (unchanged) |
| 2 | md | **INTRO**: War story — RunPod GPU config incident. ONE agent, ONE rule, ONE wrong evaluation. "Let's see exactly what happened." |
| 3 | code | Show the single failing example: one rule + one agent_output + `contains` returns 0.0. Reader feels the wrongness. |
| 4 | md | "The agent followed the rule. The check said it didn't. How would YOU evaluate this?" |
| 5 | md | Guide intuition: "What words jump out? Protocol, ABC, abstractmethod — the rule's *vocabulary*." No formula, no label. |
| 6 | code | Provided: `extract_code_blocks` + `SYNONYM_MAP` |
| 7 | md | **Problem 2 header**: "Score vocabulary overlap between rule and output" |
| 8 | code | `linguistic_signal` exercise (stub + collapsed solution) |
| 9 | code | Verify on the single example. Print score. "One example works. But what about edge cases?" |
| 10 | md | Transition: list edge cases as questions (rule doesn't apply? agent quotes rule but doesn't follow? code crashes?) |
| 11 | md | **Problem 1 header**: "Build the Test Suite" — now the reader wants the full dataset |
| 12 | code | `BuildlogEntry` dataclass + all 10 `ENTRIES` |
| 13 | code | Run `contains_check` on all 10 + run `linguistic_signal` on all 10, side-by-side. Reader sees vocabulary matching is better but Entry 3 (false positive) still scores high. |
| 14 | md | Transition: "Words aren't enough. Entry 3 mentions interfaces without building one. We need to check *structure*." |
| 15 | md | **Problem 3 header**: structural signal |
| 16 | code | `check_interface_before_impl`, `check_date_validation`, `structural_signal` (stub + solution) |
| 17 | code | Verification — Entry 3 now caught |
| 18 | md | "Two signals down. What about Entries 7-8 where the code looks right but crashes?" |
| 19 | md | **Problem 4 header**: outcome signal |
| 20 | code | `outcome_signal` exercise |
| 21 | md | **The pivot**: "Three numbers per entry. To collapse them into one: weighted average. You've computed these since GPA." NO FORMULA YET. |
| 22 | md | **Problem 5 header**: "Package what we've been doing into a class." Formula `S = w_l*L + w_s*S + w_o*O` appears here — feels obvious, not introduced. |
| 23 | code | `SalienceResult` + `SalienceScorer` exercise |
| 24 | code | Verification |
| 25 | md | NOW name it: "This is a *salience scorer* — a weighted linear combination. Both are names for what you already built." + companion reading callout (ThinkStats Ch 1-2, ThinkBayes Ch 1, Blitzstein Ch 1-2). |
| 26 | md | **Problem 6 header**: validate against intuitive ratings |
| 27 | code | Score all, plot, Spearman correlation |
| 28 | md | Theory backfill: "Why a Weighted Linear Combination?" (unchanged — already in correct position) |
| 29 | md | Exercises 1-3 (unchanged) |
| 30 | code | Exercise 3 workspace |
| 31 | md | Outro (updated to mirror discovery order) |
| 32 | md | Resources (updated with ThinkBayes2, ThinkStats, Blitzstein references) |

**Key changes from current**:
- Remove cell-3 feedback cell (addressed by the rewrite)
- War story opening replaces bandit-presupposing intro
- Problem 2 (linguistic) comes BEFORE Problem 1 (dataset) — discovery order
- Formula appears after all 3 signals built, not in Layer 1
- Companion text references woven into post-build callouts

---

## Track 2: Curriculum & Syllabus Updates

### 2a. Update Arc 0 syllabus with companion texts

**File**: `syllabus/arcs/arc-0-probabilistic-foundations.md`

**Changes**:
- Add ThinkBayes2 as primary text for Modules 0.2-0.4 (Ch 1-5, 18)
- Add ThinkStats as primary text for Modules 0.2, 0.5-0.6 (Ch 1-4, 8-9)
- Keep Blitzstein & McElreath as secondary
- Update Resources section with chapter-level mappings

### 2b. Add companion mini-arc: LLM Internals

**New file**: `syllabus/arcs/arc-LLM-internals.md`

A companion arc (not numbered in the main sequence) synced against existing arcs on a "suggested accompaniment" basis. Structure:

```
Arc LLM: Language Model Internals (Companion Track)

Suggested pacing:
  - LLM.1-2 alongside Arc 1 (linear algebra gives you the tools for attention math)
  - LLM.3-4 alongside Arc 2 (Bayesian thinking for training evaluation)
  - LLM.5-6 after Arc 0.5-0.6 (use hypothesis testing / bootstrap for model comparison)

Modules:
  LLM.1: Tokenization & Text Processing (Raschka Ch 1-2)
  LLM.2: Attention from First Principles (Raschka Ch 3)
  LLM.3: Building GPT Architecture (Raschka Ch 4)
  LLM.4: Pretraining (Raschka Ch 5)
  LLM.5: Finetuning for Classification (Raschka Ch 6)
  LLM.6: Instruction Following & Alignment (Raschka Ch 7)
  LLM.7: LRMs from Scratch (extension — reasoning models)

Future extensions (lower priority):
  - Domain-Specific SLMs
  - Rearchitecting LLMs
```

### 2c. Update curriculum-brief.yaml

**File**: `syllabus/curriculum-brief.yaml`

- Add ThinkBayes2, ThinkStats, Raschka to learner gaps/resources
- Add `arc-LLM-internals.md` to arc_files list
- Update arc_count to 6

### 2d. Update sources/books.md

**File**: `sources/books.md`

- Add ThinkBayes2 with chapter-to-module mapping
- Add ThinkStats with chapter-to-module mapping
- Add Raschka with LLM arc mapping

---

## Track 3: Update Lesson-Generator Skill

**File**: `.claude/skills/lesson-generator/SKILL.md`

Add a new section after "Voice & Tone" called **Narrative Flow Principles** encoding the story-first patterns:

1. **Story-first opening**: Every notebook opens with a concrete problem/incident, not abstract theory. The reader should feel cognitive tension before any framework is introduced.

2. **Code-before-formula rule**: Mathematical notation appears ONLY AFTER working code demonstrates the concept. The sequence is: compute it → verify it → name it.

3. **Single-example-first**: Introduce concepts with ONE concrete example before building the full dataset. The reader should understand the problem on one case before seeing ten.

4. **Discovery order over logical order**: Problems can appear in the order the reader discovers concepts, not the order they'd appear in a textbook. (e.g., vocabulary matching before building the test suite, because you need to feel the problem before constructing systematic tests.)

5. **Formula emergence**: Weighted combinations, probability rules, etc. should feel *recognized*, not *introduced*. The reader has already been computing the thing; the formula just names it.

6. **Companion text callouts**: Reference companion texts (ThinkBayes2, ThinkStats, etc.) AFTER the reader has built something, as "go deeper" pointers. Never as prerequisites.

7. **Transition pattern**: Use RECAP → PROBLEM RESTATEMENT → NEW APPROACH between sections. Never jump to a new concept without bridging from the previous one.

Also update the "Narrative Bookends > Intro Section" template to show the war-story pattern alongside the existing hotel-startup example.

---

## Verification

1. **Notebook**: Open in JupyterLab, run all cells top-to-bottom. All assertions should pass. Narrative should read as a coherent story.
2. **Curriculum**: Check that ThinkBayes2/ThinkStats chapter references are accurate against actual chapter lists. Check that LLM companion arc sync suggestions make sense against prerequisite chains.
3. **Skill**: Generate a test prompt ("generate module 0.2 notebook") and verify the skill's narrative flow principles would produce story-first output.
