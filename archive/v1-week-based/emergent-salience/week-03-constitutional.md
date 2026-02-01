# Week 3: Constitutional Rules (Safe RL)

**Goal**: Design constraint systems that impose structure without killing emergence.

**Time**: 3-4 sessions × 30 min = 1.5-2 hours

**Milestone**: Working `constitutional.yaml` schema + enforcement logic.

---

## Overview

| Day | Notebook | Time | Topic |
|-----|----------|------|-------|
| 3.1 | `03a-axioms-vs-learned.ipynb` | 30 min | What can't be learned? |
| 3.2 | `03b-yaml-schema.ipynb` | 30 min | Designing the constitutional format |
| 3.3 | `03c-enforcement.ipynb` | 30 min | Block, warn, notify semantics |
| 3.4 | `03d-escape-hatches.ipynb` | 30 min | [EXTEND] Preaxiomatic overrides |

---

## Day 3.1: Axioms vs Learned Rules

### Learning Objectives
- [ ] Distinguish constitutional (imposed) from emergent (learned) rules
- [ ] Identify what can't be left to emergence
- [ ] Design the CONSTITUTIONAL tier

### Linguistics Anchor: Phonotactics

> English phonotactics forbids certain consonant clusters. You can't start a word with /tl/ or /sr/. These aren't learned through feedback — they're structural constraints of the system.
>
> Try pronouncing "tlop" or "srink." You can't, not because you've never heard them, but because the phonological grammar forbids them.
>
> Constitutional rules are phonotactics for your codebase.

### What Can't Be Learned?

Some rules have catastrophic failure modes:

| Rule | Why Constitutional? |
|------|-------------------|
| "Never commit secrets" | One failure = irreversible damage |
| "Never skip tests" | Hard to detect violations post-hoc |
| "No force-push to main" | One violation = team chaos |

These can't wait for emergence. They must be imposed.

### The Two-Tier System

```
CONSTITUTIONAL (imposed)
    ↓ always checked first
    ↓ blocks execution on violation

EMERGENT (learned)
    ↓ Thompson Sampling selects
    ↓ feedback updates beliefs
    ↓ tiers shift naturally
```

---

## Day 3.2: YAML Schema Design

### Learning Objectives
- [ ] Design the `constitutional.yaml` format
- [ ] Support categories, enforcement levels, metadata
- [ ] Parse and validate the schema

### The Schema

```yaml
# .buildlog/constitutional.yaml
version: 1

rules:
  security:
    - text: "Never commit secrets or credentials"
      enforcement: block
      rationale: "Secrets in git history are irrecoverable"

    - text: "Never disable security checks without review"
      enforcement: warn
      rationale: "Security debt accumulates silently"

  testing:
    - text: "Never commit code without tests"
      enforcement: block
      rationale: "Untested code is unknown code"

  workflow:
    - text: "Never force-push to main or master"
      enforcement: block
      rationale: "Protects shared history"

preaxiomatic:
  - "arch-abc123"  # Known architecture decision override
  - "hotfix-2024-01-15"  # Emergency escape hatch

metadata:
  last_updated: "2024-01-15"
  author: "team"
  review_cycle: "quarterly"
```

### Implementation

```python
@dataclass
class ConstitutionalRule:
    text: str
    enforcement: Enforcement  # BLOCK | WARN | NOTIFY
    category: str
    rationale: str | None = None

@dataclass
class Constitution:
    rules: list[ConstitutionalRule]
    preaxiomatic: set[str]  # Overrides
    version: int

    @classmethod
    def from_yaml(cls, path: Path) -> "Constitution":
        """Parse and validate constitutional.yaml."""
        # TODO: Implement
```

---

## Day 3.3: Enforcement Semantics

### Learning Objectives
- [ ] Implement block, warn, notify behaviors
- [ ] Integrate enforcement into rule selection
- [ ] Handle conflicts between constitutional and emergent rules

### Enforcement Levels

| Level | Behavior | Use Case |
|-------|----------|----------|
| **BLOCK** | Halt execution, require override | Security, data integrity |
| **WARN** | Continue but log prominently | Best practices |
| **NOTIFY** | Silent log, surface in reports | Monitoring |

### Implementation

```python
class Enforcement(Enum):
    BLOCK = "block"
    WARN = "warn"
    NOTIFY = "notify"

def check_constitutional(
    action: Action,
    constitution: Constitution,
    context: Context
) -> EnforcementResult:
    """
    Check action against constitutional rules.
    """
    for rule in constitution.rules:
        if violates(action, rule, context):
            if has_override(context, constitution.preaxiomatic):
                return EnforcementResult(
                    status="overridden",
                    rule=rule,
                    override_id=get_override_id(context)
                )
            return EnforcementResult(
                status=rule.enforcement.value,
                rule=rule,
                message=f"Constitutional violation: {rule.text}"
            )
    return EnforcementResult(status="pass")
```

### The Check Order

```
1. Check constitutional rules (BLOCK > WARN > NOTIFY)
2. If blocked and no override → halt
3. If passed → proceed to Thompson Sampling for emergent rules
```

---

## Day 3.4: Preaxiomatic Overrides [EXTEND]

### Learning Objectives
- [ ] Design the override system
- [ ] Balance flexibility with accountability
- [ ] Implement audit logging

### Why Overrides?

Sometimes you need to violate a constitutional rule:
- Emergency hotfix at 3am
- Architecture decision that supersedes old rule
- Migration period where rule is temporarily suspended

### The Preaxiomatic System

```yaml
preaxiomatic:
  - id: "arch-abc123"
    granted_by: "lead-architect"
    expires: "2024-03-01"
    scope: ["security/rate-limiting"]
    rationale: "Migrating to new rate limiter, old checks temporarily disabled"
```

### Implementation

```python
def has_override(context: Context, preaxiomatic: set[str]) -> bool:
    """Check if context includes a valid override."""
    return any(
        override in preaxiomatic
        for override in context.declared_overrides
    )

def audit_override(
    action: Action,
    rule: ConstitutionalRule,
    override_id: str
) -> None:
    """Log override usage for audit trail."""
    log.warning(
        f"Constitutional override: {rule.text}",
        override_id=override_id,
        action=action,
        timestamp=datetime.now()
    )
```

### Key Insight

Overrides are escape hatches, not loopholes. They must be:
- Explicitly declared
- Logged immutably
- Reviewed periodically
- Time-bounded when possible

---

## Week 3 Milestone: `constitutional.yaml` + Enforcement

Deliverable:

```python
# You've implemented:
class Constitution: ...
class ConstitutionalRule: ...
class Enforcement(Enum): ...
def check_constitutional(action, constitution, context) -> EnforcementResult: ...
def has_override(context, preaxiomatic) -> bool: ...
```

Plus a working `constitutional.yaml` for buildlog-template.

---

## Safe RL Connection

This is **Safe Reinforcement Learning** in action:

- **Constraint**: Constitutional rules define hard boundaries
- **Learning**: Emergent rules learn within those boundaries
- **Safety**: The agent can't learn to violate the constitution

The formal framework: Constrained MDPs (CMDPs), but we don't need the full theory. The intuition is enough: learn freely within guardrails.

---

## Reflection Questions

1. What makes a rule "constitutional" vs "should just learn it"?
2. How do you handle a constitutional rule that becomes obsolete?
3. What's the risk of too many overrides?

---

## [OPTIONAL DEPTH] Constrained MDPs

The formal theory of safe RL uses Constrained Markov Decision Processes. The agent maximizes reward subject to constraint satisfaction.

Lagrangian relaxation converts constraints to penalty terms. Primal-dual methods optimize both objectives.

For our use case, hard constraints (block) are simpler than soft penalties.

**Time estimate**: 30-40 min rabbit hole. Skip freely.

---

→ [Week 4: Learning Dynamics](week-04-dynamics.md)
