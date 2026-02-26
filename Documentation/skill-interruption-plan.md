# Skill Interruption & Priority — Next Iteration Plan

## Context

This document captures the agreed design for skill interruption handling. It exists so future implementation work starts from a shared understanding without re-explaining the reasoning.

### What is implemented today (the minimal stable base)

- **ask_user reply slot**: when a skill calls `context.ask_user(prompt)`, a Redis slot (`ella:skill:reply:{chat_id}`) is written. The next incoming message is routed directly as the answer — no BrainAgent turn fires. The skill polls and picks it up within 3 seconds.
- **Running-skill guard**: when a skill is `status=running` (mid-research, not waiting for input), any incoming message triggers a short text acknowledgement ("还在研究X，等我一下～") and returns immediately. No LLM loads, no objective update, no new planning.
- **Objective freeze**: the chat objective is not updated during a skill run (because the running-skill guard returns before Phase 1 / `summarise_recent_history` runs).

### What is NOT yet implemented (this document's scope)

---

## Design: Skill Priority

### The core idea

Skills carry a `priority` field. It determines how Ella behaves when interrupted mid-execution.

```python
# ella/skills/base.py — BaseSkill
priority: Literal["low", "high"] = "low"
```

| Priority | Example triggers | Interrupt behaviour |
|---|---|---|
| `low` | Casual "learn how to cook X" during chat | Pause, reply to user, resume after idle |
| `high` | "Implement this new skill for me", project work | Hold: brief "please don't interrupt" response, continue running |

### Low-priority interruption flow

```
User sends off-topic message while low-priority skill runs
        │
        ▼
BrainAgent running-skill guard detects status=running, priority=low
        │
        ▼
Skill execution is checkpointed and marked status=paused
        │
        ▼
BrainAgent runs a normal turn — responds to the user's message
Objective is updated from this new turn normally
        │
        ▼
[idle timer: no new messages for N minutes]
        │
        ▼
Resume scheduler fires → sends Ella an internal prompt:
  "You were researching '{goal}' — do you want to pick it up again?"
        │
        ▼
BrainAgent resumes the skill
```

### High-priority hold flow

```
User sends any message while high-priority skill runs
        │
        ▼
BrainAgent running-skill guard detects status=running, priority=high
        │
        ▼
Sends a firm (but warm) hold message — no full BrainAgent turn
  e.g. "I'm in the middle of something important, I'll come back to you when I'm done!"
Skill continues uninterrupted
```

---

## Implementation Tasks

### 1. Add `priority` to `BaseSkill`

File: `ella/skills/base.py`

```python
class BaseSkill(ABC):
    name: str
    description: str
    priority: Literal["low", "high"] = "low"
```

Update `LearnSkill` and `ResearchSkill` to declare `priority = "low"`.  
Any future project/code-generation skills should declare `priority = "high"`.

### 2. Expose priority in `SkillCheckpoint`

File: `ella/skills/base.py`, `ella/skills/checkpoint.py`

Add `priority: str = "low"` to `SkillCheckpoint`. Write it when the checkpoint is first created in `SkillExecutionRegistry.start()`.

MySQL: add `priority VARCHAR(10) DEFAULT 'low'` to `ella_skill_runs`. Migration must be rerunnable (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`).

### 3. Update BrainAgent running-skill guard

File: `ella/agents/brain_agent.py`

Replace the current single-path guard with priority-branched logic:

```python
if _active_cp.priority == "high":
    # Send firm hold message, return — skill keeps running
    ...
else:  # low priority
    # Checkpoint skill as paused, then fall through to normal BrainAgent turn
    await _skill_exec.pause(_active_cp.run_id)
    # ... normal turn continues, objective updates normally ...
```

`SkillExecutionRegistry` needs a `pause(run_id)` method that sets `status=paused` and writes a checkpoint.

### 4. Resume scheduler

File: `ella/skills/scheduler.py` (new, ~60 lines)

An `asyncio` background task started from `ella/main.py`. Every `SKILL_RESUME_CHECK_MINUTES` (default 5, configurable via `.env`):

1. Query `ella_skill_runs` for `status=paused` runs older than `SKILL_RESUME_IDLE_MINUTES` (default 10)
2. For each: check if the chat has been idle (no new `steps_done` since the pause)
3. If idle: send a nudge message to the chat — *"Hey, I still have some research on '{goal}' to finish — should I pick it up?"*
4. The nudge is stored as an `ask_user` reply slot so the next message is routed as a resume decision

The scheduler does **not** automatically resume without user confirmation — it only nudges.

Config additions to `.env` / `ella/config.py`:
```
SKILL_RESUME_CHECK_MINUTES=5
SKILL_RESUME_IDLE_MINUTES=10
```

### 5. Personality influence (optional / later)

The `priority` field is the structural hook. Personality influence on interrupt behaviour can be layered on top without changing the data model:

- Read `Personality.json` fields `dominance` and `resilience` at runtime
- If `dominance > 0.7` and `resilience > 0.7`: even low-priority skills push back once ("just let me finish this bit")
- This is a single conditional in the guard — no architecture change needed

---

## What does NOT change

- One `objective` field in `JobGoal` — not two. Objective freezes naturally during high-priority skill runs (guard returns early). For low-priority skills, objective updates normally during the pause period (the user is in a new conversation).
- The `ask_user` reply slot mechanism stays as-is. It handles the case where the skill explicitly needs input.
- `SkillCheckpoint.goal` remains the skill's research objective — it is fixed at creation and never updated by conversation.

---

## Files to touch

| File | Change |
|---|---|
| `ella/skills/base.py` | Add `priority` to `BaseSkill` and `SkillCheckpoint` |
| `ella/skills/builtin/learn.py` | `priority = "low"` |
| `ella/skills/builtin/research.py` | `priority = "low"` |
| `ella/skills/execution.py` | Pass `priority` into checkpoint on `start()`; add `pause(run_id)` method |
| `ella/skills/checkpoint.py` | Add `priority` field; store/load from Redis + MySQL |
| `ella/agents/brain_agent.py` | Update running-skill guard to branch on priority |
| `ella/skills/scheduler.py` | New file — idle-resume nudge loop |
| `ella/main.py` | Start scheduler as asyncio background task |
| `scripts/migrate_skills.sql` | `ALTER TABLE ella_skill_runs ADD COLUMN IF NOT EXISTS priority VARCHAR(10) DEFAULT 'low'` |
| `.env` / `ella/config.py` | `SKILL_RESUME_CHECK_MINUTES`, `SKILL_RESUME_IDLE_MINUTES` |
