"""Tier 2 — The Goal: conversation-scoped shared memory backed by Redis.

Persists across the entire conversation thread for a chat_id.  It answers
"why are we here?" and accumulates:
  - objective: the evolving statement of the user's current intent (only updated
    when the LLM detects a topic shift)
  - steps_done: compact per-turn summaries (conversation history)
  - tool_focuses: per-tool-per-turn scratchpads (what each tool did and found)
  - shared_notes: free-form key/value notes (knowledge snippets, etc.)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import redis.asyncio as aioredis

from ella.config import get_settings


@dataclass
class ToolFocus:
    """Isolated scratchpad for a single tool call within one conversation turn.

    Each tool invocation gets its own context — the LLM reasons about the tool
    result in isolation before the result is surfaced to the final reply.
    """
    turn_index: int       # which conversation turn this belongs to
    tool_name: str
    tool_args: dict       # what we asked the tool
    tool_result: str      # what the tool returned (truncated for storage)
    reasoning: str        # LLM's one-sentence interpretation of the result
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class StepSummary:
    step_index: int
    agent: str
    summary: str
    # Raw verbatim text stored alongside the compressed summary stub.
    # Used by summarise_recent_history to give the objective LLM the full
    # picture of recent exchanges — without inflating the phrase-repetition
    # prevention stubs that live in `summary`.
    raw_user_text: str = ""
    raw_ella_text: str = ""
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class JobGoal:
    job_id: str
    chat_id: int
    objective: str
    steps_total: int = 0
    steps_done: list[StepSummary] = field(default_factory=list)
    tool_focuses: list[ToolFocus] = field(default_factory=list)
    shared_notes: dict[str, Any] = field(default_factory=dict)
    partial_outputs: list[str] = field(default_factory=list)
    status: Literal["running", "complete", "failed"] = "running"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def new(cls, chat_id: int, objective: str) -> "JobGoal":
        return cls(job_id=str(uuid.uuid4()), chat_id=chat_id, objective=objective)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "JobGoal":
        d = json.loads(data)
        d["steps_done"] = [StepSummary(**s) for s in d.get("steps_done", [])]
        d["tool_focuses"] = [ToolFocus(**tf) for tf in d.get("tool_focuses", [])]
        return cls(**d)


class GoalStore:
    """Redis-backed store for JobGoal objects."""

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._ttl = get_settings().goal_ttl_seconds

    def _key(self, job_id: str) -> str:
        return f"ella:goal:{job_id}"

    async def create(self, goal: JobGoal) -> None:
        await self._redis.set(self._key(goal.job_id), goal.to_json(), ex=self._ttl)

    async def read(self, job_id: str) -> JobGoal | None:
        raw = await self._redis.get(self._key(job_id))
        if raw is None:
            return None
        return JobGoal.from_json(raw)

    async def append_step(self, job_id: str, summary: StepSummary) -> None:
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.steps_done.append(summary)
        await self._redis.set(self._key(job_id), goal.to_json(), ex=self._ttl)

    async def append_tool_focus(self, job_id: str, tf: "ToolFocus") -> None:
        """Record a completed tool call's isolated scratchpad into the Goal."""
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.tool_focuses.append(tf)
        goal.tool_focuses = goal.tool_focuses[-20:]
        await self._redis.set(self._key(goal.job_id), goal.to_json(), ex=self._ttl)

    async def update_objective(self, job_id: str, new_objective: str) -> None:
        """Replace the goal's objective — called only on a confirmed topic shift."""
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.objective = new_objective[:500]
        await self._redis.set(self._key(goal.job_id), goal.to_json(), ex=self._ttl)

    async def update_notes(self, job_id: str, notes: dict[str, Any]) -> None:
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.shared_notes.update(notes)
        await self._redis.set(self._key(job_id), goal.to_json(), ex=self._ttl)

    async def add_output(self, job_id: str, output: str) -> None:
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.partial_outputs.append(output)
        await self._redis.set(self._key(job_id), goal.to_json(), ex=self._ttl)

    async def complete(self, job_id: str) -> None:
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.status = "complete"
        await self._redis.set(self._key(job_id), goal.to_json(), ex=self._ttl)

    async def fail(self, job_id: str, reason: str = "") -> None:
        goal = await self.read(job_id)
        if goal is None:
            return
        goal.status = "failed"
        if reason:
            goal.shared_notes["failure_reason"] = reason
        await self._redis.set(self._key(job_id), goal.to_json(), ex=self._ttl)

    async def delete(self, job_id: str) -> None:
        await self._redis.delete(self._key(job_id))

    # ── Chat → Goal index (survives restart) ─────────────────────────────────
    # Stores the most recent job_id for each chat_id so we can restore the
    # goal after a process restart (when in-memory sessions are wiped).

    def _chat_index_key(self, chat_id: int) -> str:
        return f"ella:chat:goal:{chat_id}"

    async def bind_chat(self, chat_id: int, job_id: str) -> None:
        """Record that chat_id is currently working on job_id (7-day TTL)."""
        await self._redis.set(self._chat_index_key(chat_id), job_id, ex=7 * 24 * 3600)

    async def find_goal_for_chat(self, chat_id: int) -> "JobGoal | None":
        """Return the most recent goal for chat_id, or None if not found."""
        job_id = await self._redis.get(self._chat_index_key(chat_id))
        if not job_id:
            return None
        return await self.read(job_id)


_goal_store: GoalStore | None = None


def get_goal_store() -> GoalStore:
    """Return the shared GoalStore, creating it lazily on first call.

    Lazy creation ensures the underlying redis.asyncio client is bound to the
    event loop that is active at call time. This is critical in Celery forked
    workers: the parent-process loop no longer exists after a fork, so any
    client created before the fork must be discarded. celery_app.py resets
    _goal_store = None via the worker_process_init signal so that this
    function creates a fresh client on the new loop.
    """
    global _goal_store
    if _goal_store is None:
        settings = get_settings()
        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        _goal_store = GoalStore(client)
    return _goal_store
