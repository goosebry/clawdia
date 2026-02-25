"""Core dataclasses and abstract base class for the Ella skill system.

Tools are atomic and stateless. Skills are stateful multi-step workflows that
checkpoint their progress so execution can survive interruptions and be resumed.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from ella.agents.protocol import SessionContext
    from ella.skills.execution import SkillExecutionRegistry
    from ella.skills.registry import SkillRegistry
    from ella.tools.registry import ToolRegistry


@dataclass
class SkillCheckpoint:
    """Serialisable snapshot of a skill execution's progress.

    Persisted to Redis (fast working copy) and MySQL (permanent backup) after
    every meaningful phase so execution can be resumed after interruption.
    """
    run_id: str                  # UUID — also the MySQL ella_skill_runs.run_id
    skill_name: str
    chat_id: int
    goal: str
    phase: str                   # "research" | "read" | "analyse" | "accumulate" | "done"
    cycle: int                   # current research cycle (1-indexed)
    notes: list[str]             # accumulated knowledge passages
    questions: list[str]         # open questions from last Analyse phase
    artifacts: list[str]         # local file paths downloaded so far
    sources_done: list[str]      # URLs / file paths already processed (dedup on resume)
    status: Literal["running", "paused", "completed", "failed", "cancelled"]
    updated_at: str              # ISO 8601 UTC timestamp

    @classmethod
    def new(
        cls,
        skill_name: str,
        chat_id: int,
        goal: str,
    ) -> "SkillCheckpoint":
        return cls(
            run_id=str(uuid.uuid4()),
            skill_name=skill_name,
            chat_id=chat_id,
            goal=goal,
            phase="research",
            cycle=1,
            notes=[],
            questions=[],
            artifacts=[],
            sources_done=[],
            status="running",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SkillResult:
    """Outcome of a completed skill execution."""
    summary: str                 # synthesis of what was learned / done
    stored_points: int           # knowledge entries written to Qdrant
    artifacts: list[str]         # paths to downloaded files
    open_questions: list[str]    # questions that could not be resolved after max cycles


@dataclass
class SkillContext:
    """Shared mutable state passed through the full skill call chain.

    When a skill invokes a sub-skill via invoke_skill(), the same context is
    passed so all accumulated notes, artifacts, and questions flow into a
    single cohesive result.

    Each mutating operation should be followed by context.checkpoint(phase)
    to persist progress to Redis + MySQL.
    """
    chat_id: int
    run_id: str                              # ties to SkillCheckpoint.run_id
    session: "SessionContext"                # access to memory tiers
    tool_executor: "ToolRegistry"            # execute any registered tool
    skill_registry: "SkillRegistry"          # look up skill definitions
    execution_registry: "SkillExecutionRegistry"  # invoke sub-skills
    send_update: Callable[[str], Any]        # send interim Telegram message
    ask_user: Callable[[str], Any]           # pause + save checkpoint, await reply

    # Mutable state — carried across phases and cycles
    notes: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    sources_done: list[str] = field(default_factory=list)
    cycle: int = 1
    _active_skills: set[str] = field(default_factory=set)

    async def checkpoint(self, phase: str) -> None:
        """Persist current state to Redis + MySQL via SkillExecutionRegistry."""
        await self.execution_registry.save_checkpoint(
            run_id=self.run_id,
            phase=phase,
            cycle=self.cycle,
            notes=self.notes,
            questions=self.questions,
            artifacts=self.artifacts,
            sources_done=self.sources_done,
            status="running",
        )

    async def invoke_skill(self, name: str, goal: str) -> SkillResult:
        """Invoke a sub-skill, passing the shared context state.

        Guards against a skill directly invoking itself (circular invocation).
        Sub-skills share notes/questions/artifacts/sources_done with the parent.
        """
        if name in self._active_skills:
            raise RuntimeError(
                f"Circular skill invocation detected: '{name}' is already active. "
                f"Active skills: {self._active_skills}"
            )
        return await self.execution_registry.start_sub_skill(
            skill_name=name,
            goal=goal,
            parent_context=self,
        )


class BaseSkill(ABC):
    """Abstract base class for all Ella skills.

    Each concrete skill must define a clear name and description. The description
    is injected verbatim into the BrainAgent task planner prompt so the LLM can
    decide when to invoke this skill — make it unambiguous and specific.

    The run() method must call context.checkpoint(phase) after each meaningful
    phase so execution can be resumed if interrupted.
    """
    name: str
    description: str

    @abstractmethod
    async def run(self, goal: str, context: SkillContext) -> SkillResult:
        """Execute the skill. Must checkpoint after each phase.

        goal: natural-language description of what to accomplish
        context: shared mutable state — read notes/questions, call tools,
                 invoke sub-skills, checkpoint progress
        """
        ...
