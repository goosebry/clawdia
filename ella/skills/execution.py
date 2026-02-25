"""SkillExecutionRegistry — runtime lifecycle manager for skill executions.

This is the RUNTIME counterpart to SkillRegistry (the catalogue).
It knows what executions are currently running or paused, and manages
their full lifecycle: start → checkpoint → pause → resume → complete/cancel.

Separation of concerns:
  SkillRegistry          — "what skills exist?" (stateless, hot-reloadable)
  SkillExecutionRegistry — "what is running right now?" (stateful, owns checkpoints)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TYPE_CHECKING

from ella.skills.base import SkillCheckpoint, SkillContext, SkillResult

if TYPE_CHECKING:
    from ella.agents.protocol import SessionContext
    from ella.skills.checkpoint import SkillCheckpointStore
    from ella.skills.registry import SkillRegistry
    from ella.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SkillExecutionRegistry:
    """Runtime execution manager.

    Owns the SkillCheckpointStore and provides the interface BrainAgent uses
    to start or resume skill executions.
    """

    def __init__(
        self,
        skill_registry: "SkillRegistry",
        checkpoint_store: "SkillCheckpointStore",
    ) -> None:
        self._skill_registry = skill_registry
        self._checkpoint_store = checkpoint_store

    # ── Public API (called by BrainAgent) ────────────────────────────────────

    async def start(
        self,
        skill_name: str,
        goal: str,
        session: "SessionContext",
        tool_executor: "ToolRegistry",
        send_update: Callable[[str], Any],
        ask_user: Callable[[str], Any],
        on_run_id: Callable[[str], None] | None = None,
    ) -> SkillResult:
        """Create a new execution and run the skill from the beginning.

        on_run_id: optional callback invoked with the assigned run_id immediately
        after the checkpoint is created. Used by BrainAgent to wire the ask_user
        reply slot with the correct run_id before the skill coroutine executes.
        """
        skill_cls = self._skill_registry.get(skill_name)
        if skill_cls is None:
            raise ValueError(f"Unknown skill: '{skill_name}'")

        checkpoint = SkillCheckpoint.new(
            skill_name=skill_name,
            chat_id=session.chat_id,
            goal=goal,
        )
        await self._checkpoint_store.save(checkpoint)
        logger.info("[Execution] Starting skill '%s' run_id=%s goal=%r", skill_name, checkpoint.run_id, goal[:80])

        if on_run_id is not None:
            try:
                on_run_id(checkpoint.run_id)
            except Exception:
                pass

        context = self._build_context(
            checkpoint=checkpoint,
            session=session,
            tool_executor=tool_executor,
            send_update=send_update,
            ask_user=ask_user,
        )
        return await self._run(skill_cls(), context, checkpoint.run_id, goal)

    async def resume(
        self,
        run_id: str,
        session: "SessionContext",
        tool_executor: "ToolRegistry",
        send_update: Callable[[str], Any],
        ask_user: Callable[[str], Any],
    ) -> SkillResult:
        """Load a paused checkpoint and re-enter the skill at the saved phase."""
        checkpoint = await self._checkpoint_store.load(run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for run_id='{run_id}'")

        skill_cls = self._skill_registry.get(checkpoint.skill_name)
        if skill_cls is None:
            raise ValueError(f"Skill '{checkpoint.skill_name}' is no longer registered — cannot resume run_id='{run_id}'")

        logger.info(
            "[Execution] Resuming skill '%s' run_id=%s at phase='%s' cycle=%d notes=%d (was %s)",
            checkpoint.skill_name, run_id, checkpoint.phase, checkpoint.cycle,
            len(checkpoint.notes), checkpoint.status,
        )
        checkpoint.status = "running"
        await self._checkpoint_store.save(checkpoint)

        context = self._build_context(
            checkpoint=checkpoint,
            session=session,
            tool_executor=tool_executor,
            send_update=send_update,
            ask_user=ask_user,
        )
        return await self._run(skill_cls(), context, run_id, checkpoint.goal)

    async def cancel(self, run_id: str) -> None:
        """Cancel a running or paused execution."""
        await self._checkpoint_store.mark_cancelled(run_id)
        logger.info("[Execution] Cancelled run_id=%s", run_id)

    async def list_active(self, chat_id: int) -> list[SkillCheckpoint]:
        """Return all running or paused executions for a chat."""
        return await self._checkpoint_store.list_active(chat_id)

    async def list_all_paused(self) -> list[SkillCheckpoint]:
        """Return all paused executions across all chats (used at startup)."""
        return await self._checkpoint_store.list_paused()

    # ── Internal: sub-skill composition ──────────────────────────────────────

    async def start_sub_skill(
        self,
        skill_name: str,
        goal: str,
        parent_context: SkillContext,
    ) -> SkillResult:
        """Invoke a sub-skill sharing the parent context's state.

        The sub-skill inherits notes/questions/artifacts/sources_done from the
        parent so all collected knowledge flows into a single unified result.
        The parent's run_id is reused — sub-skill activity is part of the same
        execution record.
        """
        skill_cls = self._skill_registry.get(skill_name)
        if skill_cls is None:
            # Hot-reload may have temporarily unregistered builtin skills.
            # Attempt to re-load them from the builtin directory before giving up.
            logger.warning(
                "[Execution] Sub-skill '%s' not found — attempting re-registration from builtins",
                skill_name,
            )
            try:
                from pathlib import Path as _Path
                _builtins_dir = _Path(__file__).parent / "builtin"
                for _py in _builtins_dir.glob("*.py"):
                    if _py.name.startswith("_"):
                        continue
                    self._skill_registry._load_file(_py)
                skill_cls = self._skill_registry.get(skill_name)
            except Exception as _re_exc:
                logger.warning("[Execution] Re-registration attempt failed: %s", _re_exc)
        if skill_cls is None:
            raise ValueError(f"Unknown sub-skill: '{skill_name}'")

        logger.info(
            "[Execution] ── SUB-SKILL '%s' run_id=%s goal=%r notes_before=%d sources_done=%d",
            skill_name, parent_context.run_id, goal[:60],
            len(parent_context.notes), len(parent_context.sources_done),
        )

        sub_context = SkillContext(
            chat_id=parent_context.chat_id,
            run_id=parent_context.run_id,
            session=parent_context.session,
            tool_executor=parent_context.tool_executor,
            skill_registry=parent_context.skill_registry,
            execution_registry=self,
            send_update=parent_context.send_update,
            ask_user=parent_context.ask_user,
            notes=parent_context.notes,
            questions=parent_context.questions,
            artifacts=parent_context.artifacts,
            sources_done=parent_context.sources_done,
            cycle=parent_context.cycle,
            _active_skills=parent_context._active_skills | {skill_name},
        )
        skill_instance = skill_cls()
        # Sub-skills share the parent run_id (same execution record).
        # We call run() directly here — sub-skills don't manage their own lifecycle.
        result = await skill_instance.run(goal, sub_context)
        logger.info(
            "[Execution] ── SUB-SKILL '%s' run_id=%s done | notes_after=%d artifacts=%d",
            skill_name, parent_context.run_id,
            len(sub_context.notes), len(sub_context.artifacts),
        )
        return result

    # ── Checkpoint bridge (called by SkillContext.checkpoint()) ──────────────

    async def save_checkpoint(
        self,
        run_id: str,
        phase: str,
        cycle: int,
        notes: list[str],
        questions: list[str],
        artifacts: list[str],
        sources_done: list[str],
        status: str,
    ) -> None:
        cp = await self._checkpoint_store.load(run_id)
        if cp is None:
            logger.warning("[Execution] save_checkpoint: no checkpoint found for run_id=%s", run_id)
            return
        cp.phase = phase
        cp.cycle = cycle
        cp.notes = notes
        cp.questions = questions
        cp.artifacts = artifacts
        cp.sources_done = sources_done
        cp.status = status  # type: ignore[assignment]
        await self._checkpoint_store.save(cp)
        logger.debug(
            "[Execution] Checkpoint saved run_id=%s phase=%s cycle=%d notes=%d sources_done=%d status=%s",
            run_id, phase, cycle, len(notes), len(sources_done), status,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_context(
        self,
        checkpoint: SkillCheckpoint,
        session: "SessionContext",
        tool_executor: "ToolRegistry",
        send_update: Callable,
        ask_user: Callable,
    ) -> SkillContext:
        return SkillContext(
            chat_id=checkpoint.chat_id,
            run_id=checkpoint.run_id,
            session=session,
            tool_executor=tool_executor,
            skill_registry=self._skill_registry,
            execution_registry=self,
            send_update=send_update,
            ask_user=ask_user,
            notes=list(checkpoint.notes),
            questions=list(checkpoint.questions),
            artifacts=list(checkpoint.artifacts),
            sources_done=list(checkpoint.sources_done),
            cycle=checkpoint.cycle,
            _active_skills={checkpoint.skill_name},
        )

    async def _run(
        self,
        skill_instance: Any,
        context: SkillContext,
        run_id: str,
        goal: str,
    ) -> SkillResult:
        try:
            result = await skill_instance.run(goal, context)
            await self._checkpoint_store.mark_completed(run_id)
            if result.open_questions:
                await self._checkpoint_store.save_open_questions(run_id, result.open_questions)
            if result.summary:
                await self._checkpoint_store.update_summary(run_id, result.summary, result.stored_points)
            logger.info(
                "[Execution] Skill '%s' completed run_id=%s stored=%d open_q=%d",
                skill_instance.name, run_id, result.stored_points, len(result.open_questions),
            )
            return result
        except asyncio.CancelledError:
            # Shutdown mid-skill — mark as failed so it can be auto-resumed on
            # next startup. Do NOT mark cancelled (which blocks resume).
            await self._checkpoint_store.mark_failed(run_id)
            logger.info("[Execution] Skill '%s' run_id=%s interrupted by shutdown — marked failed for resume", skill_instance.name, run_id)
            raise
        except Exception:
            logger.exception("[Execution] Skill '%s' failed run_id=%s", skill_instance.name, run_id)
            await self._checkpoint_store.mark_failed(run_id)
            return SkillResult(
                summary="Skill execution failed due to an unexpected error.",
                stored_points=0,
                artifacts=context.artifacts,
                open_questions=context.questions,
            )


_execution_registry: SkillExecutionRegistry | None = None


async def get_execution_registry() -> SkillExecutionRegistry:
    """Return the shared SkillExecutionRegistry, creating it lazily on first call."""
    global _execution_registry
    if _execution_registry is None:
        from ella.skills.registry import get_skill_registry
        from ella.skills.checkpoint import get_checkpoint_store
        _execution_registry = SkillExecutionRegistry(
            skill_registry=get_skill_registry(),
            checkpoint_store=await get_checkpoint_store(),
        )
    return _execution_registry
