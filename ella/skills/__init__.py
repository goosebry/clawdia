"""Ella Skill System.

A skill is a named, stateful, multi-step capability that orchestrates tools
and other skills to produce a durable outcome stored in long-term memory.

Unlike tools (atomic, stateless, fire-and-forget), skills can be paused,
interrupted, and resumed — surviving process crashes or machine reboots via
Redis + MySQL dual-write checkpointing.

Key components:
  SkillRegistry          — catalogue of available skills (@ella_skill decorator)
  SkillExecutionRegistry — runtime lifecycle manager (start/resume/cancel)
  SkillCheckpointStore   — Redis + MySQL dual-write persistence
  BaseSkill              — abstract base class all skills must implement
  SkillContext           — shared state passed through the skill call chain
"""
from ella.skills.base import BaseSkill, SkillCheckpoint, SkillContext, SkillResult
from ella.skills.registry import SkillRegistry, ella_skill, get_skill_registry
from ella.skills.checkpoint import SkillCheckpointStore, get_checkpoint_store
from ella.skills.execution import SkillExecutionRegistry, get_execution_registry

__all__ = [
    "BaseSkill",
    "SkillCheckpoint",
    "SkillContext",
    "SkillResult",
    "SkillRegistry",
    "ella_skill",
    "get_skill_registry",
    "SkillCheckpointStore",
    "get_checkpoint_store",
    "SkillExecutionRegistry",
    "get_execution_registry",
]
