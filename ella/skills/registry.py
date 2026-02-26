"""SkillRegistry — catalogue of available skill definitions.

This is the LIBRARY of what skills Ella can perform. It is stateless — it only
knows what skills exist, not what executions are in flight.

Usage:
    @ella_skill(name="learn", description="Learn about a topic...")
    class LearnSkill(BaseSkill):
        ...

The registry supports hot-reload: drop a .py file with @ella_skill decorated
classes into ella/skills/custom/ and the skill becomes available on the next
message — no restart needed.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from watchfiles import awatch

from ella.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Stateless catalogue of skill definitions.

    Knows *what skills exist*. Does not track executions.
    """

    def __init__(self) -> None:
        self._skills: dict[str, type[BaseSkill]] = {}
        self._lock = asyncio.Lock()
        self._module_skills: dict[str, list[str]] = {}

    def register(self, skill_cls: type[BaseSkill]) -> None:
        """Register a skill class synchronously (called by @ella_skill at import time)."""
        name = getattr(skill_cls, "name", None)
        if not name:
            raise ValueError(f"Skill class {skill_cls} must define a 'name' class attribute.")
        self._skills[name] = skill_cls
        mod = getattr(skill_cls, "__module__", "")
        self._module_skills.setdefault(mod, []).append(name)
        logger.debug("Registered skill: %s", name)

    def get(self, name: str) -> type[BaseSkill] | None:
        """Return the skill class for the given name, or None if not found."""
        return self._skills.get(name)

    def get_skills_schema(self) -> dict[str, str]:
        """Return {name: description} for all registered skills.

        Injected verbatim into BrainAgent task planner prompt so the LLM can
        decide when to invoke a skill. Descriptions must be unambiguous.
        """
        return {
            name: getattr(cls, "description", "")
            for name, cls in self._skills.items()
        }

    def all_names(self) -> list[str]:
        return list(self._skills.keys())

    def _unregister_module(self, module_name: str) -> None:
        names = self._module_skills.pop(module_name, [])
        for n in names:
            self._skills.pop(n, None)
        if names:
            logger.info("Unregistered %d skill(s) from module %s", len(names), module_name)

    @staticmethod
    def _module_name_for(path: Path) -> str:
        """Derive a stable module name from the file path.

        Builtin skills live inside the ella package and get a dotted package
        name (e.g. ella.skills.builtin.learn) so importlib.reload works
        correctly — the parent package already exists in sys.modules.

        Custom skills live outside the package and get a flat namespace
        (ella_skill_custom.<stem>) with the parent package bootstrapped on
        first use.
        """
        try:
            # Resolve to an absolute path and check if it's inside ella.skills.builtin
            abs_path = path.resolve()
            # Find the ella package root by locating ella/__init__.py upward
            for parent in abs_path.parents:
                ella_init = parent / "ella" / "__init__.py"
                if ella_init.exists():
                    rel = abs_path.relative_to(parent)
                    # Convert path segments to dotted module name, strip .py
                    parts = list(rel.parts)
                    parts[-1] = parts[-1].removesuffix(".py")
                    return ".".join(parts)
        except (ValueError, RuntimeError):
            pass
        # Fallback: custom skill outside the package tree
        return f"ella_skill_custom.{path.stem}"

    def _ensure_parent_package(self, module_name: str) -> None:
        """Make sure all parent packages exist in sys.modules so reload works."""
        parts = module_name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                # Bootstrap a minimal parent package
                pkg = importlib.util.module_from_spec(
                    importlib.util.spec_from_loader(parent, loader=None)  # type: ignore[arg-type]
                )
                pkg.__path__ = []  # type: ignore[attr-defined]
                pkg.__package__ = parent
                sys.modules[parent] = pkg

    def _load_file(self, path: Path) -> None:
        module_name = self._module_name_for(path)
        self._ensure_parent_package(module_name)

        if module_name in sys.modules:
            self._unregister_module(module_name)
            module = sys.modules[module_name]
            try:
                importlib.reload(module)
                logger.info("Reloaded skill module: %s (%s)", path.name, module_name)
                return
            except ModuleNotFoundError:
                # reload() requires __spec__ to be set — modules loaded via
                # spec_from_file_location may not have it. Fall through to
                # re-exec via a fresh spec.
                logger.debug("reload() missing spec for %s — re-executing from file", module_name)
                sys.modules.pop(module_name, None)
            except Exception:
                logger.exception("Failed to reload skill module %s", path)
                return
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            module.__spec__ = spec  # ensure reload() can find the spec next time
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                logger.info("Loaded skill module: %s (%s)", path.name, module_name)
            except Exception:
                logger.exception("Failed to load skill module %s", path)
                sys.modules.pop(module_name, None)

    def load_directory(self, directory: str | Path) -> None:
        """Scan a directory and import all .py files (except __init__.py and _*)."""
        d = Path(directory)
        if not d.exists():
            return
        for path in sorted(d.glob("*.py")):
            if path.name.startswith("_"):
                continue
            self._load_file(path)

    async def watch(self, *directories: str | Path) -> None:
        """Background asyncio task: watch directories for skill file changes."""
        watch_paths = [str(Path(d).resolve()) for d in directories]
        logger.info("SkillRegistry watching: %s", watch_paths)
        async for changes in awatch(*watch_paths):
            for change_type, path_str in changes:
                path = Path(path_str)
                if path.suffix != ".py" or path.name.startswith("_"):
                    continue
                change_name = change_type.name if hasattr(change_type, "name") else str(change_type)
                logger.info("Skill file %s: %s", change_name, path.name)
                if change_name in ("added", "modified"):
                    async with self._lock:
                        self._load_file(path)
                elif change_name == "deleted":
                    module_name = self._module_name_for(path)
                    async with self._lock:
                        self._unregister_module(module_name)
                        sys.modules.pop(module_name, None)


_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def ella_skill(name: str, description: str) -> Callable:
    """Class decorator that registers a BaseSkill subclass in the SkillRegistry.

    The description is injected into BrainAgent prompts — make it specific and
    unambiguous so the LLM knows exactly when to use this skill.

    Usage:
        @ella_skill(
            name="learn",
            description="Deeply research a topic using web search, PDF reading, "
                        "and social media, then store the findings as knowledge.",
        )
        class LearnSkill(BaseSkill):
            name = "learn"
            description = "..."
    """
    def decorator(cls: type[BaseSkill]) -> type[BaseSkill]:
        cls.name = name
        cls.description = description
        get_skill_registry().register(cls)
        return cls

    return decorator
