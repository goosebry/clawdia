"""Dynamic hot-reload tool registry.

Drop any .py file with @ella_tool decorated functions into ella/tools/custom/
and the tool becomes available on the next message batch — no restart needed.

The registry uses watchfiles.awatch() as a background asyncio task to detect
file changes and reload modules atomically.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from watchfiles import awatch

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable
    module_path: str


def _build_json_schema(fn: Callable, name: str, description: str) -> dict[str, Any]:
    """Auto-generate a JSON schema from a function's type hints and docstring."""
    sig = inspect.signature(fn)
    hints = {}
    try:
        hints = {
            k: v
            for k, v in fn.__annotations__.items()
            if k != "return"
        }
    except Exception:
        pass

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        py_type = hints.get(param_name, str)
        json_type = type_map.get(py_type, "string")
        doc = ""
        if fn.__doc__:
            for line in fn.__doc__.splitlines():
                line = line.strip()
                if line.startswith(f"{param_name}:"):
                    doc = line[len(param_name) + 1:].strip()
                    break
        properties[param_name] = {"type": json_type, "description": doc}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._lock = asyncio.Lock()
        self._module_tools: dict[str, list[str]] = {}

    def _make_decorator(self, name: str, description: str) -> Callable:
        registry = self

        def decorator(fn: Callable) -> Callable:
            schema = _build_json_schema(fn, name, description)
            entry = ToolEntry(
                name=name,
                description=description,
                parameters=schema["function"]["parameters"],
                fn=fn,
                module_path=getattr(fn, "__module__", ""),
            )
            asyncio.get_event_loop().run_until_complete(
                registry._register_sync(entry)
            ) if asyncio.get_event_loop().is_running() else None
            # fallback for sync context (import-time registration)
            if name not in registry._tools:
                registry._tools[name] = entry
                mod = entry.module_path
                registry._module_tools.setdefault(mod, []).append(name)
            return fn

        return decorator

    async def _register_sync(self, entry: ToolEntry) -> None:
        async with self._lock:
            self._tools[entry.name] = entry
            mod = entry.module_path
            if mod not in self._module_tools:
                self._module_tools[mod] = []
            if entry.name not in self._module_tools[mod]:
                self._module_tools[mod].append(entry.name)

    def register(self, fn: Callable, name: str, description: str) -> None:
        """Register a tool synchronously (called by @ella_tool at import time)."""
        schema = _build_json_schema(fn, name, description)
        entry = ToolEntry(
            name=name,
            description=description,
            parameters=schema["function"]["parameters"],
            fn=fn,
            module_path=getattr(inspect.getmodule(fn), "__name__", ""),
        )
        self._tools[name] = entry
        mod = entry.module_path
        self._module_tools.setdefault(mod, []).append(name)
        logger.debug("Registered tool: %s", name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return a live snapshot of all tool schemas for LLM injection."""
        schemas = []
        for entry in self._tools.values():
            schema = _build_json_schema(entry.fn, entry.name, entry.description)
            schemas.append(schema)
        return schemas

    async def execute(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch a tool call by name with args dict.

        Any keys in *args* that are not valid parameters of the tool function
        are silently dropped before the call.  This guards against LLMs
        hallucinating extra arguments (e.g. injecting internal memory fields
        like 'tool_focuses' into tool calls after seeing them in the prompt).
        """
        async with self._lock:
            entry = self._tools.get(name)
        if entry is None:
            return f"Error: tool '{name}' not found."

        # Filter to only the parameters the function actually accepts
        sig = inspect.signature(entry.fn)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered_args = {k: v for k, v in args.items() if k in valid_params}
        if len(filtered_args) != len(args):
            dropped = set(args) - set(filtered_args)
            logger.warning(
                "Tool '%s': dropped unexpected arg(s) %s (LLM hallucination)",
                name, dropped,
            )

        # Coerce argument types to match the function signature — LLMs frequently
        # pass integers as strings (e.g. max_results="5" instead of 5).
        type_map = {int: int, float: float, bool: lambda v: str(v).lower() not in ("false", "0", "")}
        coerced_args: dict[str, Any] = {}
        for k, v in filtered_args.items():
            param = sig.parameters.get(k)
            if param and param.annotation in type_map:
                try:
                    coerced_args[k] = type_map[param.annotation](v)
                except (TypeError, ValueError):
                    logger.warning("Tool '%s': could not coerce arg %s=%r to %s", name, k, v, param.annotation)
                    coerced_args[k] = v
            else:
                coerced_args[k] = v
        filtered_args = coerced_args

        import time as _time
        _t0 = _time.monotonic()
        try:
            result = entry.fn(**filtered_args)
            if asyncio.iscoroutine(result):
                result = await result
            elapsed = _time.monotonic() - _t0
            result_str = str(result)
            logger.info(
                "[Tool] %s(%s) → %d chars in %.2fs | %s",
                name,
                ", ".join(f"{k}={v!r}" for k, v in filtered_args.items()),
                len(result_str),
                elapsed,
                result_str[:120].replace("\n", " "),
            )
            return result
        except Exception as exc:
            elapsed = _time.monotonic() - _t0
            logger.exception("[Tool] %s raised an error after %.2fs", name, elapsed)
            return f"Error executing tool '{name}': {exc}"

    def _unregister_module(self, module_name: str) -> None:
        tool_names = self._module_tools.pop(module_name, [])
        for n in tool_names:
            self._tools.pop(n, None)
        if tool_names:
            logger.info("Unregistered %d tool(s) from module %s", len(tool_names), module_name)

    @staticmethod
    def _module_name_for(path: Path) -> str:
        """Derive a dotted module name from the file path.

        For files inside the ella package (e.g. ella/tools/builtin/web_search.py)
        returns the proper dotted name (ella.tools.builtin.web_search).
        For files outside the package (custom tools dropped anywhere) falls back
        to ella_tool_custom.<stem>.
        """
        try:
            abs_path = path.resolve()
            for parent in abs_path.parents:
                ella_init = parent / "ella" / "__init__.py"
                if ella_init.exists():
                    rel = abs_path.relative_to(parent)
                    parts = list(rel.parts)
                    parts[-1] = parts[-1].removesuffix(".py")
                    return ".".join(parts)
        except (ValueError, RuntimeError):
            pass
        return f"ella_tool_custom.{path.stem}"

    @staticmethod
    def _ensure_parent_package(module_name: str) -> None:
        """Bootstrap any missing intermediate packages in sys.modules."""
        parts = module_name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
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
                logger.info("Reloaded tool module: %s (%s)", path.name, module_name)
                return
            except ModuleNotFoundError:
                # reload() requires __spec__ — re-exec from file instead.
                logger.debug("reload() missing spec for %s — re-executing from file", module_name)
                sys.modules.pop(module_name, None)
            except Exception:
                logger.exception("Failed to reload tool module %s", path)
                return
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            module.__spec__ = spec  # ensure reload() can find the spec next time
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                logger.info("Loaded tool module: %s (%s)", path.name, module_name)
            except Exception:
                logger.exception("Failed to load tool module %s", path)
                sys.modules.pop(module_name, None)

    def load_directory(self, directory: str | Path) -> None:
        """Scan a directory and import all .py files (except __init__.py)."""
        d = Path(directory)
        if not d.exists():
            return
        for path in sorted(d.glob("*.py")):
            if path.name.startswith("_"):
                continue
            self._load_file(path)

    async def watch(self, *directories: str | Path) -> None:
        """Background asyncio task: watch directories for file changes."""
        watch_paths = [str(Path(d).resolve()) for d in directories]
        logger.info("ToolRegistry watching: %s", watch_paths)
        async for changes in awatch(*watch_paths):
            for change_type, path_str in changes:
                path = Path(path_str)
                if not path.suffix == ".py" or path.name.startswith("_"):
                    continue
                change_name = change_type.name if hasattr(change_type, "name") else str(change_type)
                logger.info("Tool file %s: %s", change_name, path.name)
                if change_name in ("added", "modified"):
                    async with self._lock:
                        self._load_file(path)
                elif change_name == "deleted":
                    module_name = self._module_name_for(path)
                    async with self._lock:
                        self._unregister_module(module_name)
                        sys.modules.pop(module_name, None)


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def ella_tool(name: str, description: str) -> Callable:
    """Decorator that registers a function as an Ella tool.

    Usage:
        @ella_tool(name="my_tool", description="Does something useful.")
        def my_tool(arg1: str, arg2: int = 0) -> str:
            \"\"\"arg1: the input. arg2: optional count.\"\"\"
            ...
    """
    def decorator(fn: Callable) -> Callable:
        get_registry().register(fn, name, description)
        return fn

    return decorator
