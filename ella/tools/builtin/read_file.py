"""Built-in tool: read a local file."""
from __future__ import annotations

from pathlib import Path

from ella.tools.registry import ella_tool

MAX_BYTES = 50_000  # ~50 KB cap to avoid flooding the context window


@ella_tool(
    name="read_file",
    description=(
        "Read the contents of a local file and return it as text (up to 50 KB). "
        "Use when: the user asks you to read, review, or summarise a specific file; "
        "a task requires reading a file's contents before acting on it. "
        "Do NOT use speculatively — only when a specific file path is known and needed."
    ),
)
def read_file(path: str, encoding: str = "utf-8") -> str:
    """path: absolute or relative file path. encoding: file encoding (default utf-8)."""
    file = Path(path).expanduser()
    if not file.exists():
        return f"File not found: {path}"
    if not file.is_file():
        return f"Path is not a file: {path}"

    try:
        size = file.stat().st_size
        with file.open("r", encoding=encoding, errors="replace") as f:
            content = f.read(MAX_BYTES)
        if size > MAX_BYTES:
            content += f"\n\n[... truncated — file is {size} bytes, showing first {MAX_BYTES}]"
        return content
    except Exception as exc:
        return f"Error reading file: {exc}"
