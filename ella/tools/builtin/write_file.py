"""Built-in tool: write or append to a local file."""
from __future__ import annotations

from pathlib import Path

from ella.tools.registry import ella_tool


@ella_tool(
    name="write_file",
    description=(
        "Write or append text to a local file. Creates missing directories automatically. "
        "Use when: the user explicitly asks to create, save, or write something to a file. "
        "Do NOT use unless the user has a clear intent to persist content to disk."
    ),
)
def write_file(path: str, content: str, mode: str = "write") -> str:
    """path: absolute or relative file path. content: text to write. mode: 'write' (overwrite) or 'append'."""
    if mode not in ("write", "append"):
        return "Error: mode must be 'write' or 'append'."

    file = Path(path).expanduser()
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        open_mode = "w" if mode == "write" else "a"
        with file.open(open_mode, encoding="utf-8") as f:
            f.write(content)
        action = "Written" if mode == "write" else "Appended"
        return f"{action} {len(content)} characters to {file}."
    except Exception as exc:
        return f"Error writing file: {exc}"
