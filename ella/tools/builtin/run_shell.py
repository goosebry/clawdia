"""Built-in tool: run a sandboxed shell command."""
from __future__ import annotations

import shlex
import subprocess

from ella.tools.registry import ella_tool

# Commands that are never allowed regardless of input
_BLOCKED = frozenset({
    "rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "kill", "killall",
    "pkill", "halt", "poweroff", "sudo", "su", "chmod", "chown",
    "curl", "wget",  # use web_search tool instead
})


def _is_safe(command: str) -> tuple[bool, str]:
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return False, f"Invalid command syntax: {e}"
    if not parts:
        return False, "Empty command."
    executable = parts[0].split("/")[-1]
    if executable in _BLOCKED:
        return False, f"Command '{executable}' is not allowed for safety reasons."
    return True, ""


@ella_tool(
    name="run_shell",
    description=(
        "Run a shell command and return its stdout. Dangerous commands (rm, sudo, curl, etc.) are blocked. "
        "Use when: the user explicitly asks to run a command, check system state, or execute a script. "
        "Do NOT use speculatively — only when the user has a clear, specific shell task to accomplish."
    ),
)
def run_shell(command: str, timeout: int = 30) -> str:
    """command: the shell command to execute. timeout: max seconds to wait (default 30)."""
    safe, reason = _is_safe(command)
    if not safe:
        return f"Blocked: {reason}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=min(timeout, 120),
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            return f"Exit code {result.returncode}\nstdout: {output}\nstderr: {stderr}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as exc:
        return f"Error: {exc}"
