"""Tests for ella/tools/builtin/*.py — no external calls needed for most."""
import os
import pytest


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_exists(tmp_path):
    from ella.tools.builtin.read_file import read_file
    f = tmp_path / "hello.txt"
    f.write_text("Hello, Ella!")
    result = read_file(str(f))
    assert result == "Hello, Ella!"


def test_read_file_not_found():
    from ella.tools.builtin.read_file import read_file
    result = read_file("/nonexistent/path/file.txt")
    assert "not found" in result.lower()


def test_read_file_directory():
    from ella.tools.builtin.read_file import read_file
    result = read_file("/tmp")
    assert "not a file" in result.lower()


def test_read_file_truncates_large(tmp_path):
    from ella.tools.builtin.read_file import read_file, MAX_BYTES
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (MAX_BYTES + 100))
    result = read_file(str(f))
    assert "truncated" in result.lower()


# ── write_file ────────────────────────────────────────────────────────────────

def test_write_file_creates_and_writes(tmp_path):
    from ella.tools.builtin.write_file import write_file
    dest = tmp_path / "output.txt"
    result = write_file(str(dest), "written content")
    assert "Written" in result
    assert dest.read_text() == "written content"


def test_write_file_append(tmp_path):
    from ella.tools.builtin.write_file import write_file
    dest = tmp_path / "append.txt"
    write_file(str(dest), "line 1\n", mode="write")
    write_file(str(dest), "line 2\n", mode="append")
    assert dest.read_text() == "line 1\nline 2\n"


def test_write_file_creates_parent_dirs(tmp_path):
    from ella.tools.builtin.write_file import write_file
    dest = tmp_path / "a" / "b" / "c" / "file.txt"
    result = write_file(str(dest), "nested")
    assert "Written" in result
    assert dest.read_text() == "nested"


def test_write_file_invalid_mode(tmp_path):
    from ella.tools.builtin.write_file import write_file
    dest = tmp_path / "x.txt"
    result = write_file(str(dest), "data", mode="invalid")
    assert "Error" in result


# ── run_shell ─────────────────────────────────────────────────────────────────

def test_run_shell_basic():
    from ella.tools.builtin.run_shell import run_shell
    result = run_shell("echo hello")
    assert result.strip() == "hello"


def test_run_shell_exit_code():
    from ella.tools.builtin.run_shell import run_shell
    result = run_shell("exit 1", timeout=5)
    assert "Exit code 1" in result or "exit" in result.lower()


def test_run_shell_blocked_command():
    from ella.tools.builtin.run_shell import run_shell
    result = run_shell("rm -rf /tmp/ella_test_should_not_run")
    assert "Blocked" in result or "not allowed" in result.lower()


def test_run_shell_sudo_blocked():
    from ella.tools.builtin.run_shell import run_shell
    result = run_shell("sudo ls")
    assert "Blocked" in result


def test_run_shell_multiword_output():
    from ella.tools.builtin.run_shell import run_shell
    result = run_shell("echo 'hello world'")
    assert "hello world" in result


def test_run_shell_empty_command():
    from ella.tools.builtin.run_shell import run_shell
    result = run_shell("")
    assert "Blocked" in result or "Empty" in result


# ── web_search (offline — just test error handling) ───────────────────────────

def test_web_search_returns_string():
    """web_search always returns a string even if the network is down."""
    from ella.tools.builtin.web_search import web_search
    result = web_search("test query")
    assert isinstance(result, str)


def test_web_search_max_results_clamp():
    """max_results is clamped 1–10 without crashing."""
    from ella.tools.builtin.web_search import web_search
    # These should not raise
    result = web_search("python", max_results=0)
    assert isinstance(result, str)
    result = web_search("python", max_results=999)
    assert isinstance(result, str)
