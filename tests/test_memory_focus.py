"""Tests for ella/memory/focus.py — pure in-memory, no external deps."""
import pytest
from ella.agents.protocol import LLMMessage
from ella.memory.focus import (
    build_focus_prompt,
    build_system_message,
    summarise_focus,
    SYSTEM_PERSONA,
)


def test_system_message():
    msg = build_system_message()
    assert msg.role == "system"
    assert "Ella" in msg.content


def test_build_focus_prompt_minimal():
    focus = [LLMMessage(role="user", content="Hello")]
    messages = build_focus_prompt(focus, goal=None, knowledge_snippets=[])
    # First message is always the system persona
    assert messages[0].role == "system"
    assert "Ella" in messages[0].content
    # Last message is the user message
    assert messages[-1].role == "user"
    assert messages[-1].content == "Hello"


def test_build_focus_prompt_with_knowledge():
    focus = [LLMMessage(role="user", content="What time is it?")]
    snippets = ["[2024-01-01] user: previous question", "[2024-01-01] assistant: 3pm"]
    messages = build_focus_prompt(focus, goal=None, knowledge_snippets=snippets)
    # One system + one knowledge injection + one user
    roles = [m.role for m in messages]
    assert roles.count("system") == 2
    combined_content = " ".join(m.content for m in messages)
    assert "previous question" in combined_content


def test_build_focus_prompt_with_goal():
    from ella.memory.goal import JobGoal, StepSummary
    goal = JobGoal.new(chat_id=1, objective="Write a report")
    goal.steps_done = [StepSummary(step_index=0, agent="BrainAgent", summary="Researched topic")]
    goal.shared_notes = {"language": "English"}

    focus = [LLMMessage(role="user", content="Continue")]
    messages = build_focus_prompt(focus, goal=goal, knowledge_snippets=[])

    combined = " ".join(m.content for m in messages)
    assert "Write a report" in combined
    assert "Researched topic" in combined
    assert "English" in combined


def test_summarise_focus_empty():
    result = summarise_focus([])
    assert result == "(empty step)"


def test_summarise_focus_mixed_roles():
    focus = [
        LLMMessage(role="user", content="A" * 300),
        LLMMessage(role="assistant", content="B" * 400),
        LLMMessage(role="tool", content="C" * 300, tool_name="web_search"),
    ]
    result = summarise_focus(focus)
    assert "User:" in result
    assert "Assistant:" in result
    assert "Tool(web_search):" in result
    # Content is truncated
    assert len(result) < 1500


def test_summarise_focus_skips_system():
    focus = [
        LLMMessage(role="system", content="You are Ella."),
        LLMMessage(role="user", content="Hello"),
    ]
    result = summarise_focus(focus)
    assert "system" not in result.lower()
    assert "User: Hello" in result
