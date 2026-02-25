"""Tests for BrainAgent logic — LLM and memory are fully stubbed."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from ella.agents.protocol import (
    HandoffMessage, LLMMessage, MessageUnit,
    ReplyPayload, SessionContext, Task,
)
from ella.agents.brain_agent import _parse_brain_output, _extract_tool_call, _contains_chinese


# ── Pure parsing helpers ──────────────────────────────────────────────────────

def test_parse_brain_output_valid_json():
    text = '{"reply": "Hello!", "language": "en", "tasks": []}'
    reply, lang, tasks = _parse_brain_output(text)
    assert reply == "Hello!"
    assert lang == "en"
    assert tasks == []


def test_parse_brain_output_with_tasks():
    text = json.dumps({
        "reply": "I'll help with that.",
        "language": "en",
        "tasks": [{"type": "coding", "description": "Write a script", "priority": 1}]
    })
    reply, lang, tasks = _parse_brain_output(text)
    assert len(tasks) == 1
    assert tasks[0]["type"] == "coding"


def test_parse_brain_output_chinese():
    text = '{"reply": "你好！", "language": "zh", "tasks": []}'
    reply, lang, tasks = _parse_brain_output(text)
    assert lang == "zh"
    assert reply == "你好！"


def test_parse_brain_output_fallback_plain_text():
    """Non-JSON output should be returned as reply with auto-detected language."""
    text = "I'm sorry, I can't do that."
    reply, lang, tasks = _parse_brain_output(text)
    assert reply == text
    assert lang == "en"
    assert tasks == []


def test_parse_brain_output_fallback_chinese_text():
    text = "很抱歉，我无法完成这个任务。"
    reply, lang, tasks = _parse_brain_output(text)
    assert lang == "zh"


def test_parse_brain_output_json_embedded_in_text():
    """JSON block inside surrounding text should still be extracted."""
    text = 'Here is my response: {"reply": "Done!", "language": "en", "tasks": []} Thanks.'
    reply, lang, tasks = _parse_brain_output(text)
    assert reply == "Done!"


# ── Tool call extraction ──────────────────────────────────────────────────────

def test_extract_tool_call_qwen_format():
    text = '<tool_call>{"name": "web_search", "arguments": {"query": "python MLX"}}</tool_call>'
    result = _extract_tool_call(text)
    assert result is not None
    assert result["name"] == "web_search"
    assert result["arguments"]["query"] == "python MLX"


def test_extract_tool_call_none_for_plain_text():
    text = "This is just a regular reply with no tool call."
    result = _extract_tool_call(text)
    assert result is None


def test_extract_tool_call_malformed_json():
    text = "<tool_call>{invalid json here}</tool_call>"
    result = _extract_tool_call(text)
    assert result is None


# ── Chinese detection ─────────────────────────────────────────────────────────

def test_contains_chinese_yes():
    assert _contains_chinese("你好世界") is True
    assert _contains_chinese("Hello 世界") is True


def test_contains_chinese_no():
    assert _contains_chinese("Hello world") is False
    assert _contains_chinese("") is False
    assert _contains_chinese("123 abc !@#") is False


# ── BrainAgent integration (LLM fully stubbed) ───────────────────────────────

def _make_session(chat_id: int = 1) -> SessionContext:
    session = SessionContext(chat_id=chat_id)
    session.focus = [LLMMessage(role="user", content="What is 2+2?")]
    return session


class StubReplyAgent:
    def __init__(self):
        self.calls = []

    async def handle(self, message):
        self.calls.append(message)
        return []


class StubTaskAgent:
    def __init__(self):
        self.calls = []

    async def handle(self, message):
        self.calls.append(message)
        return []


def test_brain_agent_routes_reply_and_tasks():
    """BrainAgent should always call both ReplyAgent and TaskAgent."""
    from ella.agents.brain_agent import BrainAgent

    reply_agent = StubReplyAgent()
    task_agent = StubTaskAgent()
    brain = BrainAgent(reply_agent=reply_agent, task_agent=task_agent)

    session = _make_session()

    handoff = HandoffMessage(
        payload=[MessageUnit(
            text="What is 2+2?",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            message_id=1,
            source="text",
            chat_id=1,
        )],
        session=session,
    )

    llm_output = json.dumps({"reply": "The answer is 4.", "language": "en", "tasks": []})

    with (
        patch("ella.agents.brain_agent.get_registry") as mock_reg,
        patch("ella.agents.brain_agent.get_goal_store") as mock_goal,
        patch("ella.agents.brain_agent.BrainAgent._run_tool_loop",
              new=AsyncMock(return_value=("The answer is 4.", "en", []))),
    ):
        mock_reg.return_value.get_schemas.return_value = []

        goal_store = MagicMock()
        goal_store.create = AsyncMock()
        goal_store.append_step = AsyncMock()
        goal_store.complete = AsyncMock()
        mock_goal.return_value = goal_store

        asyncio.run(brain.handle(handoff))

    assert len(reply_agent.calls) == 1
    reply_payload = reply_agent.calls[0].payload
    assert isinstance(reply_payload, ReplyPayload)
    assert reply_payload.text == "The answer is 4."
    assert reply_payload.language == "en"

    assert len(task_agent.calls) == 1
    task_list = task_agent.calls[0].payload
    assert task_list == []


def test_brain_agent_routes_tasks():
    """BrainAgent should create Task objects and pass them to TaskAgent."""
    from ella.agents.brain_agent import BrainAgent

    reply_agent = StubReplyAgent()
    task_agent = StubTaskAgent()
    brain = BrainAgent(reply_agent=reply_agent, task_agent=task_agent)

    session = _make_session()
    handoff = HandoffMessage(payload=[], session=session)

    tasks_from_llm = [
        {"type": "coding", "description": "Write unit tests", "priority": 1},
        {"type": "document", "description": "Update README", "priority": 2},
    ]

    with (
        patch("ella.agents.brain_agent.get_registry") as mock_reg,
        patch("ella.agents.brain_agent.get_goal_store") as mock_goal,
        patch("ella.agents.brain_agent.BrainAgent._run_tool_loop",
              new=AsyncMock(return_value=("I'll get that done.", "en", tasks_from_llm))),
    ):
        mock_reg.return_value.get_schemas.return_value = []

        goal_store = MagicMock()
        goal_store.create = AsyncMock()
        goal_store.append_step = AsyncMock()
        goal_store.complete = AsyncMock()
        mock_goal.return_value = goal_store

        asyncio.run(brain.handle(handoff))

    task_list: list[Task] = task_agent.calls[0].payload
    assert len(task_list) == 2
    assert task_list[0].task_type == "coding"
    assert task_list[1].task_type == "document"
    assert task_list[0].priority == 1
    assert task_list[1].priority == 2


def test_brain_agent_ignores_non_handoff():
    from ella.agents.brain_agent import BrainAgent
    from ella.agents.protocol import UserTask

    brain = BrainAgent(reply_agent=StubReplyAgent(), task_agent=StubTaskAgent())
    session = _make_session()
    task = UserTask(raw_updates=[], session=session)

    result = asyncio.run(brain.handle(task))
    assert result == []
