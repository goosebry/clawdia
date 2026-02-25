"""Tests for ella/agents/protocol.py"""
from datetime import datetime, timezone
import pytest
from ella.agents.protocol import (
    LLMMessage,
    MessageUnit,
    ReplyPayload,
    Task,
    SessionContext,
    UserTask,
    HandoffMessage,
)


def test_llm_message_fields():
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.tool_call_id is None
    assert msg.tool_name is None


def test_llm_message_tool():
    msg = LLMMessage(role="tool", content="result", tool_call_id="abc", tool_name="web_search")
    assert msg.tool_name == "web_search"
    assert msg.tool_call_id == "abc"


def test_message_unit():
    ts = datetime.now(timezone.utc)
    unit = MessageUnit(text="hi", timestamp=ts, message_id=42, source="text", chat_id=1)
    assert unit.text == "hi"
    assert unit.message_id == 42
    assert unit.source == "text"


def test_reply_payload():
    p = ReplyPayload(text="Hello!", language="en")
    assert p.text == "Hello!"
    assert p.language == "en"


def test_task_fields():
    t = Task(task_id="t1", job_id="j1", task_type="coding",
             description="Write tests", priority=1, chat_id=100)
    assert t.task_type == "coding"
    assert t.priority == 1


def test_session_context_defaults():
    ctx = SessionContext(chat_id=99)
    assert ctx.chat_id == 99
    assert ctx.focus == []
    assert ctx.goal is None
    assert ctx.knowledge is None


def test_session_context_focus_append():
    ctx = SessionContext(chat_id=1)
    ctx.focus.append(LLMMessage(role="user", content="test"))
    assert len(ctx.focus) == 1


def test_user_task():
    ctx = SessionContext(chat_id=1)
    task = UserTask(raw_updates=[{"update_id": 1}], session=ctx)
    assert task.raw_updates[0]["update_id"] == 1
    assert task.session is ctx


def test_handoff_message():
    ctx = SessionContext(chat_id=1)
    hm = HandoffMessage(payload="some payload", session=ctx)
    assert hm.payload == "some payload"
    assert hm.session is ctx
