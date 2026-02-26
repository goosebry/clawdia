"""Tests for ingestion layer — text handler, sequencer, and IngestionAgent with stubs."""
import asyncio
from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, patch

from ella.agents.protocol import LLMMessage, MessageUnit, SessionContext, UserTask, HandoffMessage
from ella.ingestion.text_handler import process_text
from ella.ingestion.sequencer import sort_by_message_id


# ── text_handler ──────────────────────────────────────────────────────────────

def test_process_text_plain():
    assert process_text("Hello world") == "Hello world"


def test_process_text_strips_html():
    assert process_text("<b>Bold</b> text") == "Bold text"


def test_process_text_strips_whitespace():
    assert process_text("  hello  ") == "hello"


def test_process_text_empty():
    assert process_text("") == ""


def test_process_text_chinese():
    assert process_text("你好世界") == "你好世界"


def test_process_text_mixed_html_chinese():
    result = process_text("<b>你好</b> <i>世界</i>")
    assert result == "你好 世界"


# ── sequencer ─────────────────────────────────────────────────────────────────

def _make_unit(message_id: int) -> MessageUnit:
    return MessageUnit(
        text=f"msg {message_id}",
        timestamp=datetime.now(timezone.utc),
        message_id=message_id,
        source="text",
        chat_id=1,
    )


def test_sort_already_ordered():
    units = [_make_unit(1), _make_unit(2), _make_unit(3)]
    result = sort_by_message_id(units)
    assert [u.message_id for u in result] == [1, 2, 3]


def test_sort_reverse_order():
    units = [_make_unit(3), _make_unit(1), _make_unit(2)]
    result = sort_by_message_id(units)
    assert [u.message_id for u in result] == [1, 2, 3]


def test_sort_single():
    units = [_make_unit(99)]
    result = sort_by_message_id(units)
    assert result[0].message_id == 99


def test_sort_empty():
    assert sort_by_message_id([]) == []


def test_sort_does_not_mutate_original():
    units = [_make_unit(3), _make_unit(1)]
    original_ids = [u.message_id for u in units]
    sort_by_message_id(units)
    assert [u.message_id for u in units] == original_ids


# ── IngestionAgent (with stub brain) ─────────────────────────────────────────

def _text_raw_update(message_id: int, text: str, chat_id: int = 100) -> dict:
    return {
        "update_id": message_id + 1000,
        "message": {
            "message_id": message_id,
            "date": 1700000000 + message_id,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "User"},
            "text": text,
        }
    }


class StubBrainAgent:
    """Captures whatever HandoffMessage IngestionAgent sends it."""
    def __init__(self):
        self.received: HandoffMessage | None = None

    async def handle(self, message):
        self.received = message
        return []


def test_ingestion_agent_text_batch():
    from ella.agents.ingestion_agent import IngestionAgent

    brain = StubBrainAgent()
    agent = IngestionAgent(brain_agent=brain)

    session = SessionContext(chat_id=100)
    updates = [
        _text_raw_update(3, "Third message"),
        _text_raw_update(1, "First message"),
        _text_raw_update(2, "Second message"),
    ]
    task = UserTask(raw_updates=updates, session=session)

    asyncio.run(agent.handle(task))

    assert brain.received is not None
    units: list[MessageUnit] = brain.received.payload

    # Order must be preserved by message_id
    assert [u.message_id for u in units] == [1, 2, 3]
    assert units[0].text == "First message"
    assert units[1].text == "Second message"
    assert units[2].text == "Third message"

    # Focus should have been populated
    assert len(session.focus) == 3
    assert all(m.role == "user" for m in session.focus)


def test_ingestion_agent_skips_empty_text():
    from ella.agents.ingestion_agent import IngestionAgent

    brain = StubBrainAgent()
    agent = IngestionAgent(brain_agent=brain)

    session = SessionContext(chat_id=100)
    updates = [
        _text_raw_update(1, "   "),   # whitespace only → stripped to ""
        _text_raw_update(2, "Valid"),
    ]
    task = UserTask(raw_updates=updates, session=session)

    asyncio.run(agent.handle(task))

    units: list[MessageUnit] = brain.received.payload
    assert len(units) == 1
    assert units[0].text == "Valid"


def test_ingestion_agent_no_processable_updates():
    from ella.agents.ingestion_agent import IngestionAgent

    brain = StubBrainAgent()
    agent = IngestionAgent(brain_agent=brain)

    session = SessionContext(chat_id=100)
    # Update with no message field
    task = UserTask(raw_updates=[{"update_id": 1}], session=session)

    result = asyncio.run(agent.handle(task))

    assert result == []
    assert brain.received is None


def test_ingestion_agent_source_tagged_correctly():
    from ella.agents.ingestion_agent import IngestionAgent

    brain = StubBrainAgent()
    agent = IngestionAgent(brain_agent=brain)

    session = SessionContext(chat_id=100)
    updates = [_text_raw_update(1, "A text message")]
    task = UserTask(raw_updates=updates, session=session)

    asyncio.run(agent.handle(task))
    units: list[MessageUnit] = brain.received.payload
    assert units[0].source == "text"
