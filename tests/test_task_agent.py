"""Tests for TaskAgent — Celery and Telegram are stubbed."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ella.agents.protocol import HandoffMessage, SessionContext, Task, UserTask


def _make_tasks(n: int = 2, chat_id: int = 100) -> list[Task]:
    return [
        Task(
            task_id=f"task-{i}",
            job_id="job-1",
            task_type="coding",
            description=f"Task {i} description",
            priority=1,
            chat_id=chat_id,
        )
        for i in range(n)
    ]


def test_task_agent_enqueues_tasks():
    from ella.agents.task_agent import TaskAgent

    session = SessionContext(chat_id=100)
    tasks = _make_tasks(2)
    handoff = HandoffMessage(payload=tasks, session=session)

    enqueued_names = []
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.id = "celery-id-1"

    with (
        patch("ella.agents.task_agent.get_sender") as mock_sender_fn,
        patch("ella.tasks.celery_app.celery_app.send_task",
              side_effect=lambda name, **kw: (enqueued_names.append(kw.get("kwargs", {}).get("task_id", "?")), mock_result)[1]),
    ):
        sender = MagicMock()
        sender.send_message = AsyncMock()
        mock_sender_fn.return_value = sender

        agent = TaskAgent()
        asyncio.run(agent.handle(handoff))

    # Both tasks were enqueued
    assert len(enqueued_names) == 2
    assert "task-0" in enqueued_names
    assert "task-1" in enqueued_names

    # Confirmation message was sent
    sender.send_message.assert_called()


def test_task_agent_sends_confirmation():
    from ella.agents.task_agent import TaskAgent

    session = SessionContext(chat_id=100)
    tasks = _make_tasks(1)
    handoff = HandoffMessage(payload=tasks, session=session)

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.id = "celery-id-2"

    messages_sent = []

    with (
        patch("ella.agents.task_agent.get_sender") as mock_sender_fn,
        patch("ella.tasks.celery_app.celery_app.send_task", return_value=mock_result),
    ):
        sender = MagicMock()
        sender.send_message = AsyncMock(side_effect=lambda **kw: messages_sent.append(kw["text"]))
        mock_sender_fn.return_value = sender

        agent = TaskAgent()
        asyncio.run(agent.handle(handoff))

    # At least the initial confirmation was sent
    assert any("Scheduling" in m or "task" in m.lower() for m in messages_sent)


def test_task_agent_empty_list_is_noop():
    from ella.agents.task_agent import TaskAgent

    session = SessionContext(chat_id=100)
    handoff = HandoffMessage(payload=[], session=session)

    with patch("ella.agents.task_agent.get_sender") as mock_sender_fn:
        sender = MagicMock()
        sender.send_message = AsyncMock()
        mock_sender_fn.return_value = sender

        agent = TaskAgent()
        result = asyncio.run(agent.handle(handoff))

    assert result == []
    sender.send_message.assert_not_called()


def test_task_agent_ignores_non_handoff():
    from ella.agents.task_agent import TaskAgent

    agent = TaskAgent()
    session = SessionContext(chat_id=1)
    task = UserTask(raw_updates=[], session=session)
    result = asyncio.run(agent.handle(task))
    assert result == []


def test_task_agent_handles_enqueue_failure():
    from ella.agents.task_agent import TaskAgent

    session = SessionContext(chat_id=100)
    tasks = _make_tasks(1)
    handoff = HandoffMessage(payload=tasks, session=session)

    with (
        patch("ella.agents.task_agent.get_sender") as mock_sender_fn,
        patch("ella.tasks.celery_app.celery_app.send_task",
              side_effect=Exception("Redis not available")),
    ):
        sender = MagicMock()
        sender.send_message = AsyncMock()
        mock_sender_fn.return_value = sender

        agent = TaskAgent()
        # Should not raise
        asyncio.run(agent.handle(handoff))

    # An error/warning message should have been sent
    calls = [call.kwargs.get("text", "") for call in sender.send_message.call_args_list]
    assert any("Failed" in t or "failed" in t or "⚠️" in t for t in calls)
