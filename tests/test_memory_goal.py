"""Tests for ella/memory/goal.py — serialisation and GoalStore with a mock Redis."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from ella.memory.goal import JobGoal, GoalStore, StepSummary


# ── JobGoal serialisation ─────────────────────────────────────────────────────

def test_job_goal_new():
    goal = JobGoal.new(chat_id=42, objective="Do something")
    assert goal.chat_id == 42
    assert goal.objective == "Do something"
    assert goal.status == "running"
    assert goal.job_id != ""
    assert goal.steps_done == []
    assert goal.shared_notes == {}
    assert goal.partial_outputs == []


def test_job_goal_round_trip_json():
    goal = JobGoal.new(chat_id=1, objective="Test round-trip")
    goal.steps_done = [StepSummary(step_index=0, agent="BrainAgent", summary="Step done")]
    goal.shared_notes = {"key": "value"}
    goal.partial_outputs = ["output.txt"]
    goal.status = "complete"

    serialised = goal.to_json()
    restored = JobGoal.from_json(serialised)

    assert restored.chat_id == goal.chat_id
    assert restored.objective == goal.objective
    assert restored.status == "complete"
    assert len(restored.steps_done) == 1
    assert restored.steps_done[0].agent == "BrainAgent"
    assert restored.steps_done[0].summary == "Step done"
    assert restored.shared_notes == {"key": "value"}
    assert restored.partial_outputs == ["output.txt"]


def test_step_summary_fields():
    s = StepSummary(step_index=2, agent="TaskAgent", summary="Done")
    assert s.step_index == 2
    assert s.agent == "TaskAgent"
    assert s.completed_at != ""


# ── GoalStore with mock Redis ─────────────────────────────────────────────────

def _make_store():
    """Return a GoalStore backed by an in-memory dict mock."""
    store_data = {}

    redis_mock = MagicMock()
    redis_mock.set = AsyncMock(side_effect=lambda key, val, ex=None: store_data.__setitem__(key, val))
    redis_mock.get = AsyncMock(side_effect=lambda key: store_data.get(key))
    redis_mock.delete = AsyncMock(side_effect=lambda key: store_data.pop(key, None))

    store = GoalStore(redis_mock)
    store._ttl = 3600
    return store, store_data


def test_goal_store_create_and_read():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Test")

    asyncio.run(store.create(goal))
    retrieved = asyncio.run(store.read(goal.job_id))

    assert retrieved is not None
    assert retrieved.job_id == goal.job_id
    assert retrieved.objective == "Test"


def test_goal_store_read_missing():
    store, _ = _make_store()
    result = asyncio.run(store.read("nonexistent-id"))
    assert result is None


def test_goal_store_append_step():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Multi-step job")
    asyncio.run(store.create(goal))

    summary = StepSummary(step_index=0, agent="BrainAgent", summary="First step done")
    asyncio.run(store.append_step(goal.job_id, summary))

    updated = asyncio.run(store.read(goal.job_id))
    assert len(updated.steps_done) == 1
    assert updated.steps_done[0].summary == "First step done"


def test_goal_store_update_notes():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Notes test")
    asyncio.run(store.create(goal))

    asyncio.run(store.update_notes(goal.job_id, {"lang": "zh", "priority": "high"}))
    updated = asyncio.run(store.read(goal.job_id))
    assert updated.shared_notes["lang"] == "zh"
    assert updated.shared_notes["priority"] == "high"


def test_goal_store_complete():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Finish me")
    asyncio.run(store.create(goal))
    asyncio.run(store.complete(goal.job_id))

    updated = asyncio.run(store.read(goal.job_id))
    assert updated.status == "complete"


def test_goal_store_fail():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Fail me")
    asyncio.run(store.create(goal))
    asyncio.run(store.fail(goal.job_id, reason="network error"))

    updated = asyncio.run(store.read(goal.job_id))
    assert updated.status == "failed"
    assert "network error" in updated.shared_notes.get("failure_reason", "")


def test_goal_store_delete():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Delete me")
    asyncio.run(store.create(goal))
    asyncio.run(store.delete(goal.job_id))

    result = asyncio.run(store.read(goal.job_id))
    assert result is None


def test_goal_store_add_output():
    store, _ = _make_store()
    goal = JobGoal.new(chat_id=1, objective="Output test")
    asyncio.run(store.create(goal))
    asyncio.run(store.add_output(goal.job_id, "result.txt"))
    asyncio.run(store.add_output(goal.job_id, "report.pdf"))

    updated = asyncio.run(store.read(goal.job_id))
    assert "result.txt" in updated.partial_outputs
    assert "report.pdf" in updated.partial_outputs
