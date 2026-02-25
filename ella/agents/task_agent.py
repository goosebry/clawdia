"""TaskAgent: enqueues extracted tasks to Redis/Celery and monitors progress.

Receives a list[Task] from BrainAgent, enqueues each one to Celery,
then polls task state and sends Telegram progress updates.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ella.agents.base_agent import BaseAgent
from ella.agents.protocol import HandoffMessage, Task, UserTask
from ella.communications.telegram.sender import get_sender

logger = logging.getLogger(__name__)

# Polling config
_POLL_INTERVAL = 5.0   # seconds between status checks
_MAX_WAIT = 3600.0     # give up after 1 hour


class TaskAgent(BaseAgent):
    async def handle(self, message: UserTask | HandoffMessage) -> list[HandoffMessage]:
        if not isinstance(message, HandoffMessage):
            return []

        tasks: list[Task] = message.payload
        if not isinstance(tasks, list) or not tasks:
            return []

        # Filter out non-Task items just in case
        tasks = [t for t in tasks if isinstance(t, Task)]
        if not tasks:
            return []

        sender = get_sender()

        # Enqueue and monitor each task as background asyncio tasks so they
        # do not block the main message-processing loop.
        for task in tasks:
            asyncio.create_task(
                self._enqueue_and_monitor(task, message.session.chat_id)
            )

        return []

    async def _enqueue_and_monitor(self, task: Task, chat_id: int) -> None:
        from ella.tasks.celery_app import celery_app

        try:
            celery_result = celery_app.send_task(
                "ella.tasks.worker.execute_task",
                kwargs={
                    "task_id": task.task_id,
                    "job_id": task.job_id,
                    "task_type": task.task_type,
                    "description": task.description,
                    "chat_id": chat_id,
                    "priority": task.priority,
                },
            )
            logger.info(
                "Enqueued task %s (type=%s) → celery_id=%s",
                task.task_id,
                task.task_type,
                celery_result.id,
            )
        except Exception:
            logger.exception("Failed to enqueue task %s", task.task_id)
            return

        # Poll until done or timeout — SUCCESS is handled by the worker sending
        # its own voice reply. We only surface genuine failures here.
        elapsed = 0.0
        last_state = ""

        while elapsed < _MAX_WAIT:
            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

            try:
                state = celery_result.state
            except Exception:
                state = "UNKNOWN"

            if state == last_state:
                continue
            last_state = state

            logger.info("Task %s state → %s", task.task_id, state)

            if state in ("SUCCESS", "FAILURE", "REVOKED"):
                if state == "FAILURE":
                    try:
                        result_info = str(celery_result.result)[:200]
                        logger.error("Task %s failed: %s", task.task_id, result_info)
                    except Exception:
                        pass
                break

        if elapsed >= _MAX_WAIT:
            logger.warning("Task %s timed out after 1 hour", task.task_id)


