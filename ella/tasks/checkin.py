"""Weekly check-in task.

This task is designed to be run periodically by Celery Beat (e.g., every Friday).
It sends a proactive, warm message to known chat IDs asking how their week went,
feeding the response back into the Tiago Forte memory pipeline.
"""
from __future__ import annotations

import logging
from celery.app.task import Task

from ella.tasks.celery_app import celery_app
from ella.communications.telegram.sender import get_sender

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def weekly_check_in(self: Task, chat_id: int) -> dict[str, Any]:
    """Proactively ask the user how they are feeling this week."""
    from asgiref.sync import async_to_sync

    message = (
        "Hey! Just checking in — how has your week been? "
        "Are we making good progress on our projects and goals, or are you feeling overwhelmed?"
    )
    
    try:
        sender = get_sender()
        async_to_sync(sender.send_message)(chat_id, message)
        logger.info("Sent weekly proactive check-in to chat_id=%s", chat_id)
        return {"status": "sent", "chat_id": chat_id}
    except Exception as exc:
        logger.error("Failed to send weekly check-in to %s: %s", chat_id, exc)
        self.retry(exc=exc, countdown=60)
        return {"status": "failed"}

