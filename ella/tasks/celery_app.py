"""Celery application instance.

Uses Redis as both the message broker and result backend.
Worker processes are started separately:
    celery -A ella.tasks.celery_app worker --loglevel=info
"""
from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init

from ella.config import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "ella",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["ella.tasks.worker"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        result_expires=86400,  # 24h
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )
    return app


@worker_process_init.connect
def _reset_async_singletons(**kwargs):
    """Reset all async singletons after Celery forks a worker process.

    httpx.AsyncClient, redis.asyncio, and other async clients bind to the
    event loop that was active when they were first constructed. After a fork
    that loop no longer exists in the child, so any cached instance from the
    parent process must be discarded. Each singleton will be recreated on
    first use, bound to the fresh loop created by execute_task.
    """
    import ella.communications.telegram.sender as _sender_mod
    import ella.memory.goal as _goal_mod

    _sender_mod._sender = None
    _goal_mod._goal_store = None


celery_app = create_celery_app()
