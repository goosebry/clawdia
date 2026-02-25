"""Telegram long-poll loop.

Uses getUpdates with timeout=20 (long-polling). Telegram holds the connection
for up to 20s before returning an empty list, so this is efficient with no
wasted polling requests. No hard rate limit applies to getUpdates.

On each batch, groups updates by chat_id and dispatches a UserTask to
IngestionAgent for each chat.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ella.agents.protocol import SessionContext, UserTask
from ella.config import get_settings
from ella.memory.goal import get_goal_store
from ella.memory.identity import register_reset_callback
from ella.memory.knowledge import get_knowledge_store
from ella.communications.telegram.models import TelegramUpdate

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org/bot{token}/getUpdates"

# Retry delays for exponential backoff (seconds)
_BACKOFF = [1, 2, 4, 8, 16, 30]


class TelegramPoller:
    def __init__(self, ingestion_agent: Any) -> None:
        self._agent = ingestion_agent
        self._token = get_settings().telegram_bot_token
        self._offset: int = 0
        self._client = httpx.AsyncClient(timeout=30.0)
        self._url = _BASE.format(token=self._token)
        # Persistent session context per chat_id — survives across message batches
        # so that conversation history, goal, and memory references are reused.
        self._sessions: dict[int, SessionContext] = {}
        # Register identity-change reset: when ~/Ella/ files change, wipe all
        # in-memory sessions and their Redis goals so Ella starts fresh with
        # her new sense of self.
        register_reset_callback(self._on_identity_changed)

    async def _get_updates(self) -> list[dict[str, Any]]:
        resp = await self._client.post(
            self._url,
            json={
                "offset": self._offset,
                "timeout": 20,
                "allowed_updates": ["message"],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"getUpdates error: {data.get('description')}")
        return data.get("result", [])

    def _group_by_chat(
        self, raw_updates: list[dict[str, Any]]
    ) -> dict[int, list[dict[str, Any]]]:
        groups: dict[int, list[dict[str, Any]]] = {}
        for raw in raw_updates:
            msg = raw.get("message")
            if not msg:
                continue
            chat_id = msg.get("chat", {}).get("id")
            if chat_id is None:
                continue
            groups.setdefault(chat_id, []).append(raw)
        return groups

    async def _dispatch(self, chat_id: int, raw_updates: list[dict[str, Any]]) -> None:
        # Reuse the existing session for this chat so that conversation history
        # (focus) and the current goal survive across message batches.
        if chat_id not in self._sessions:
            self._sessions[chat_id] = SessionContext(
                chat_id=chat_id,
                focus=[],
                goal=None,
                knowledge=get_knowledge_store(),
            )
            logger.info("Created new session for chat_id=%d", chat_id)
        session = self._sessions[chat_id]
        task = UserTask(raw_updates=raw_updates, session=session)
        try:
            await self._agent.handle(task)
        except Exception:
            logger.exception("Error processing chat_id=%d", chat_id)

    async def run(self) -> None:
        logger.info("Telegram poller started.")
        backoff_idx = 0

        while True:
            try:
                updates = await self._get_updates()
                backoff_idx = 0  # reset on success

                if updates:
                    # Advance offset to avoid reprocessing
                    self._offset = updates[-1]["update_id"] + 1

                    groups = self._group_by_chat(updates)
                    await asyncio.gather(
                        *(self._dispatch(chat_id, msgs) for chat_id, msgs in groups.items())
                    )

            except httpx.HTTPStatusError as exc:
                retry_after = int(exc.response.headers.get("Retry-After", _BACKOFF[backoff_idx]))
                logger.warning("HTTP %d from Telegram, retrying in %ds", exc.response.status_code, retry_after)
                await asyncio.sleep(retry_after)
                backoff_idx = min(backoff_idx + 1, len(_BACKOFF) - 1)

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                delay = _BACKOFF[backoff_idx]
                logger.warning("Network error (%s), retrying in %ds", exc, delay)
                await asyncio.sleep(delay)
                backoff_idx = min(backoff_idx + 1, len(_BACKOFF) - 1)

            except asyncio.CancelledError:
                logger.info("Poller cancelled.")
                break

            except Exception:
                delay = _BACKOFF[backoff_idx]
                logger.exception("Unexpected poller error, retrying in %ds", delay)
                await asyncio.sleep(delay)
                backoff_idx = min(backoff_idx + 1, len(_BACKOFF) - 1)

    async def _on_identity_changed(self) -> None:
        """Called by the identity watcher when ~/Ella/ files change.

        Clears all in-memory sessions (focus + goal refs) and deletes their
        Redis goals so every chat starts a fresh conversation with the new
        identity.  Qdrant long-term knowledge is left untouched.
        """
        if not self._sessions:
            logger.info("Identity changed — no active sessions to reset.")
            return

        goal_store = get_goal_store()
        chat_ids = list(self._sessions.keys())
        for chat_id in chat_ids:
            session = self._sessions[chat_id]
            # Delete the Redis goal for this session
            if session.goal is not None:
                try:
                    await goal_store.delete(session.goal.job_id)
                except Exception:
                    logger.warning(
                        "Could not delete goal %s for chat_id=%d",
                        session.goal.job_id, chat_id,
                    )

        # Clear all sessions — they will be recreated fresh on the next message
        self._sessions.clear()
        logger.info(
            "Identity changed — reset %d session(s): %s",
            len(chat_ids), chat_ids,
        )

    async def close(self) -> None:
        await self._client.aclose()
