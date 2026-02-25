"""Async MySQL persistence for the emotion engine.

One row per chat_id in ella_emotion_state — cross-session, never expires.
ella_emotion_history keeps a rolling log of the last 10 state changes per user.

Requires aiomysql. Connection pool is created lazily on first use and shared
for the process lifetime.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List

from ella.emotion.models import AgentState, UserState

logger = logging.getLogger(__name__)

_pool = None


async def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    try:
        import aiomysql
        from ella.config import get_settings
        settings = get_settings()
        dsn = settings.database_url
        if not dsn:
            raise ValueError("DATABASE_URL is not configured — emotion store unavailable")
        # Parse mysql+aiomysql://user:pass@host:port/db or mysql://...
        # Strip scheme
        url = dsn.replace("mysql+aiomysql://", "").replace("mysql://", "")
        userpass, hostdb = url.split("@", 1)
        user, password = (userpass.split(":", 1) + [""])[:2]
        hostport, db = hostdb.split("/", 1)
        host, port = (hostport.split(":", 1) + ["3306"])[:2]
        _pool = await aiomysql.create_pool(
            host=host,
            port=int(port),
            user=user,
            password=password,
            db=db.split("?")[0],
            charset="utf8mb4",
            autocommit=True,
            minsize=1,
            maxsize=5,
        )
        logger.info("Emotion store MySQL pool ready (host=%s db=%s)", host, db.split("?")[0])
    except Exception:
        logger.exception("Failed to create emotion store MySQL pool")
        _pool = None
    return _pool


class EmotionStore:
    """Async CRUD for ella_emotion_state and ella_emotion_history."""

    async def upsert_agent_state(self, chat_id: int, state: AgentState) -> None:
        pool = await _get_pool()
        if pool is None:
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
            INSERT INTO ella_emotion_state
              (chat_id, agent_valence, agent_energy, agent_dominance, agent_emotion,
               agent_intensity, agent_momentum, session_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) AS new_vals
            ON DUPLICATE KEY UPDATE
              agent_valence   = new_vals.agent_valence,
              agent_energy    = new_vals.agent_energy,
              agent_dominance = new_vals.agent_dominance,
              agent_emotion   = new_vals.agent_emotion,
              agent_intensity = new_vals.agent_intensity,
              agent_momentum  = new_vals.agent_momentum,
              session_count   = new_vals.session_count,
              updated_at      = new_vals.updated_at
        """
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (
                        chat_id,
                        state.valence, state.energy, state.dominance, state.emotion,
                        state.intensity, state.momentum, state.session_count, now,
                    ))
        except Exception:
            logger.exception("upsert_agent_state failed for chat_id=%d", chat_id)

    async def upsert_user_state(self, chat_id: int, state: UserState) -> None:
        pool = await _get_pool()
        if pool is None:
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sql = """
            INSERT INTO ella_emotion_state
              (chat_id, user_valence, user_energy, user_dominance, user_emotion,
               user_intensity, user_detected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s) AS new_vals
            ON DUPLICATE KEY UPDATE
              user_valence     = new_vals.user_valence,
              user_energy      = new_vals.user_energy,
              user_dominance   = new_vals.user_dominance,
              user_emotion     = new_vals.user_emotion,
              user_intensity   = new_vals.user_intensity,
              user_detected_at = new_vals.user_detected_at
        """
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (
                        chat_id,
                        state.valence, state.energy, state.dominance, state.emotion,
                        state.intensity, now,
                    ))
        except Exception:
            logger.exception("upsert_user_state failed for chat_id=%d", chat_id)

    async def read_agent_state(self, chat_id: int) -> AgentState | None:
        pool = await _get_pool()
        if pool is None:
            return None
        sql = """
            SELECT agent_valence, agent_energy, agent_dominance, agent_emotion,
                   agent_intensity, agent_momentum, session_count, updated_at
            FROM ella_emotion_state
            WHERE chat_id = %s
        """
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (chat_id,))
                    row = await cur.fetchone()
                    if row is None:
                        return None
                    return AgentState(
                        valence=float(row[0]),
                        energy=float(row[1]),
                        dominance=float(row[2]),
                        emotion=str(row[3]),
                        intensity=float(row[4]),
                        momentum=float(row[5]),
                        session_count=int(row[6]),
                        updated_at=str(row[7]) if row[7] else "",
                    )
        except Exception:
            logger.exception("read_agent_state failed for chat_id=%d", chat_id)
            return None

    async def read_user_state(self, chat_id: int) -> UserState | None:
        pool = await _get_pool()
        if pool is None:
            return None
        sql = """
            SELECT user_valence, user_energy, user_dominance, user_emotion,
                   user_intensity, user_detected_at
            FROM ella_emotion_state
            WHERE chat_id = %s
        """
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, (chat_id,))
                    row = await cur.fetchone()
                    if row is None:
                        return None
                    return UserState(
                        valence=float(row[0]),
                        energy=float(row[1]),
                        dominance=float(row[2]),
                        emotion=str(row[3]),
                        intensity=float(row[4]),
                        detected_at=str(row[5]) if row[5] else "",
                    )
        except Exception:
            logger.exception("read_user_state failed for chat_id=%d", chat_id)
            return None

    async def append_history(
        self,
        chat_id: int,
        source: str,
        state: AgentState,
        trigger: str = "",
        note: str = "",
    ) -> None:
        """Append a history entry and prune to last 10 per chat_id."""
        pool = await _get_pool()
        if pool is None:
            return
        insert_sql = """
            INSERT INTO ella_emotion_history
              (chat_id, source, emotion, valence, energy, dominance, intensity, trigger_emotion, note)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        prune_sql = """
            DELETE FROM ella_emotion_history
            WHERE chat_id = %s
              AND id NOT IN (
                SELECT id FROM (
                  SELECT id FROM ella_emotion_history
                  WHERE chat_id = %s
                  ORDER BY created_at DESC
                  LIMIT 10
                ) t
              )
        """
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(insert_sql, (
                        chat_id, source, state.emotion,
                        state.valence, state.energy, state.dominance, state.intensity,
                        trigger[:255] if trigger else None,
                        note[:255] if note else None,
                    ))
                    await cur.execute(prune_sql, (chat_id, chat_id))
        except Exception:
            logger.exception("append_history failed for chat_id=%d", chat_id)

    async def all_chat_ids(self) -> List[int]:
        """Return all chat_ids that have an emotion state row — used by the decay loop."""
        pool = await _get_pool()
        if pool is None:
            return []
        sql = "SELECT chat_id FROM ella_emotion_state"
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql)
                    rows = await cur.fetchall()
                    return [int(r[0]) for r in rows]
        except Exception:
            logger.exception("all_chat_ids failed")
            return []


# ── Singleton ─────────────────────────────────────────────────────────────────

_store: EmotionStore | None = None


def get_emotion_store() -> EmotionStore:
    global _store
    if _store is None:
        _store = EmotionStore()
    return _store
