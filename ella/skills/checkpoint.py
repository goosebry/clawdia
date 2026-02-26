"""SkillCheckpointStore — Redis + MySQL dual-write persistence for skill executions.

Redis is the fast working copy (AOF-persisted, survives reboots).
MySQL is the permanent source of truth (survives Docker volume loss).

On startup, if a paused run exists in MySQL but has no Redis key (volume was
lost), the system reconstructs the SkillCheckpoint from MySQL and re-saves it
to Redis before resuming.

Redis key pattern: ella:skill:{run_id}  (no TTL — cleared on completion/cancel)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Literal

import aiomysql
import redis.asyncio as aioredis

from ella.config import get_settings
from ella.skills.base import SkillCheckpoint

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "ella:skill:"


class SkillCheckpointStore:
    """Dual-write checkpoint store: Redis (fast) + MySQL (permanent)."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        db_pool: aiomysql.Pool,
    ) -> None:
        self._redis = redis_client
        self._pool = db_pool

    def _redis_key(self, run_id: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{run_id}"

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save(self, checkpoint: SkillCheckpoint) -> None:
        """Persist checkpoint to Redis (immediate) and MySQL (async upsert)."""
        checkpoint.touch()
        data = asdict(checkpoint)
        # Redis: fast working copy, no TTL
        await self._redis.set(self._redis_key(checkpoint.run_id), json.dumps(data))
        # MySQL: permanent backup
        await self._upsert_mysql(checkpoint)

    async def _upsert_mysql(self, cp: SkillCheckpoint) -> None:
        sql = """
            INSERT INTO ella_skill_runs
                (run_id, chat_id, skill_name, goal, status, phase, cycle,
                 sources_done, notes_snapshot, started_at, updated_at)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            AS new_row
            ON DUPLICATE KEY UPDATE
                status        = new_row.status,
                phase         = new_row.phase,
                cycle         = new_row.cycle,
                sources_done  = new_row.sources_done,
                notes_snapshot= new_row.notes_snapshot,
                updated_at    = new_row.updated_at,
                completed_at  = IF(new_row.status IN ('completed','cancelled','failed'),
                                   new_row.updated_at, completed_at)
        """
        now = cp.updated_at
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (
                    cp.run_id,
                    cp.chat_id,
                    cp.skill_name,
                    cp.goal,
                    cp.status,
                    cp.phase,
                    cp.cycle,
                    json.dumps(cp.sources_done),
                    json.dumps(cp.notes),
                    now,
                    now,
                ))
            await conn.commit()

    async def update_summary(self, run_id: str, summary: str, stored_points: int) -> None:
        """Write final synthesis summary and stored_points on completion."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE ella_skill_runs SET summary=%s, stored_points=%s WHERE run_id=%s",
                    (summary, stored_points, run_id),
                )
            await conn.commit()

    async def save_open_questions(self, run_id: str, questions: list[str]) -> None:
        """Persist unresolved questions after max cycles."""
        if not questions:
            return
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                rows = [(run_id, q, datetime.now(timezone.utc).isoformat()) for q in questions]
                await cur.executemany(
                    "INSERT INTO ella_skill_open_questions (run_id, question, created_at) VALUES (%s, %s, %s)",
                    rows,
                )
            await conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    async def load(self, run_id: str) -> SkillCheckpoint | None:
        """Load checkpoint from Redis; fall back to MySQL if Redis key missing."""
        raw = await self._redis.get(self._redis_key(run_id))
        if raw:
            try:
                return SkillCheckpoint(**json.loads(raw))
            except Exception:
                logger.warning("[Checkpoint] Failed to parse Redis data for %s — falling back to MySQL", run_id)

        # Redis miss (volume lost?) — reconstruct from MySQL
        return await self._load_from_mysql(run_id)

    async def _load_from_mysql(self, run_id: str) -> SkillCheckpoint | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM ella_skill_runs WHERE run_id = %s", (run_id,)
                )
                row = await cur.fetchone()
        if not row:
            return None

        cp = SkillCheckpoint(
            run_id=row["run_id"],
            skill_name=row["skill_name"],
            chat_id=row["chat_id"],
            goal=row["goal"],
            phase=row["phase"] or "research",
            cycle=row["cycle"] or 1,
            notes=json.loads(row["notes_snapshot"] or "[]"),
            questions=[],
            artifacts=[],
            sources_done=json.loads(row["sources_done"] or "[]"),
            status=row["status"],
            updated_at=(row["updated_at"] or datetime.now(timezone.utc)).isoformat()
            if not isinstance(row["updated_at"], str)
            else row["updated_at"],
        )
        # Re-save to Redis so subsequent loads are fast
        await self._redis.set(self._redis_key(run_id), json.dumps(asdict(cp)))
        logger.info("[Checkpoint] Reconstructed %s from MySQL and restored to Redis", run_id)
        return cp

    async def list_paused(self, chat_id: int | None = None) -> list[SkillCheckpoint]:
        """Return all paused executions, optionally filtered by chat_id."""
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if chat_id is not None:
                    await cur.execute(
                        "SELECT run_id FROM ella_skill_runs WHERE status='paused' AND chat_id=%s",
                        (chat_id,),
                    )
                else:
                    await cur.execute(
                        "SELECT run_id FROM ella_skill_runs WHERE status='paused'"
                    )
                rows = await cur.fetchall()

        checkpoints: list[SkillCheckpoint] = []
        for row in rows:
            cp = await self.load(row["run_id"])
            if cp:
                checkpoints.append(cp)
        return checkpoints

    async def list_active(self, chat_id: int) -> list[SkillCheckpoint]:
        """Return all running or paused executions for a chat."""
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT run_id FROM ella_skill_runs WHERE status IN ('running','paused') AND chat_id=%s",
                    (chat_id,),
                )
                rows = await cur.fetchall()

        checkpoints: list[SkillCheckpoint] = []
        for row in rows:
            cp = await self.load(row["run_id"])
            if cp:
                checkpoints.append(cp)
        return checkpoints

    async def list_resumable(self, chat_id: int | None, max_age_hours: int = 24) -> list[SkillCheckpoint]:
        """Return recently failed/paused executions that can be resumed.

        Looks back max_age_hours so stale failures are not offered for resume.
        Ordered newest first. Pass chat_id=None to query across all chats (used
        at startup to auto-resume all interrupted runs).
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if chat_id is not None:
                    await cur.execute(
                        "SELECT run_id FROM ella_skill_runs "
                        "WHERE status IN ('failed','paused') AND chat_id=%s "
                        "  AND updated_at >= NOW() - INTERVAL %s HOUR "
                        "ORDER BY updated_at DESC",
                        (chat_id, max_age_hours),
                    )
                else:
                    await cur.execute(
                        "SELECT run_id FROM ella_skill_runs "
                        "WHERE status IN ('failed','paused') "
                        "  AND updated_at >= NOW() - INTERVAL %s HOUR "
                        "ORDER BY updated_at DESC",
                        (max_age_hours,),
                    )
                rows = await cur.fetchall()

        checkpoints: list[SkillCheckpoint] = []
        for row in rows:
            cp = await self.load(row["run_id"])
            if cp:
                checkpoints.append(cp)
        return checkpoints

    # ── Lifecycle transitions ─────────────────────────────────────────────────

    async def mark_completed(self, run_id: str) -> None:
        """Mark as completed and remove from Redis."""
        await self._redis.delete(self._redis_key(run_id))
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE ella_skill_runs SET status='completed', completed_at=%s, updated_at=%s WHERE run_id=%s",
                    (now, now, run_id),
                )
            await conn.commit()

    async def mark_cancelled(self, run_id: str) -> None:
        """Mark as cancelled and remove from Redis."""
        await self._redis.delete(self._redis_key(run_id))
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE ella_skill_runs SET status='cancelled', completed_at=%s, updated_at=%s WHERE run_id=%s",
                    (now, now, run_id),
                )
            await conn.commit()

    async def mark_failed(self, run_id: str) -> None:
        """Mark as failed and remove from Redis."""
        await self._redis.delete(self._redis_key(run_id))
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE ella_skill_runs SET status='failed', completed_at=%s, updated_at=%s WHERE run_id=%s",
                    (now, now, run_id),
                )
            await conn.commit()

    # ── Skill reply slot (ask_user ↔ BrainAgent handshake) ───────────────────
    # When a skill calls ask_user(), it writes a pending-reply slot to Redis.
    # BrainAgent detects this on the next turn, writes the user's message as
    # the answer, and clears the slot. The skill polls until it finds the answer.

    def _reply_key(self, chat_id: int) -> str:
        return f"ella:skill:reply:{chat_id}"

    async def set_pending_reply(self, chat_id: int, run_id: str, question: str, ttl: int = 300) -> None:
        """Record that run_id is waiting for a user reply to question (TTL=5 min)."""
        data = json.dumps({"run_id": run_id, "question": question, "answer": None})
        await self._redis.set(self._reply_key(chat_id), data, ex=ttl)

    async def get_pending_reply(self, chat_id: int) -> dict | None:
        """Return the pending reply slot for this chat, or None if none exists."""
        raw = await self._redis.get(self._reply_key(chat_id))
        if raw:
            return json.loads(raw)
        return None

    async def deliver_reply(self, chat_id: int, answer: str) -> str | None:
        """Write the user's answer into the slot. Returns run_id or None if no slot."""
        raw = await self._redis.get(self._reply_key(chat_id))
        if not raw:
            return None
        slot = json.loads(raw)
        slot["answer"] = answer
        # Keep TTL going (another 5 min for the skill to pick it up)
        await self._redis.set(self._reply_key(chat_id), json.dumps(slot), ex=300)
        return slot.get("run_id")

    async def clear_pending_reply(self, chat_id: int) -> None:
        """Remove the reply slot once the skill has consumed the answer."""
        await self._redis.delete(self._reply_key(chat_id))

    # ── Search-confirm slot (tiered confirm-before-search handshake) ──────────
    # When the planner fires confirm_first=true, BrainAgent writes this slot
    # with the goal and original query, then sends the user a tiered-ask message.
    # On the next turn, BrainAgent reads the slot, classifies the reply into
    # one of four intents (learn / web / rednote / skip), and routes accordingly.

    def _search_confirm_key(self, chat_id: int) -> str:
        return f"ella:search:confirm:{chat_id}"

    async def set_search_confirm(
        self, chat_id: int, goal: str, original_query: str, ttl: int = 300
    ) -> None:
        """Store a pending search-confirm for this chat (TTL = 5 min)."""
        data = json.dumps({
            "goal": goal,
            "original_query": original_query,
            "asked_at": datetime.now(timezone.utc).isoformat(),
        })
        await self._redis.set(self._search_confirm_key(chat_id), data, ex=ttl)
        logger.debug("[Confirm] set_search_confirm chat_id=%d goal=%r", chat_id, goal[:60])

    async def get_search_confirm(self, chat_id: int) -> dict | None:
        """Return the pending search-confirm slot, or None if expired/absent."""
        raw = await self._redis.get(self._search_confirm_key(chat_id))
        if raw:
            return json.loads(raw)
        return None

    async def clear_search_confirm(self, chat_id: int) -> None:
        """Remove the search-confirm slot after it has been consumed."""
        await self._redis.delete(self._search_confirm_key(chat_id))
        logger.debug("[Confirm] cleared search_confirm for chat_id=%d", chat_id)


_store: SkillCheckpointStore | None = None


async def get_checkpoint_store() -> SkillCheckpointStore:
    """Return the shared SkillCheckpointStore, creating it lazily on first call."""
    global _store
    if _store is None:
        settings = get_settings()
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        pool = await aiomysql.create_pool(
            host=_parse_mysql_host(settings.database_url),
            port=_parse_mysql_port(settings.database_url),
            user=_parse_mysql_user(settings.database_url),
            password=_parse_mysql_password(settings.database_url),
            db=_parse_mysql_db(settings.database_url),
            autocommit=False,
            minsize=1,
            maxsize=5,
        )
        _store = SkillCheckpointStore(redis_client=redis_client, db_pool=pool)
    return _store


# ── MySQL DSN helpers (reuse pattern from emotion/store.py) ───────────────────

def _parse_mysql_host(dsn: str) -> str:
    from urllib.parse import urlparse
    return urlparse(dsn).hostname or "localhost"

def _parse_mysql_port(dsn: str) -> int:
    from urllib.parse import urlparse
    return urlparse(dsn).port or 3306

def _parse_mysql_user(dsn: str) -> str:
    from urllib.parse import urlparse
    return urlparse(dsn).username or "ella"

def _parse_mysql_password(dsn: str) -> str:
    from urllib.parse import urlparse
    return urlparse(dsn).password or ""

def _parse_mysql_db(dsn: str) -> str:
    from urllib.parse import urlparse
    path = urlparse(dsn).path
    return path.lstrip("/") if path else "ella"
