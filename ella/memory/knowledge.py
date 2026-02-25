"""Tier 3 — The Knowledge: permanent semantic memory backed by Qdrant.

Collections:
  ella_conversations  — user+assistant turn embeddings per chat_id
  ella_task_patterns  — successful multi-step job patterns for seeding future jobs
  ella_user_prefs     — inferred user preferences and working style
  ella_identity       — Ella's own identity (re-embedded on every identity reload)

Access frequency: once per job start (recall) + once per job end (consolidate).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ella.config import get_settings
from ella.memory.embedder import embed

logger = logging.getLogger(__name__)


def _summarise_exchange(user_text: str, assistant_text: str) -> str:
    """Produce a compact one-line digest of a conversation exchange.

    The summary is what gets stored in Qdrant and shown to the LLM during
    recall — so it must be short (< 150 chars), informative, and written in
    third-person so the LLM doesn't mistake it for something to repeat.

    Strategy (no LLM required):
      - Strip Telegram timestamps from the user text
      - Take the first sentence of each side (the most informative part)
      - Trim to fit a compact format: "User asked: X | Ella replied: Y"
    """
    import re as _re

    # Strip leading [YYYY-MM-DD HH:MM:SS UTC] timestamps injected by the poller
    user_clean = _re.sub(r"^\[[\d\-T :UTC+]+\]\s*", "", user_text).strip()
    ella_clean = _re.sub(r"^\[[\d\-T :UTC+]+\]\s*", "", assistant_text).strip()

    # Take only the first sentence of each side
    def _first_sentence(text: str, max_chars: int = 60) -> str:
        # Split on Chinese or English sentence-ending punctuation
        m = _re.search(r"[。！？.!?]", text)
        if m and m.start() > 0:
            text = text[:m.start() + 1]
        return text[:max_chars].strip()

    u = _first_sentence(user_clean, 60)
    e = _first_sentence(ella_clean, 60)

    if u and e:
        return f"User: {u} | Ella: {e}"
    elif u:
        return f"User: {u}"
    elif e:
        return f"Ella: {e}"
    return "exchange"

COLLECTION_CONVERSATIONS   = "ella_conversations"
COLLECTION_TASK_PATTERNS   = "ella_task_patterns"
COLLECTION_USER_PREFS      = "ella_user_prefs"
COLLECTION_IDENTITY        = "ella_identity"
COLLECTION_TOPIC_KNOWLEDGE = "ella_topic_knowledge"
VECTOR_DIM = 384

SENSITIVITY_LEVELS = ("public", "internal", "private", "secret")
SENSITIVITY_DEFAULT = "secret"


class KnowledgeStore:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self._top_k = get_settings().knowledge_recall_top_k

    async def recall(
        self,
        query: str,
        chat_id: int,
        top_k: int | None = None,
        include_task_patterns: bool = True,
        skip_conversations: bool = False,
    ) -> list[str]:
        """Return top-K semantically relevant snippets for this chat_id.

        Always includes the most relevant chunks from ella_identity so Ella
        can draw on her own backstory, relationship context, and personality
        even when the user's query doesn't match past conversations.
        Identity snippets are prepended so they appear first in the prompt.

        When skip_conversations=True (topic shift detected), conversation
        memory is omitted so stale context from the previous topic doesn't
        pollute the new one.  Identity is always returned.

        Conversation recall is limited to exchanges within
        settings.knowledge_conv_recall_hours to avoid surfacing yesterday's
        (or older) chats.  Set that setting to 0 to disable the window.
        """
        settings = get_settings()
        k = top_k or self._top_k
        vector = embed(query)
        chat_filter = Filter(
            must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]
        )

        # ── ella_identity: always recall top-2 relevant identity chunks ──────
        identity_snippets: list[str] = []
        try:
            id_response = await self._client.query_points(
                collection_name=COLLECTION_IDENTITY,
                query=vector,
                limit=2,
                with_payload=True,
            )
            for hit in id_response.points:
                payload = hit.payload or {}
                text = payload.get("text", "")
                if text:
                    identity_snippets.append(f"[Ella's identity] {text}")
        except Exception:
            logger.warning("ella_identity recall failed — skipping identity snippets")

        # ── ella_conversations: past exchanges for this chat ──────────────────
        # Fetch k+1 so we can drop the single most-recent exchange.
        # The most recent turn is already present in the Goal's step history
        # (injected via focus.py Tier 2).  Including it here as well causes the
        # LLM to see the same exchange twice — strongly priming it to repeat
        # whatever Ella said last.  Dropping the newest point avoids this.
        #
        # Recency gate: only surface exchanges from the last N hours so stale
        # conversations from previous days don't pollute today's context.
        conv_snippets: list[str] = []
        if not skip_conversations:
            # Build conversation filter — optionally restricted to a recency window
            recall_minutes = getattr(settings, "knowledge_conv_recall_minutes", 15)
            if recall_minutes > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=recall_minutes)
                conv_query_filter = Filter(
                    must=[
                        FieldCondition(key="chat_id", match=MatchValue(value=chat_id)),
                        FieldCondition(
                            key="timestamp",
                            range=DatetimeRange(gte=cutoff),
                        ),
                    ]
                )
            else:
                conv_query_filter = chat_filter

            # Always use chat_id-only filter for Qdrant (timestamp field is stored as
            # a plain string payload, not a Qdrant datetime index, so DatetimeRange
            # filter is silently ignored — it accepts the query but returns unfiltered
            # results). Apply the recency window in Python after retrieval instead.
            # Fetch extra candidates (k*3) to ensure we have enough after trimming.
            try:
                conv_response = await self._client.query_points(
                    collection_name=COLLECTION_CONVERSATIONS,
                    query=vector,
                    query_filter=chat_filter,
                    limit=(k + 1) * 3,
                    with_payload=True,
                )
            except Exception:
                logger.warning("[Knowledge] ella_conversations recall failed")
                conv_response = type("R", (), {"points": []})()

            if recall_minutes > 0:
                cutoff_iso = cutoff.isoformat()
                before = len(conv_response.points)
                conv_response.points = [
                    h for h in conv_response.points
                    if (h.payload or {}).get("timestamp", "") >= cutoff_iso
                ]
                logger.info(
                    "[Knowledge] conv recall: %d → %d hit(s) after %d-min window trim (cutoff=%s)",
                    before, len(conv_response.points), recall_minutes, cutoff_iso[:16],
                )

            # Sort by timestamp descending, skip the most recent one, cap at k
            hits = sorted(
                conv_response.points,
                key=lambda h: (h.payload or {}).get("timestamp", ""),
                reverse=True,
            )[:k + 1]
            logger.debug(
                "[Knowledge] conv recall: %d hit(s) within %dm window",
                len(hits), recall_minutes,
            )
            for hit in hits[1:]:  # skip index 0 = most recent
                payload = hit.payload or {}
                role = payload.get("role", "?")
                ts = payload.get("timestamp", "")
                if role == "exchange":
                    # Use pre-computed summary when available (new format).
                    # Fall back to truncated raw text for legacy points.
                    summary = payload.get("summary", "")
                    if summary:
                        conv_snippets.append(f"[{ts}] {summary}")
                    else:
                        user_t = (payload.get("user_text", "") or "")[:60]
                        ella_t = (payload.get("assistant_text", "") or "")[:60]
                        conv_snippets.append(f"[{ts}] User: {user_t} | Ella: {ella_t}")
                else:
                    # Legacy single-role point
                    text = (payload.get("text", "") or "")[:80]
                    conv_snippets.append(f"[{ts}] {role}: {text}")

        # ── ella_task_patterns ────────────────────────────────────────────────
        task_snippets: list[str] = []
        if include_task_patterns and not skip_conversations:
            pattern_response = await self._client.query_points(
                collection_name=COLLECTION_TASK_PATTERNS,
                query=vector,
                limit=max(1, k // 2),
                with_payload=True,
            )
            for hit in pattern_response.points:
                payload = hit.payload or {}
                task_type = payload.get("task_type", "")
                summary = payload.get("steps_summary", "")
                outcome = payload.get("outcome", "")
                task_snippets.append(
                    f"[Past task pattern — {task_type}] {summary} → {outcome}"
                )

        # ── ella_topic_knowledge: learned knowledge from skill sessions ───────
        # Always queried (not gated on topic-shift) so Ella can answer "what did
        # you learn" questions using the actual stored knowledge, not just
        # conversation history. Sensitivity filter: only public + internal by
        # default (private/secret knowledge is not surfaced in general recall).
        topic_snippets: list[str] = []
        try:
            topic_response = await self._client.query_points(
                collection_name=COLLECTION_TOPIC_KNOWLEDGE,
                query=vector,
                limit=3,
                with_payload=True,
            )
            settings = get_settings()
            freshness_days = getattr(settings, "knowledge_freshness_days", 30)
            now = datetime.now(timezone.utc)
            for hit in topic_response.points:
                payload = hit.payload or {}
                sensitivity = payload.get("sensitivity", SENSITIVITY_DEFAULT)
                if sensitivity in ("private", "secret"):
                    continue
                chunk = payload.get("chunk_text", "")
                topic = payload.get("topic", "")
                learned_at_str = payload.get("learned_at", "")
                staleness = ""
                if learned_at_str:
                    try:
                        age_days = (now - datetime.fromisoformat(learned_at_str.replace("Z", "+00:00"))).days
                        if age_days > freshness_days:
                            staleness = f" [learned {age_days}d ago — may be stale]"
                    except Exception:
                        pass
                if chunk:
                    topic_snippets.append(f"[Learned knowledge — {topic}]{staleness}\n{chunk[:300]}")
        except Exception:
            logger.warning("[Knowledge] ella_topic_knowledge recall failed — skipping")

        # Identity first, then learned knowledge, then conversation history.
        return identity_snippets + topic_snippets + conv_snippets + task_snippets

    async def store(
        self,
        chat_id: int,
        role: str,
        text: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a single conversation turn into The Knowledge."""
        vector = embed(text)
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "role": role,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        await self._client.upsert(
            collection_name=COLLECTION_CONVERSATIONS, points=[point]
        )

    async def store_exchange(
        self,
        chat_id: int,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Store a completed turn as a compact summary in Qdrant.

        A short digest is generated from the raw exchange and stored instead
        of the full verbatim text.  This prevents recall from flooding the
        LLM prompt with long exchanges that it then copies verbatim.

        The vector is computed from the combined raw text (for good semantic
        retrieval) but only the summary is stored in the payload.
        """
        summary = _summarise_exchange(user_text, assistant_text)
        combined = f"user: {user_text}\nella: {assistant_text}"
        vector = embed(combined)
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "role": "exchange",
            "user_text": user_text[:120],
            "assistant_text": assistant_text[:120],
            "summary": summary,
            # 'text' field used by the recall formatter
            "text": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        await self._client.upsert(
            collection_name=COLLECTION_CONVERSATIONS, points=[point]
        )
        logger.debug(
            "[Knowledge] stored exchange summary (%d chars): %s",
            len(summary), summary[:100],
        )

    async def consolidate_task_pattern(
        self,
        task_type: str,
        steps_summary: str,
        outcome: str,
    ) -> None:
        """Record a successful multi-step job pattern for future recall."""
        text = f"{task_type}: {steps_summary} -> {outcome}"
        vector = embed(text)
        payload: dict[str, Any] = {
            "task_type": task_type,
            "steps_summary": steps_summary,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        await self._client.upsert(
            collection_name=COLLECTION_TASK_PATTERNS, points=[point]
        )

    async def store_user_pref(
        self, chat_id: int, preference_key: str, value: str
    ) -> None:
        text = f"{preference_key}: {value}"
        vector = embed(text)
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "preference_key": preference_key,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        await self._client.upsert(
            collection_name=COLLECTION_USER_PREFS, points=[point]
        )

    async def store_topic_knowledge(
        self,
        topic: str,
        chunk_text: str,
        source_url: str = "",
        source_type: str = "web",
        sensitivity: str = SENSITIVITY_DEFAULT,
        learned_by_chat_id: int = 0,
    ) -> None:
        """Store a knowledge chunk from a learning session into ella_topic_knowledge.

        topic: normalised topic label (e.g. "machine learning / transformers")
        chunk_text: the knowledge passage (~512 tokens)
        source_url: origin URL or file path
        source_type: "web" | "pdf" | "rednote" | "user_input" | "bot_input"
        sensitivity: "public" | "internal" | "private" | "secret" (default "secret")
        learned_by_chat_id: chat_id that triggered the learning run
        """
        if sensitivity not in SENSITIVITY_LEVELS:
            sensitivity = SENSITIVITY_DEFAULT

        vector = embed(chunk_text)
        now = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "topic": topic,
            "source_url": source_url,
            "source_type": source_type,
            "chunk_text": chunk_text,
            "sensitivity": sensitivity,
            "learned_at": now,
            "learned_by_chat_id": learned_by_chat_id,
        }
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        await self._client.upsert(
            collection_name=COLLECTION_TOPIC_KNOWLEDGE, points=[point]
        )
        logger.debug(
            "[Knowledge] stored topic chunk: topic=%r source_type=%s sensitivity=%s len=%d",
            topic[:50], source_type, sensitivity, len(chunk_text),
        )

    async def recall_topic_knowledge(
        self,
        query: str,
        top_k: int = 5,
        sensitivity_allow: tuple[str, ...] = SENSITIVITY_LEVELS,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic search over ella_topic_knowledge.

        Returns list of payload dicts (with an added '_score' key), each
        annotated with a staleness warning if learned_at is older than
        KNOWLEDGE_FRESHNESS_DAYS.

        sensitivity_allow: tuple of sensitivity levels to include.
        min_score: minimum cosine similarity to include a result (0.0 = no filter).
        """
        settings = get_settings()
        freshness_days = getattr(settings, "knowledge_freshness_days", 30)

        vector = embed(query)
        try:
            response = await self._client.query_points(
                collection_name=COLLECTION_TOPIC_KNOWLEDGE,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            logger.warning("[Knowledge] ella_topic_knowledge recall failed")
            return []

        results: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for hit in response.points:
            if min_score > 0.0 and hit.score < min_score:
                logger.debug(
                    "[Knowledge] topic_knowledge hit skipped (score=%.3f < %.3f): topic=%r",
                    hit.score, min_score, (hit.payload or {}).get("topic", "?")[:40],
                )
                continue
            payload = dict(hit.payload or {})
            payload["_score"] = hit.score
            if payload.get("sensitivity", SENSITIVITY_DEFAULT) not in sensitivity_allow:
                continue
            learned_at_str = payload.get("learned_at", "")
            if learned_at_str:
                try:
                    learned_dt = datetime.fromisoformat(learned_at_str.replace("Z", "+00:00"))
                    age_days = (now - learned_dt).days
                    if age_days > freshness_days:
                        payload["stale_warning"] = (
                            f"This knowledge was learned {age_days} days ago — it may be out of date."
                        )
                except Exception:
                    pass
            results.append(payload)
        return results


_knowledge_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _knowledge_store
    if _knowledge_store is None:
        settings = get_settings()
        client = AsyncQdrantClient(url=settings.qdrant_url)
        _knowledge_store = KnowledgeStore(client)
    return _knowledge_store


async def ensure_collections() -> None:
    """Create Qdrant collections if they don't exist. Called at startup."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    existing = {c.name for c in (await client.get_collections()).collections}

    for name in (
        COLLECTION_CONVERSATIONS,
        COLLECTION_TASK_PATTERNS,
        COLLECTION_USER_PREFS,
        COLLECTION_IDENTITY,
        COLLECTION_TOPIC_KNOWLEDGE,
    ):
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )


async def refresh_identity_knowledge() -> None:
    """Re-embed Ella's identity files into the ella_identity Qdrant collection.

    This is called at startup and on every identity file change.  The entire
    collection is replaced (delete-all then re-insert) so stale content from
    a previous version of the files is never returned by recall.

    Each section of each identity file becomes a separate vector point,
    tagged with source='identity', file=<filename>, section=<heading>.
    This lets the LLM recall specific facets of Ella's identity (e.g. her
    personality, her role with the user) when composing a reply.
    """
    from ella.memory.identity import get_identity, ELLA_DIR

    ctx = get_identity()
    if not ctx.prompt_block:
        logger.warning("refresh_identity_knowledge: no identity content to embed.")
        return

    import re
    import uuid as _uuid

    settings = get_settings()
    # Short timeout — if Qdrant is busy (e.g. during a skill run) fail fast
    # rather than blocking the identity watcher for tens of seconds.
    client = AsyncQdrantClient(url=settings.qdrant_url, timeout=10.0)

    file_map = {
        "Identity.md": ctx.identity,
        "Soul.md": ctx.soul,
        "User.md": ctx.user,
    }

    points: list[PointStruct] = []
    now = datetime.now(timezone.utc).isoformat()
    # Namespace UUID for deterministic point IDs — same file+section always
    # produces the same UUID so upsert overwrites stale points in-place.
    # No drop-and-recreate needed, safe under concurrent load.
    _NS = _uuid.UUID("b1a2c3d4-e5f6-7890-abcd-ef1234567890")

    for filename, content in file_map.items():
        if not content:
            continue
        # Split into sections on ## headings so each section gets its own vector.
        # If there are no headings, treat the whole file as one chunk.
        raw_sections = re.split(r"(?m)^(## .+)$", content)
        sections: list[tuple[str, str]] = []
        # raw_sections alternates: [text_before_first_heading, heading, body, heading, body …]
        if raw_sections[0].strip():
            sections.append(("general", raw_sections[0].strip()))
        i = 1
        while i + 1 < len(raw_sections):
            heading = raw_sections[i].strip("# ").strip()
            body = raw_sections[i + 1].strip()
            if body:
                sections.append((heading, body))
            i += 2

        if not sections:
            sections = [("general", content)]

        for heading, body in sections:
            chunk = f"[{filename} — {heading}]\n{body}"
            vector = embed(chunk)
            payload: dict[str, Any] = {
                "source": "identity",
                "file": filename,
                "section": heading,
                "text": chunk,
                "timestamp": now,
            }
            # Deterministic ID: same file+section → same point → safe upsert
            point_id = str(_uuid.uuid5(_NS, f"{filename}::{heading}"))
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    if points:
        try:
            await client.upsert(collection_name=COLLECTION_IDENTITY, points=points)
        except Exception:
            logger.exception("Failed to upsert identity knowledge — skipping refresh.")
            return
        logger.info(
            "Identity knowledge refreshed: %d chunk(s) embedded from %s",
            len(points), ELLA_DIR,
        )
    else:
        logger.warning("refresh_identity_knowledge: nothing to embed after parsing.")
