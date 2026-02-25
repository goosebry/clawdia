from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ella.memory.goal import JobGoal
    from ella.memory.knowledge import KnowledgeStore


@dataclass
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class MessageUnit:
    """A single normalised message after ingestion (text, STT transcript, or video summary)."""
    text: str
    timestamp: datetime
    message_id: int
    source: Literal["text", "voice", "video", "photo"]
    chat_id: int


@dataclass
class ReplyPayload:
    text: str                        # Full reply as a flat string (storage/logging)
    language: Literal["en", "zh"]
    # Each element is one complete spoken sentence — sent as a separate voice message.
    # If empty, ReplyAgent falls back to splitting `text` with the regex splitter.
    sentences: list[str] = field(default_factory=list)
    detail_text: str | None = None   # Full tool output sent as follow-up text message
    # Ordered emoji insertions: each entry is {"after": N, "emoji": "😂"} meaning
    # send that emoji as a plain text message after sentence N (0-indexed).
    # after=-1 means before the first sentence; after=999 means after the last.
    emojis: list[dict] = field(default_factory=list)
    # Ella's current emotion label — passed to TTS to shape delivery style.
    # One of the 27 Cowen & Keltner labels, or None when emotion engine is off.
    emotion: str | None = None


@dataclass
class Task:
    task_id: str
    job_id: str
    task_type: str
    description: str
    priority: int
    chat_id: int


@dataclass
class SessionContext:
    chat_id: int
    focus: list[LLMMessage] = field(default_factory=list)
    goal: "JobGoal | None" = None
    knowledge: "KnowledgeStore | None" = None


@dataclass
class UserTask:
    raw_updates: list[dict[str, Any]]
    session: SessionContext


@dataclass
class HandoffMessage:
    payload: Any
    session: SessionContext
