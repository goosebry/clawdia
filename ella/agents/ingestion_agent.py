"""IngestionAgent: converts raw Telegram updates into an ordered list[MessageUnit].

Pipeline per update:
  text   → pass-through (strip HTML formatting)
  voice  → mlx-whisper STT (on-demand, ~0.5 GB) — transcript only, no acoustic SER
  video  → mlx-vlm Qwen2.5-VL summary (on-demand, ~4-5 GB)

All units are sorted by message_id to preserve original send order,
then forwarded to BrainAgent as a HandoffMessage.

Emotion is detected by the brain LLM from the transcript text — not from
acoustic features. The LLM understands what the words mean; acoustic models
don't, especially for natural conversational Chinese.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from ella.agents.base_agent import BaseAgent
from ella.agents.protocol import (
    HandoffMessage,
    LLMMessage,
    MessageUnit,
    SessionContext,
    UserTask,
)
from ella.ingestion.photo_handler import describe_photo
from ella.ingestion.sequencer import sort_by_message_id
from ella.ingestion.text_handler import process_text
from ella.ingestion.video_handler import summarise_video
from ella.ingestion.voice_handler import transcribe_voice  # returns str (transcript only)
from ella.communications.telegram.models import TelegramMessage
from ella.communications.telegram.sender import get_sender

logger = logging.getLogger(__name__)

# Telegram clears the typing indicator after 5 seconds, so we refresh at 4s.
_TYPING_REFRESH_INTERVAL = 4.0


async def _keep_typing(chat_id: int, stop_event: asyncio.Event) -> None:
    """Repeatedly send 'typing' action until stop_event is set."""
    sender = get_sender()
    while not stop_event.is_set():
        try:
            await sender.send_chat_action(chat_id, action="typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.ensure_future(stop_event.wait())),
                timeout=_TYPING_REFRESH_INTERVAL,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


class IngestionAgent(BaseAgent):
    def __init__(self, brain_agent: "BaseAgent") -> None:
        self._brain = brain_agent

    async def handle(self, message: UserTask | HandoffMessage) -> list[HandoffMessage]:
        if not isinstance(message, UserTask):
            logger.warning("IngestionAgent received unexpected message type: %s", type(message))
            return []

        chat_id = message.session.chat_id

        # Fire typing indicator immediately and keep it alive for the full
        # processing pipeline (STT / VL / LLM / TTS can take 10-30 seconds).
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_keep_typing(chat_id, stop_typing))

        try:
            units = await self._process_updates(message.raw_updates, message.session)
            ordered = sort_by_message_id(units)

            if not ordered:
                logger.info("No processable messages in batch for chat_id=%d", chat_id)
                return []

            # Reset Focus to a clean slate for this turn.
            # Focus is the scratchpad for the current tool-call loop only —
            # it should contain only what is happening right now, not history.
            # Conversation history lives in the Goal's step summaries (Redis).
            message.session.focus.clear()
            for unit in ordered:
                ts = unit.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                content = f"[{ts}] {unit.text}"
                message.session.focus.append(LLMMessage(role="user", content=content))

            handoff = HandoffMessage(payload=ordered, session=message.session)
            return await self._brain.handle(handoff)
        finally:
            # Stop the typing loop before the voice reply is sent
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _process_updates(
        self, raw_updates: list[dict], session: SessionContext
    ) -> list[MessageUnit]:
        units: list[MessageUnit] = []

        for raw in raw_updates:
            raw_msg = raw.get("message")
            if not raw_msg:
                continue
            try:
                msg = TelegramMessage.from_raw(raw_msg)
            except Exception:
                logger.exception("Failed to parse TelegramMessage")
                continue

            timestamp = datetime.fromtimestamp(msg.date, tz=timezone.utc)

            if msg.text:
                text = process_text(msg.text)
                if text:
                    units.append(MessageUnit(
                        text=text,
                        timestamp=timestamp,
                        message_id=msg.message_id,
                        source="text",
                        chat_id=session.chat_id,
                    ))

            elif msg.voice:
                logger.info("Processing voice message %d", msg.message_id)
                text = await transcribe_voice(msg.voice.file_id)
                units.append(MessageUnit(
                    text=text,
                    timestamp=timestamp,
                    message_id=msg.message_id,
                    source="voice",
                    chat_id=session.chat_id,
                ))

            elif msg.video:
                logger.info("Processing video message %d", msg.message_id)
                text = await summarise_video(msg.video.file_id)
                units.append(MessageUnit(
                    text=text,
                    timestamp=timestamp,
                    message_id=msg.message_id,
                    source="video",
                    chat_id=session.chat_id,
                ))

            elif msg.audio:
                logger.info("Processing audio document message %d", msg.message_id)
                text = await transcribe_voice(msg.audio.file_id)
                units.append(MessageUnit(
                    text=text,
                    timestamp=timestamp,
                    message_id=msg.message_id,
                    source="voice",
                    chat_id=session.chat_id,
                ))

            elif msg.best_photo:
                logger.info("Processing photo message %d", msg.message_id)
                text = await describe_photo(msg.best_photo.file_id, caption=msg.caption)
                units.append(MessageUnit(
                    text=f"[User sent a photo] {text}",
                    timestamp=timestamp,
                    message_id=msg.message_id,
                    source="photo",
                    chat_id=session.chat_id,
                ))

            else:
                logger.debug("Skipping unsupported message type for message_id=%d", msg.message_id)

        return units
