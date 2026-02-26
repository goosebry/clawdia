"""ReplyAgent: converts BrainAgent's reply text to audio and sends via Telegram.

Steps:
  1. Receive ReplyPayload(text, language) from BrainAgent
  2. Call Qwen3-TTS (mlx-audio) → temp WAV file
  3. Convert WAV → OGG/Opus (Telegram's native voice format) via ffmpeg
  4. sendVoice via Telegram
  5. Consolidate the completed exchange into The Knowledge (Qdrant, Tier 3)
  6. Clean up temp files

OGG/Opus is Telegram's native voice codec — it's what Telegram records and
plays back natively, produces the smallest file size, and shows the waveform
animation in the chat. WAV is ~10x larger and shows as a plain audio file.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ella.agents.base_agent import BaseAgent
from ella.agents.protocol import HandoffMessage, ReplyPayload, UserTask
from ella.communications.telegram.sender import get_sender
from ella.tts.qwen3 import is_emoji_only, split_into_sentences, tts_to_wav

logger = logging.getLogger(__name__)

# Telegram's hard per-message character limit
_TG_MAX_CHARS = 4096


def _split_text(text: str, limit: int = _TG_MAX_CHARS) -> list[str]:
    """Split text into chunks that fit within Telegram's character limit.

    Splits preferring paragraph breaks (double newline), then single newlines,
    then sentence endings, then word boundaries — never mid-word.
    Each chunk is labelled (1/N), (2/N) ... when there are multiple parts.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text]

    parts: list[str] = []

    def _best_split(s: str, max_len: int) -> int:
        """Return the best split index ≤ max_len."""
        if len(s) <= max_len:
            return len(s)
        # Try paragraph break
        idx = s.rfind("\n\n", 0, max_len)
        if idx > max_len // 2:
            return idx + 2
        # Try newline
        idx = s.rfind("\n", 0, max_len)
        if idx > max_len // 2:
            return idx + 1
        # Try sentence end
        for punct in (". ", "。", "! ", "? ", "！", "？"):
            idx = s.rfind(punct, 0, max_len)
            if idx > max_len // 2:
                return idx + len(punct)
        # Try word boundary
        idx = s.rfind(" ", 0, max_len)
        if idx > 0:
            return idx + 1
        # Hard cut
        return max_len

    remaining = text
    while remaining:
        # Reserve space for the "(N/M)" label — max label is e.g. "(10/10)" = 7 chars
        split_at = _best_split(remaining, limit - 8)
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    # Add part labels when there are multiple chunks
    if len(parts) > 1:
        total = len(parts)
        parts = [f"({i}/{total})\n{p}" for i, p in enumerate(parts, 1)]

    return parts


def _wav_to_ogg(wav_path: str) -> str | None:
    """Convert a WAV file to OGG/Opus using ffmpeg.

    Returns the path to the .ogg file, or None if conversion failed.
    The caller is responsible for deleting the output file.
    """
    import json, time
    _DEBUG_LOG = "/Users/cl/Documents/App Project/Projects/ai.Ella/.cursor/debug-eb85ec.log"

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        tmp.close()
        ogg_path = tmp.name

        # #region debug log — measure WAV duration before conversion
        try:
            import wave as _wave
            with _wave.open(wav_path) as _wf:
                _wav_dur = _wf.getnframes() / _wf.getframerate()
            with open(_DEBUG_LOG, "a") as _f:
                _f.write(json.dumps({"sessionId":"eb85ec","hypothesisId":"D","location":"reply_agent.py:_wav_to_ogg","message":"WAV duration before ffmpeg","data":{"wav_path":wav_path,"wav_duration_s":round(_wav_dur,3),"wav_size":Path(wav_path).stat().st_size},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception as _e:
            pass
        # #endregion

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-c:a", "libopus",
                "-b:a", "48k",       # 48 kbps — clear speech, small file
                "-vbr", "on",
                "-compression_level", "10",
                "-application", "voip",  # optimised for speech
                ogg_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.error(
                "ffmpeg WAV→OGG failed (exit %d): %s",
                result.returncode,
                result.stderr.decode(errors="ignore")[-200:],
            )
            os.unlink(ogg_path)
            return None
        ogg_size = Path(ogg_path).stat().st_size

        # #region debug log — measure OGG duration after conversion, compare with WAV
        try:
            _ffprobe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", ogg_path],
                capture_output=True, timeout=10,
            )
            _ogg_dur = float(_ffprobe.stdout.decode().strip()) if _ffprobe.returncode == 0 else -1.0
            _ffprobe_stderr = _ffprobe.stderr.decode(errors="ignore")[-200:]
            with open(_DEBUG_LOG, "a") as _f:
                _f.write(json.dumps({"sessionId":"eb85ec","hypothesisId":"D","location":"reply_agent.py:_wav_to_ogg","message":"OGG duration after ffmpeg","data":{"ogg_path":ogg_path,"ogg_duration_s":round(_ogg_dur,3),"ogg_size":ogg_size,"wav_duration_s":round(_wav_dur,3) if '_wav_dur' in dir() else -1,"duration_diff_s":round(_ogg_dur - (_wav_dur if '_wav_dur' in dir() else 0), 3),"ffprobe_stderr":_ffprobe_stderr},"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
        # #endregion

        logger.info("[OGG] WAV→OGG done: %s (%d bytes)", ogg_path, ogg_size)
        return ogg_path
    except Exception:
        logger.exception("WAV→OGG conversion failed")
        return None


class ReplyAgent(BaseAgent):
    async def handle(self, message: UserTask | HandoffMessage) -> list[HandoffMessage]:
        if not isinstance(message, HandoffMessage):
            logger.warning("ReplyAgent received unexpected message type: %s", type(message))
            return []

        payload = message.payload
        if not isinstance(payload, ReplyPayload):
            logger.warning("ReplyAgent: expected ReplyPayload, got %s", type(payload))
            return []

        session = message.session
        sender = get_sender()

        logger.info(
            "[Reply] ── START chat_id=%d lang=%s sentences=%d emojis=%d ──",
            session.chat_id, payload.language,
            len(payload.sentences), len(payload.emojis),
        )

        # Each LLM sentence is one voice message — no further splitting.
        # Fall back to regex splitter only if the LLM didn't supply sentences[].
        if payload.sentences:
            tokens = [s.strip() for s in payload.sentences if s.strip()]
            logger.info(
                "Using LLM-provided sentences (%d) for chat_id=%d: %s",
                len(tokens), session.chat_id,
                [t[:40] for t in tokens],
            )
        else:
            tokens = split_into_sentences(payload.text)
            logger.info(
                "Reply split into %d token(s) via regex for chat_id=%d: %s",
                len(tokens), session.chat_id,
                [t[:40] for t in tokens],
            )

        # Build a position-indexed map of LLM-requested emojis.
        # Key = sentence index after which the emoji should be sent (-1 = before all).
        # A sentence index here refers to speakable sentences only (not inline emoji tokens).
        emoji_map: dict[int, list[str]] = {}
        for e in payload.emojis:
            pos = e.get("after", 999)
            emoji_map.setdefault(pos, []).append(e.get("emoji", ""))

        async def _send_emojis_at(pos: int) -> None:
            for emoji in emoji_map.get(pos, []):
                if emoji:
                    try:
                        await sender.send_message(chat_id=session.chat_id, text=emoji)
                        logger.info("Sent emoji at pos=%d to chat_id=%d: %s", pos, session.chat_id, emoji)
                    except Exception:
                        logger.exception("Failed to send emoji")

        # Emit any emojis requested before sentence 0
        await _send_emojis_at(-1)

        any_voice_sent = False
        sentence_index = -1  # incremented for each speakable sentence sent

        for token in tokens:
            # Inline emoji token extracted from the reply text — send as plain text.
            if is_emoji_only(token):
                try:
                    await sender.send_message(chat_id=session.chat_id, text=token)
                    logger.info("Sent inline emoji to chat_id=%d: %s", session.chat_id, token)
                except Exception:
                    logger.exception("Failed to send inline emoji token")
                continue

            # Skip tokens with no speakable content (lone punctuation, whitespace,
            # or JSON fragments that leaked through from a bad LLM parse).
            _speakable = re.sub(r'[^\w\u4e00-\u9fff\u3040-\u30ff]', '', token)
            if not _speakable:
                logger.debug("Skipping non-speakable token: %r", token[:40])
                continue

            # Speakable sentence — synthesise and send as a voice message.
            logger.info(
                "[TTS] sentence[%d] → %r (lang=%s)",
                sentence_index + 1, token, payload.language,
            )
            try:
                await sender.send_chat_action(session.chat_id, action="record_voice")
            except Exception:
                pass

            import time as _time
            _tts_t0 = _time.monotonic()
            wav_path = tts_to_wav(token, language=payload.language, emotion=payload.emotion)
            _tts_elapsed = _time.monotonic() - _tts_t0
            ogg_path: str | None = None

            if wav_path and Path(wav_path).exists():
                wav_size = Path(wav_path).stat().st_size
                logger.info(
                    "[TTS] WAV ready in %.2fs: %s (%d bytes)",
                    _tts_elapsed, wav_path, wav_size,
                )
                ogg_path = _wav_to_ogg(wav_path)
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
            else:
                logger.warning(
                    "[TTS] synthesis failed after %.2fs for: %r",
                    _tts_elapsed, token[:60],
                )

            voice_path = ogg_path or wav_path

            if voice_path and Path(voice_path).exists():
                ogg_size = Path(voice_path).stat().st_size
                logger.info(
                    "[Telegram] sendVoice sentence[%d] → chat_id=%d | file=%s (%d bytes) | text=%r",
                    sentence_index + 1, session.chat_id,
                    Path(voice_path).name, ogg_size, token[:60],
                )
                try:
                    await sender.send_voice(
                        chat_id=session.chat_id,
                        voice_path=voice_path,
                        caption=None,
                    )
                    any_voice_sent = True
                    sentence_index += 1
                    logger.info(
                        "[Telegram] ✓ voice sentence[%d] delivered to chat_id=%d",
                        sentence_index, session.chat_id,
                    )
                except Exception:
                    logger.exception(
                        "[Telegram] ✗ failed to send voice sentence[%d] to chat_id=%d",
                        sentence_index + 1, session.chat_id,
                    )
                    sentence_index += 1
                finally:
                    try:
                        os.unlink(voice_path)
                    except OSError:
                        pass
            else:
                logger.warning(
                    "[TTS→OGG] no output file for sentence[%d]: %r",
                    sentence_index + 1, token[:60],
                )
                sentence_index += 1

            # Send any LLM-requested emojis that belong after this sentence
            await _send_emojis_at(sentence_index)

        # Catch any emojis with an "after" index beyond the last sentence
        await _send_emojis_at(999)

        # If every speakable sentence failed TTS, fall back to a single text message
        if not any_voice_sent:
            logger.warning("All TTS attempts failed — sending text fallback")
            await _send_text_fallback(sender, session.chat_id, payload.text)

        # Send full detail text as follow-up message(s) after the voice reply.
        # Split into multiple messages if the content exceeds Telegram's 4096-char limit.
        if payload.detail_text:
            chunks = _split_text(payload.detail_text)
            logger.info(
                "Sending detail text to chat_id=%d (%d chunk(s), %d chars total)",
                session.chat_id, len(chunks), len(payload.detail_text),
            )
            for chunk in chunks:
                try:
                    await sender.send_message(chat_id=session.chat_id, text=chunk)
                except Exception:
                    logger.exception("Failed to send detail text chunk")

        # Consolidate exchange into The Knowledge
        if session.knowledge:
            try:
                user_text = " ".join(
                    m.content for m in session.focus if m.role == "user"
                )
                await session.knowledge.store_exchange(
                    chat_id=session.chat_id,
                    user_text=user_text,
                    assistant_text=payload.text,
                )
                logger.info("Stored exchange in Knowledge for chat_id=%d", session.chat_id)
            except Exception:
                logger.exception("Failed to store exchange in Knowledge")

        voices_sent = sentence_index + 1 if any_voice_sent else 0
        logger.info(
            "[Reply] ── DONE chat_id=%d | %d voice message(s) sent | detail=%s ──",
            session.chat_id, voices_sent,
            f"{len(payload.detail_text)} chars" if payload.detail_text else "none",
        )

        return []


async def _send_text_fallback(sender: Any, chat_id: int, text: str) -> None:
    for chunk in _split_text(text):
        try:
            await sender.send_message(chat_id=chat_id, text=chunk)
        except Exception:
            logger.exception("Failed to send text fallback chunk to chat_id=%d", chat_id)
