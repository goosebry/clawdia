"""Ingestion handler for Telegram voice messages.

Downloads the OGG file, runs mlx-whisper for transcription + language detection,
and returns the transcript text.

Emotion detection from voice is intentionally NOT done here via acoustic SER.
Acoustic-only models (wav2vec2, HuBERT, emotion2vec) cannot reliably detect
emotion from natural conversational Chinese speech — controlled/flat-voiced
angry speech is routinely misclassified as "neutral" or "fear" at 100%
confidence.

Instead, emotion is inferred from the transcript text by the brain LLM, which
understands what the words actually mean. This is far more accurate for Chinese.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ella.config import get_settings
from ella.communications.telegram.sender import get_sender

logger = logging.getLogger(__name__)


async def transcribe_voice(file_id: str) -> str:
    """Download a Telegram voice file and return the transcription text."""
    settings = get_settings()
    sender = get_sender()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / f"{file_id}.ogg"
        await sender.download_file_id(file_id, dest)
        text, _ = _run_whisper(str(dest), settings.mlx_whisper_model)

    return text


def _run_whisper(audio_path: str, model_name: str) -> tuple[str, str]:
    """Load mlx-whisper on-demand, transcribe, return (text, language_code).

    language_code is the ISO-639 code detected by Whisper (e.g. 'zh', 'en').
    Defaults to 'en' on failure.
    """
    try:
        import gc
        import mlx.core as mx
        import mlx_whisper

        logger.info("Loading mlx-whisper model: %s", model_name)
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=None,  # auto-detect
            word_timestamps=False,
        )
        text: str = result.get("text", "").strip()
        language: str = result.get("language", "en") or "en"
        logger.info("Transcribed %d chars (lang=%s)", len(text), language)

        # Evict Whisper weights from the Metal heap before the chat LLM loads.
        # Without this, the Whisper model stays resident and the 14B chat model
        # triggers a Metal OOM when both compete for the same unified memory.
        try:
            gc.collect()
            mx.metal.clear_cache()
            logger.info("Whisper Metal cache cleared after transcription")
        except Exception:
            pass

        return (text if text else "[voice message — no speech detected]"), language

    except ImportError:
        logger.error("mlx-whisper not installed. Run: pip install mlx-whisper")
        return "[voice message — mlx-whisper not available]", "en"
    except Exception:
        logger.exception("Whisper transcription failed")
        return "[voice message — transcription failed]", "en"
