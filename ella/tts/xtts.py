"""XTTS-v2 TTS singleton — permanently resident in memory (~4.6 GB).

XTTS-v2 is the only model kept loaded at all times because:
  - It needs ~4.6 GB unified RAM regardless
  - It is called on every reply (high frequency)
  - Its initialisation time (~10-20s) makes on-demand loading impractical

Supports English and Chinese (and 16 other languages).

XTTS-v2 has an internal token limit of ~400 tokens (~250 words, ~1500 chars).
Text that exceeds this is split into sentence-sized chunks, each synthesised
separately, then concatenated into a single WAV file via numpy.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

from ella.config import get_settings

logger = logging.getLogger(__name__)

_tts_instance = None

# XTTS-v2 safe chunk size in characters.  Stay well below the ~400-token limit
# so that longer words / punctuation do not push us over.
_XTTS_CHUNK_CHARS = 250


def get_tts():
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = _load_xtts()
    return _tts_instance


def _load_xtts():
    try:
        from TTS.api import TTS

        # COQUI_TOS_AGREED=1 suppresses the interactive Terms-of-Service prompt,
        # which would block startup when running as a background service with no stdin.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        logger.info("Loading XTTS-v2 (permanent resident)...")
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        )
        logger.info("XTTS-v2 loaded.")
        return tts
    except ImportError:
        logger.error("TTS not installed. Run: pip install TTS")
        return None
    except Exception:
        logger.exception("Failed to load XTTS-v2")
        return None


_DEFAULT_SPEAKER = "Claribel Dervla"

# Sentence-boundary patterns for English and Chinese
_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?。！？])\s+'          # English: after punctuation + whitespace
    r'|(?<=[。！？])'                # Chinese: after CJK punctuation (no space needed)
)

# Emoji Unicode ranges — deliberately excludes "Enclosed Alphanumerics"
# (U+2460–U+24FF, U+24C2–U+1F251) which overlap with plain Latin letters like
# "Q", "A", "B" used in Chinese informal writing (e.g. "Q弹", "A级").
# Only ranges that are unambiguously pictographic emoji are included.
_EMOJI_RANGES = (
    r'\U0001F600-\U0001F64F'   # emoticons (😀-🙏)
    r'\U0001F300-\U0001F5FF'   # misc symbols & pictographs (🌀-🗿)
    r'\U0001F680-\U0001F6FF'   # transport & map (🚀-🛿)
    r'\U0001F700-\U0001F77F'   # alchemical symbols
    r'\U0001F780-\U0001F7FF'   # geometric shapes extended
    r'\U0001F800-\U0001F8FF'   # supplemental arrows-C
    r'\U0001F900-\U0001F9FF'   # supplemental symbols & pictographs (🤀-🧿)
    r'\U0001FA00-\U0001FA6F'   # chess symbols
    r'\U0001FA70-\U0001FAFF'   # symbols extended-A (🩰-🫿)
    r'\u2600-\u26FF'           # misc symbols (☀-⛿)
    r'\u2700-\u27BF'           # dingbats (✀-➿)
    r'\uFE0F'                  # variation selector-16 (emoji presentation)
    r'\u200D'                  # zero-width joiner (compound emoji)
    r'\u20E3'                  # combining enclosing keycap
)

# Matches a token that is entirely emoji — no speakable characters.
_EMOJI_ONLY_RE = re.compile(r'^[' + _EMOJI_RANGES + r']+$')

# Splits a string at emoji run boundaries so each emoji sequence becomes its
# own discrete token.  Whitespace is NOT a split boundary.
_EMOJI_SPLIT_RE = re.compile(r'([' + _EMOJI_RANGES + r']+)')


def is_emoji_only(text: str) -> bool:
    """Return True if *text* contains only emoji/whitespace — no speakable chars."""
    return bool(_EMOJI_ONLY_RE.match(text))


def _split_sentences_always(text: str, limit: int = _XTTS_CHUNK_CHARS) -> list[str]:
    """Split speakable text into individual sentences, always — even when the total
    text is short.  Unlike _split_into_chunks this never short-circuits on length;
    every sentence boundary produces a new item.

    Sentences that exceed *limit* chars are further split at word boundaries so
    they stay within XTTS-v2's token budget.
    """
    text = text.strip()
    if not text:
        return []

    raw_sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not raw_sentences:
        return [text] if text else []

    result: list[str] = []
    for sent in raw_sentences:
        if len(sent) <= limit:
            result.append(sent)
        else:
            # Oversized single sentence — hard-split at word/char boundaries
            while sent:
                space_idx = sent.rfind(" ", 0, limit)
                cut = space_idx if space_idx > limit // 2 else limit
                result.append(sent[:cut].strip())
                sent = sent[cut:].strip()
    return [r for r in result if r]


def split_into_sentences(text: str, limit: int = _XTTS_CHUNK_CHARS) -> list[str]:
    """Split reply text into a sequence of speakable sentence tokens and emoji tokens.

    Always splits on sentence boundaries regardless of total text length, so that
    ReplyAgent sends each sentence as a separate voice message.  Emoji-only runs
    are kept in their correct position so they are sent as plain text messages
    between voice messages, preserving natural reading order.

    Example:
        "天哪，连续工作三十多个小时。你是不是需要休息一下？😂"
        → ["天哪，连续工作三十多个小时。", "你是不是需要休息一下？", "😂"]
    """
    # Split out emoji runs first so they become discrete tokens.
    raw_parts = [p.strip() for p in _EMOJI_SPLIT_RE.split(text) if p.strip()]

    result: list[str] = []
    for part in raw_parts:
        if is_emoji_only(part):
            result.append(part)
        else:
            # Always split on sentence boundaries, regardless of total length.
            result.extend(_split_sentences_always(part, limit))
    return [r for r in result if r]


def _split_into_chunks(text: str, limit: int = _XTTS_CHUNK_CHARS) -> list[str]:
    """Split text into TTS-safe chunks, breaking at sentence boundaries.

    Each returned chunk is guaranteed to be ≤ limit characters.
    If a single sentence exceeds limit it is further split at word/char boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    # Split on sentence endings first
    raw_sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

    chunks: list[str] = []
    current = ""

    for sent in raw_sentences:
        # If a single sentence is already over the limit, hard-split it
        if len(sent) > limit:
            if current:
                chunks.append(current)
                current = ""
            # Split oversized sentence at word/char boundaries
            while sent:
                space_idx = sent.rfind(" ", 0, limit)
                cut = space_idx if space_idx > limit // 2 else limit
                chunks.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            continue

        # Accumulate sentences until we approach the limit
        joined = (current + " " + sent).strip() if current else sent
        if len(joined) <= limit:
            current = joined
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return [c for c in chunks if c]


def _concat_wavs(paths: list[str], output_path: str) -> bool:
    """Concatenate multiple WAV files into one using numpy + wave."""
    try:
        import wave

        import numpy as np

        frames_list: list[bytes] = []
        params = None

        for p in paths:
            with wave.open(p, "rb") as wf:
                if params is None:
                    params = wf.getparams()
                frames_list.append(wf.readframes(wf.getnframes()))

        with wave.open(output_path, "wb") as out:
            out.setparams(params)  # type: ignore[arg-type]
            for frames in frames_list:
                out.writeframes(frames)

        return True
    except Exception:
        logger.exception("WAV concatenation failed")
        return False


def tts_to_wav(
    text: str,
    language: Literal["en", "zh", "zh-cn"] = "en",
    speaker_wav: str | None = None,
) -> str | None:
    """Convert text to speech and save to a temp WAV file.

    Long text is automatically split into XTTS-safe chunks (≤ 250 chars each),
    synthesised in sequence, then concatenated into a single output WAV.

    Speaker resolution order:
      1. speaker_wav argument (caller override)
      2. SPEAKER_WAV_PATH from .env — if the file exists, clone that voice
      3. SPEAKER_NAME from .env — use a named built-in XTTS-v2 speaker
      4. Hardcoded default (_DEFAULT_SPEAKER = "Claribel Dervla")

    Returns the path to the WAV file, or None on failure.
    The caller is responsible for deleting the temp file after use.
    """
    tts = get_tts()
    if tts is None:
        return None

    # Strip emoji characters before synthesis — XTTS-v2 will literally speak
    # emoji names (e.g. "smiling face") if they reach the model.
    text = _EMOJI_SPLIT_RE.sub('', text).strip()
    if not text:
        return None

    settings = get_settings()

    # XTTS-v2 uses "zh-cn" for Mandarin Chinese
    tts_lang = "zh-cn" if language.startswith("zh") else language

    # --- Resolve speaker once ---
    wav_file = speaker_wav or settings.speaker_wav_path
    speaker_path = Path(wav_file)
    speed = settings.speech_speed if settings.speech_speed > 0 else 1.0
    emotion = (settings.speech_emotion or "").strip() or None

    use_voice_clone = speaker_path.exists()
    named_speaker = _DEFAULT_SPEAKER
    if use_voice_clone:
        logger.info("TTS using voice clone from %s (speed=%.2f)", wav_file, speed)
    else:
        named_speaker = (settings.speaker_name or "").strip() or _DEFAULT_SPEAKER
        logger.info(
            "Speaker WAV not found at %s — using built-in speaker: %s (speed=%.2f)",
            wav_file, named_speaker, speed,
        )

    def _synth_chunk(chunk_text: str, out_path: str) -> bool:
        try:
            if use_voice_clone:
                tts.tts_to_file(
                    text=chunk_text,
                    speaker_wav=str(speaker_path),
                    language=tts_lang,
                    emotion=emotion,
                    file_path=out_path,
                    speed=speed,
                )
            else:
                tts.tts_to_file(
                    text=chunk_text,
                    speaker=named_speaker,
                    language=tts_lang,
                    emotion=emotion,
                    file_path=out_path,
                    speed=speed,
                )
            return True
        except Exception:
            logger.exception("TTS chunk synthesis failed for: %r", chunk_text[:60])
            return False

    chunks = _split_into_chunks(text)
    logger.info("TTS: %d chunk(s) for %d chars", len(chunks), len(text))

    if len(chunks) == 1:
        # Fast path — single chunk, no concatenation needed
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            output_path = tmp.name
            if _synth_chunk(chunks[0], output_path):
                logger.info("TTS output written to %s", output_path)
                return output_path
            os.unlink(output_path)
            return None
        except Exception:
            logger.exception("TTS synthesis failed")
            return None

    # Multi-chunk path — synthesise each, then concatenate
    chunk_paths: list[str] = []
    try:
        for i, chunk in enumerate(chunks):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            chunk_path = tmp.name
            if not _synth_chunk(chunk, chunk_path):
                logger.error("TTS failed on chunk %d/%d — aborting", i + 1, len(chunks))
                return None
            chunk_paths.append(chunk_path)
            logger.debug("TTS chunk %d/%d done: %s", i + 1, len(chunks), chunk_path)

        # Concatenate all chunks into a single output WAV
        out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_tmp.close()
        output_path = out_tmp.name

        if _concat_wavs(chunk_paths, output_path):
            logger.info(
                "TTS concatenated %d chunks → %s (%d chars total)",
                len(chunks), output_path, len(text),
            )
            return output_path

        os.unlink(output_path)
        return None

    except Exception:
        logger.exception("TTS multi-chunk synthesis failed")
        return None
    finally:
        for p in chunk_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
