"""Qwen3-TTS via mlx-audio — on-demand inference, no permanent RAM residency.

Replaces XTTS-v2 as the TTS backend.  Key improvements over XTTS-v2:
  - Natively bilingual zh/en (Alibaba model, trained on Chinese from scratch)
  - No 82-char Chinese token limit — handles full sentences cleanly
  - MLX-native: runs entirely on Apple Silicon unified memory, no PyTorch
  - Voice cloning from a 3-second WAV reference (same SPEAKER_WAV_PATH flow)
  - Falls back to a built-in Qwen3-TTS voice when no reference WAV is present

Model loaded on first call, kept in a module-level singleton for the process
lifetime (similar to how XTTS-v2 was handled — warm-up cost is paid once).

Speaker resolution order (same as XTTS-v2):
  1. SPEAKER_WAV_PATH from .env — if the file exists, clone that voice
  2. TTS_VOICE from .env — a named built-in Qwen3-TTS voice
     Available voices (CustomVoice model): Chelsie, Ethan, Vivian, …
  3. Hardcoded default: "Chelsie" (clear, neutral female voice)
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from ella.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_model_name: str | None = None

# Default built-in voice when no WAV reference is provided.
_DEFAULT_VOICE = "Chelsie"

def _get_emotion_profile(emotion: str | None, language: str) -> tuple[str | None, float, float]:
    """Return (instruct, tts_speed, tts_temperature) for a 27-emotion label.

    tts_speed is the per-emotion speed multiplier (applied on top of SPEECH_SPEED).
    tts_temperature is the per-emotion absolute temperature override.
    Returns (None, 1.0, -1.0) when emotion is unknown — caller uses global defaults.
    """
    if not emotion:
        return None, 1.0, -1.0
    try:
        from ella.emotion.models import EMOTION_REGISTRY
        profile = EMOTION_REGISTRY.get(emotion)
        if profile is None:
            return None, 1.0, -1.0
        instruct = profile.tts_zh if language.startswith("zh") else profile.tts_en
        return instruct, profile.tts_speed, profile.tts_temperature
    except Exception:
        return None, 1.0, -1.0

# Emoji Unicode ranges — kept identical to old xtts.py so callers work unchanged.
_EMOJI_RANGES = (
    r'\U0001F600-\U0001F64F'
    r'\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF'
    r'\U0001F700-\U0001F77F'
    r'\U0001F780-\U0001F7FF'
    r'\U0001F800-\U0001F8FF'
    r'\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F'
    r'\U0001FA70-\U0001FAFF'
    r'\u2600-\u26FF'
    r'\u2700-\u27BF'
    r'\uFE0F'
    r'\u200D'
    r'\u20E3'
)

_EMOJI_ONLY_RE = re.compile(r'^[' + _EMOJI_RANGES + r']+$')
_EMOJI_SPLIT_RE = re.compile(r'([' + _EMOJI_RANGES + r']+)')

_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?。！？])\s+'
    r'|(?<=[。！？])'
)

def is_emoji_only(text: str) -> bool:
    return bool(_EMOJI_ONLY_RE.match(text))


def _split_sentences_always(text: str) -> list[str]:
    """Split text at every sentence boundary — always, regardless of length."""
    text = text.strip()
    if not text:
        return []
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return raw if raw else ([text] if text else [])


def split_into_sentences(text: str) -> list[str]:
    """Split reply text into TTS-sized tokens and emoji tokens.

    Splits on sentence-ending punctuation (。！？.!?) only. Each resulting
    sentence is passed whole to tts_to_wav so TTS can render the natural
    emotional arc of the full sentence. Emoji runs stay as discrete plain-text
    tokens in their correct position.
    """
    raw_parts = [p.strip() for p in _EMOJI_SPLIT_RE.split(text) if p.strip()]
    result: list[str] = []
    for part in raw_parts:
        if is_emoji_only(part):
            result.append(part)
        else:
            result.extend(_split_sentences_always(part))
    return [r for r in result if r]


def get_tts():
    """Return the loaded Qwen3-TTS model singleton (warm up on first call)."""
    global _model, _model_name
    settings = get_settings()
    target = settings.tts_model
    if _model is None or _model_name != target:
        _model = _load_model(target)
        _model_name = target
    return _model


def _patch_max_tokens_cap(model) -> None:
    """Patch mlx-audio's internal max_tokens cap.

    mlx-audio's _generate_with_instruct hard-codes:
        effective_max_tokens = min(max_tokens, max(75, target_token_count * 6))

    The *6 factor is too conservative for emotional Chinese speech — the model
    inserts 1-2 second silence gaps between comma-separated clauses, and those
    silence tokens also count against the budget.  We replace *6 with *15 so the
    model always has room to finish the full sentence naturally.

    This is a targeted monkey-patch of one line.  If mlx-audio is upgraded and
    the attribute no longer exists, the patch logs a warning and skips silently.
    """
    import types, inspect
    try:
        cls = type(model)
        original_fn = cls._generate_with_instruct
        src = inspect.getsource(original_fn)

        # Only patch if the original cap is still present
        if "target_token_count * 6" not in src:
            logger.info("Qwen3-TTS max_tokens patch: cap not found (already patched or API changed), skipping.")
            return

        patched_src = src.replace("target_token_count * 6", "target_token_count * 15")

        # Compile in a fresh namespace that mirrors the original module's globals
        orig_module = inspect.getmodule(original_fn)
        ns = vars(orig_module).copy()
        # Strip the leading indentation so exec can compile it
        import textwrap
        patched_src = textwrap.dedent(patched_src)
        exec(compile(patched_src, "<patched>", "exec"), ns)  # noqa: S102
        new_fn = ns["_generate_with_instruct"]
        cls._generate_with_instruct = new_fn
        logger.info("Qwen3-TTS max_tokens patch applied: token cap multiplier 6 → 15.")
    except Exception:
        logger.warning("Qwen3-TTS max_tokens patch failed (non-critical).", exc_info=True)


def _load_model(model_name: str):
    try:
        from mlx_audio.tts.utils import load_model
        logger.info("Loading Qwen3-TTS model: %s", model_name)
        m = load_model(model_name)
        _patch_max_tokens_cap(m)
        logger.info("Qwen3-TTS ready.")
        return m
    except ImportError:
        logger.error("mlx-audio not installed. Run: pip install mlx-audio")
        return None
    except Exception:
        logger.exception("Failed to load Qwen3-TTS model: %s", model_name)
        return None


def tts_to_wav(
    text: str,
    language: Literal["en", "zh", "zh-cn"] = "en",
    speaker_wav: str | None = None,
    emotion: str | None = None,
) -> str | None:
    """Synthesise *text* to a temp WAV file and return its path.

    The caller is responsible for deleting the file after use.
    Returns None on failure.

    Model routing (determined by TTS_MODEL in .env):
      CustomVoice — consistent named speaker (e.g. serena) + instruct for emotion.
                    This is the recommended mode: voice is always the same person,
                    and instruct fully controls emotion delivery.
                    NOTE: speed param is NOT supported on CustomVoice — speed cues
                    are embedded in the instruct string instead.

      Base        — voice cloning from SPEAKER_WAV_PATH.
                    instruct is NOT supported on Base model (silently ignored).
                    Emotion is expressed only through text semantics.

      VoiceDesign — instruct describes the full voice from scratch.
                    No consistent speaker identity across calls.
    """
    model = get_tts()
    if model is None:
        return None

    # Strip emoji — they should never reach the TTS model.
    text = _EMOJI_SPLIT_RE.sub('', text).strip()
    if not text:
        return None

    # Strip history fingerprint stubs that the LLM occasionally echoes verbatim.
    # These look like [opens:"..."] and must never reach the TTS model.
    text = re.sub(r'\[opens:[^\]]*\]', '', text).strip()

    # Normalise characters that cause the model to emit an early EOS.
    # Qwen3-TTS treats whitespace as a sentence boundary — any space mid-sentence
    # causes it to stop generating after that point.
    # ① Replace ～/~ with ，(pause, not a stop)
    text = text.replace('～', '，').replace('~', '，')
    # ② Strip all spaces — Chinese TTS doesn't need them and spaces trigger EOS
    text = text.replace(' ', '').replace('\u3000', '')
    # ③ Collapse any double punctuation left by the substitutions
    text = re.sub(r'，{2,}', '，', text).strip()

    if not text:
        return None

    settings = get_settings()

    # Qwen3-TTS uses plain "zh" for Mandarin (no "zh-cn" variant needed)
    lang_code = "zh" if language.startswith("zh") else "en"

    # Detect model type from model name
    is_custom_voice = "CustomVoice" in settings.tts_model
    is_voice_design = "VoiceDesign" in settings.tts_model
    # Base model = anything that is neither CustomVoice nor VoiceDesign

    # Resolve per-emotion instruct + generation params
    instruct, emotion_speed_mult, emotion_temperature = _get_emotion_profile(emotion, lang_code)
    if not instruct:
        instruct = (settings.speech_instruct or "").strip() or None

    # Temperature: per-emotion absolute value overrides global setting
    temperature = emotion_temperature if emotion_temperature > 0 else settings.tts_temperature

    # Speed: CustomVoice ignores the speed param — bake it into instruct instead.
    # Base/VoiceDesign honour the speed param.
    base_speed = settings.speech_speed if settings.speech_speed > 0 else 1.0
    speed = max(base_speed * emotion_speed_mult, 0.5)

    if is_custom_voice and instruct and emotion_speed_mult != 1.0:
        # Append a speed directive to the instruct so the model hears it.
        if emotion_speed_mult < 0.85:
            speed_hint = "，语速非常慢" if lang_code == "zh" else ", speak very slowly"
        elif emotion_speed_mult < 0.95:
            speed_hint = "，语速偏慢" if lang_code == "zh" else ", speak slowly"
        elif emotion_speed_mult > 1.10:
            speed_hint = "，语速很快" if lang_code == "zh" else ", speak quickly"
        elif emotion_speed_mult > 1.02:
            speed_hint = "，语速略快" if lang_code == "zh" else ", speak at a brisk pace"
        else:
            speed_hint = ""
        if speed_hint:
            instruct = instruct.rstrip("，。, ") + speed_hint

    # ── Voice source ──────────────────────────────────────────────────────────
    voice = (settings.tts_voice or "").strip() or _DEFAULT_VOICE

    logger.info(
        "Qwen3-TTS [%s]: voice=%s | emotion=%s | temp=%.2f | instruct=%r",
        "CustomVoice" if is_custom_voice else ("VoiceDesign" if is_voice_design else "Base"),
        voice, emotion or "none", temperature, instruct,
    )

    # Calculate a text-length-aware LM token budget.
    # Qwen3-TTS uses a SNAC codec with hop_length=441 and 4 codebooks, giving
    # an LM token rate of ~218 tokens/second of audio. Empirically Chinese
    # speech runs at ~0.30s/char → ~65 LM tokens/char. We use 2× headroom
    # so the model always finishes the sentence before hitting the budget.
    is_chinese = lang_code == "zh"
    lm_tokens_per_char = 65 if is_chinese else 45
    max_tokens = max(500, min(int(len(text) * lm_tokens_per_char * 2), 20000))

    try:
        if is_custom_voice:
            # CustomVoice: fixed speaker + instruct. No ref_audio, no speed param.
            results = list(model.generate(
                text=text,
                voice=voice,
                lang_code=lang_code,
                instruct=instruct,
                temperature=temperature,
                top_k=settings.tts_top_k,
                top_p=settings.tts_top_p,
                repetition_penalty=settings.tts_repetition_penalty,
                max_tokens=max_tokens,
                verbose=False,
                stream=False,
            ))
        elif is_voice_design:
            # VoiceDesign: instruct IS the voice description — must not be None.
            results = list(model.generate(
                text=text,
                lang_code=lang_code,
                instruct=instruct or "Warm, natural female voice",
                temperature=temperature,
                top_k=settings.tts_top_k,
                top_p=settings.tts_top_p,
                repetition_penalty=settings.tts_repetition_penalty,
                max_tokens=max_tokens,
                verbose=False,
                stream=False,
            ))
        else:
            # Base model: voice cloning via ref_audio. instruct is NOT supported.
            wav_file = speaker_wav or settings.speaker_wav_path
            speaker_path = Path(wav_file)
            ref_audio_array = None
            ref_text_val: str | None = (settings.speaker_ref_text or "").strip() or None

            if speaker_path.exists():
                from mlx_audio.utils import load_audio
                ref_audio_array = load_audio(
                    str(speaker_path),
                    sample_rate=model.sample_rate,
                    volume_normalize=False,
                )

            gen_kwargs: dict = dict(
                text=text,
                speed=speed,
                lang_code=lang_code,
                temperature=temperature,
                top_k=settings.tts_top_k,
                top_p=settings.tts_top_p,
                repetition_penalty=settings.tts_repetition_penalty,
                max_tokens=max_tokens,
                verbose=False,
                stream=False,
            )
            if ref_audio_array is not None:
                gen_kwargs["ref_audio"] = ref_audio_array
                gen_kwargs["ref_text"] = ref_text_val
            else:
                gen_kwargs["voice"] = voice

            results = list(model.generate(**gen_kwargs))

        if not results:
            logger.warning("Qwen3-TTS returned no audio for: %r", text[:60])
            return None

        # Concatenate all result segments into one WAV
        audio_chunks = [np.array(r.audio) for r in results]
        audio = np.concatenate(audio_chunks, axis=0) if len(audio_chunks) > 1 else audio_chunks[0]
        sample_rate = results[0].sample_rate

        # Append 300 ms of silence so the final syllable's natural decay is
        # never clipped by audio player buffer underrun or codec frame boundaries.
        silence_samples = int(sample_rate * 0.30)
        audio = np.concatenate([audio, np.zeros(silence_samples, dtype=audio.dtype)])

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        out_path = tmp.name

        _write_wav(out_path, audio, sample_rate)
        audio_duration_s = len(audio) / sample_rate
        logger.info(
            "Qwen3-TTS WAV written: %s (%d bytes, %.1fs)",
            out_path, Path(out_path).stat().st_size,
            audio_duration_s,
        )

        # #region debug log — max_tokens cap patch verification
        try:
            import json as _json, time as _time, shutil as _shutil
            _DEBUG_LOG = "/Users/cl/Documents/App Project/Projects/ai.Ella/.cursor/debug-eb85ec.log"
            _ts = int(_time.time())
            _keep_path = f"/tmp/ella_debug_tts_{_ts}.wav"
            _shutil.copy2(out_path, _keep_path)
            tokens_generated = round(audio_duration_s * 12.5)  # 12.5 Hz codec rate
            with open(_DEBUG_LOG, "a") as _f:
                _f.write(_json.dumps({"sessionId":"eb85ec","runId":"post-patch","hypothesisId":"max_tokens_cap","location":"qwen3.py:tts_to_wav","message":"TTS WAV — cap patch active","data":{"full_text":text,"text_len":len(text),"audio_duration_s":round(audio_duration_s,3),"tokens_generated_est":tokens_generated,"max_tokens_budget":max_tokens,"inspect_wav":_keep_path},"timestamp":_ts*1000}) + "\n")
        except Exception:
            pass
        # #endregion

        return out_path

    except Exception:
        logger.exception("Qwen3-TTS synthesis failed for: %r", text[:60])
        return None


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a float32 numpy array to a 16-bit PCM WAV file."""
    import wave
    # Clamp and convert to 16-bit PCM
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
