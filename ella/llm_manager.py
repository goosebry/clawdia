"""Shared LLM singleton with an async mutex.

Ella runs one Qwen3-14B model used by both BrainAgent (chat) and LearnSkill
(analysis). Loading it twice simultaneously causes GPU OOM. This module
provides a single shared instance and an asyncio lock so only one consumer
holds the model at a time.

Usage:
    async with llm_lock():
        model, tokenizer = await load_llm()
        # ... use model ...
        # model stays loaded — next consumer waits at lock boundary
    # lock released — next consumer can acquire

    # To explicitly unload (e.g. before a long idle period):
    unload_llm()

The lock is reentrant-safe: BrainAgent acquires it for a turn; LearnSkill
acquires it for the full analyse phase. They never overlap.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_model: Any = None
_tokenizer: Any = None
_lock = asyncio.Lock()
_current_holder: str = ""   # for debug logging


def is_loaded() -> bool:
    return _model is not None


def get_model_and_tokenizer() -> tuple[Any, Any]:
    """Return (model, tokenizer) — None, None if not yet loaded."""
    return _model, _tokenizer


async def load_llm() -> tuple[Any, Any]:
    """Load the LLM if not already in memory. Returns (model, tokenizer).

    Must be called while holding llm_lock().
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    try:
        from mlx_lm import load
        from ella.config import get_settings
        settings = get_settings()
        logger.info("[LLMManager] Loading %s", settings.mlx_chat_model)
        _model, _tokenizer = load(settings.mlx_chat_model)
        logger.info("[LLMManager] Model ready")
        return _model, _tokenizer
    except ImportError:
        logger.error("[LLMManager] mlx-lm not installed")
        return None, None
    except Exception as exc:
        logger.warning("[LLMManager] Load failed: %s", exc)
        return None, None


def unload_llm() -> None:
    """Delete the model from memory and clear the Metal GPU cache.

    Safe to call even if the model is not loaded.
    """
    global _model, _tokenizer
    if _model is None:
        return
    try:
        del _model
        del _tokenizer
        import mlx.core as mx
        mx.clear_cache()
        logger.info("[LLMManager] Model unloaded, GPU cache cleared")
    except Exception as exc:
        logger.warning("[LLMManager] Unload error: %s", exc)
    finally:
        _model = None
        _tokenizer = None


@asynccontextmanager
async def llm_lock(holder: str = "unknown"):
    """Async context manager that gates access to the shared LLM.

    Only one coroutine can be inside this block at a time. All others wait.

        async with llm_lock(holder="BrainAgent"):
            model, tok = await load_llm()
            ...   # do inference
        # lock released; model stays loaded for next holder

    The model is NOT automatically unloaded on exit — callers that know they
    won't need the model for a long time should call unload_llm() explicitly
    before releasing the lock (e.g. LearnSkill after its analyse phase, and
    BrainAgent after a skill is queued).
    """
    global _current_holder
    logger.debug("[LLMManager] %s waiting for LLM lock (held by '%s')", holder, _current_holder)
    async with _lock:
        _current_holder = holder
        logger.debug("[LLMManager] %s acquired LLM lock", holder)
        try:
            yield
        finally:
            _current_holder = ""
            logger.debug("[LLMManager] %s released LLM lock", holder)
