"""Unit test: verify reply_agent passes LLM sentences to TTS without any splitting.

What this proves:
- Each element of payload.sentences reaches tts_to_wav exactly as-is.
- No comma-splitting, no clause-splitting, no further tokenisation happens.
- The number of tts_to_wav calls equals exactly len(payload.sentences).

Run:
    cd /Users/cl/Documents/App\ Project/Projects/ai.Ella
    .venv/bin/python -m pytest tests/test_reply_agent_no_split.py -v
"""
from __future__ import annotations

import asyncio
import sys
import os
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Minimal stubs so reply_agent imports without Telegram/Redis/MLX ──────────

def _make_stubs():
    """Patch heavy dependencies before importing reply_agent."""
    # ella.config
    cfg = types.ModuleType("ella.config")
    settings = MagicMock()
    settings.telegram_bot_token = "fake"
    cfg.get_settings = lambda: settings
    sys.modules.setdefault("ella.config", cfg)

    # ella.communications.telegram.sender
    sender_mod = types.ModuleType("ella.communications.telegram.sender")
    sender_mod.get_sender = MagicMock()
    sys.modules.setdefault("ella.communications.telegram.sender", sender_mod)

    # ella.agents.base_agent
    base = types.ModuleType("ella.agents.base_agent")
    class BaseAgent: pass
    base.BaseAgent = BaseAgent
    sys.modules.setdefault("ella.agents.base_agent", base)

    # ella.agents.protocol
    proto = types.ModuleType("ella.agents.protocol")
    from dataclasses import dataclass, field as dc_field
    @dataclass
    class ReplyPayload:
        text: str = ""
        sentences: list = dc_field(default_factory=list)
        emojis: list = dc_field(default_factory=list)
        detail_text: str = ""
        language: str = "zh"
        emotion: str | None = None
    class HandoffMessage:
        def __init__(self, payload, session):
            self.payload = payload
            self.session = session
    class UserTask: pass
    proto.ReplyPayload = ReplyPayload
    proto.HandoffMessage = HandoffMessage
    proto.UserTask = UserTask
    sys.modules.setdefault("ella.agents.protocol", proto)

    # ella.tts.qwen3 — stub tts_to_wav, split_into_sentences, is_emoji_only
    tts_mod = types.ModuleType("ella.tts.qwen3")
    tts_mod.tts_to_wav = MagicMock(return_value=None)
    tts_mod.split_into_sentences = lambda text: [text]  # no-op
    tts_mod.is_emoji_only = lambda t: False
    sys.modules.setdefault("ella.tts.qwen3", tts_mod)

    # ella.memory stubs
    for mod_name in [
        "ella.memory", "ella.memory.knowledge", "ella.memory.goal",
        "ella.memory.focus",
    ]:
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    return tts_mod, sender_mod


TTS_MOD, SENDER_MOD = _make_stubs()

# Now import the real reply_agent from disk
import importlib
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ella.agents.reply_agent",
    "/Users/cl/Documents/App Project/Projects/ai.Ella/ella/agents/reply_agent.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ReplyAgent = _mod.ReplyAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session(chat_id=12345):
    s = MagicMock()
    s.chat_id = chat_id
    return s


def _make_payload(sentences: list[str], language: str = "zh", emotion: str | None = None):
    proto = sys.modules["ella.agents.protocol"]
    return proto.ReplyPayload(
        text=" ".join(sentences),
        sentences=sentences,
        language=language,
        emotion=emotion,
    )


def _make_handoff(sentences, **kw):
    proto = sys.modules["ella.agents.protocol"]
    payload = _make_payload(sentences, **kw)
    return proto.HandoffMessage(payload=payload, session=_make_session())


async def _run_handle(sentences, **kw):
    """Run ReplyAgent.handle and return the list of texts sent to tts_to_wav."""
    TTS_MOD.tts_to_wav.reset_mock()
    TTS_MOD.tts_to_wav.return_value = None  # no WAV → voice send skipped

    sender = AsyncMock()
    sender.send_chat_action = AsyncMock()
    sender.send_voice = AsyncMock()
    sender.send_message = AsyncMock()
    SENDER_MOD.get_sender.return_value = sender

    agent = ReplyAgent()
    msg = _make_handoff(sentences, **kw)
    await agent.handle(msg)

    # Collect every text argument passed to tts_to_wav
    return [call.args[0] for call in TTS_MOD.tts_to_wav.call_args_list]


# ── Tests ─────────────────────────────────────────────────────────────────────

import random
import string

def _rand_zh(n_clauses: int) -> str:
    """Generate a random Chinese-like sentence with n_clauses comma-separated clauses."""
    chars = "的了在是我有和人这中大来上国个到说们为子和地出道也时年得就那要下以生会自着去之过家学对可她他"
    clauses = ["".join(random.choices(chars, k=random.randint(4, 8))) for _ in range(n_clauses)]
    return "，".join(clauses) + "。"


def _rand_en(n_words: int) -> str:
    """Generate a random English sentence."""
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 7)))
             for _ in range(n_words)]
    return " ".join(words) + "."


def test_single_sentence_no_split():
    """One LLM sentence with multiple commas → exactly 1 tts_to_wav call, text unchanged."""
    sentence = _rand_zh(n_clauses=5)
    texts = asyncio.run(_run_handle([sentence]))
    assert len(texts) == 1, f"Expected 1 TTS call, got {len(texts)}: {texts}"
    assert texts[0] == sentence.strip(), f"Text was mutated:\n  in:  {sentence!r}\n  out: {texts[0]!r}"


def test_two_sentences_no_split():
    """Two LLM sentences → exactly 2 tts_to_wav calls, each text unchanged."""
    s1 = _rand_zh(n_clauses=4)
    s2 = _rand_zh(n_clauses=4)
    texts = asyncio.run(_run_handle([s1, s2]))
    assert len(texts) == 2, f"Expected 2 TTS calls, got {len(texts)}: {texts}"
    assert texts[0] == s1.strip(), f"Sentence 0 mutated:\n  in:  {s1!r}\n  out: {texts[0]!r}"
    assert texts[1] == s2.strip(), f"Sentence 1 mutated:\n  in:  {s2!r}\n  out: {texts[1]!r}"


def test_commas_not_split():
    """Sentences with many commas must NOT be broken on comma boundaries."""
    # 8 clauses = 7 commas — would produce 8 chunks if comma-splitting were active
    sentence = _rand_zh(n_clauses=8)
    texts = asyncio.run(_run_handle([sentence]))
    assert len(texts) == 1, (
        f"Comma-split happened — got {len(texts)} calls instead of 1.\n"
        f"Input: {sentence!r}\nOutputs: {texts}"
    )
    assert texts[0] == sentence.strip()


def test_tts_call_count_equals_sentence_count():
    """tts_to_wav call count must always equal len(payload.sentences), for 1–4 sentences."""
    for n in [1, 2, 3, 4]:
        # Each sentence has 3–6 commas to maximally stress any clause-splitting logic
        sentences = [_rand_zh(n_clauses=random.randint(3, 6)) for _ in range(n)]
        texts = asyncio.run(_run_handle(sentences))
        assert len(texts) == n, (
            f"For {n} LLM sentences expected {n} TTS calls, got {len(texts)}.\n"
            f"Inputs: {sentences}\nOutputs: {texts}"
        )


def test_english_sentence_no_split():
    """English sentence with many words must reach TTS unchanged."""
    sentence = _rand_en(n_words=12)
    texts = asyncio.run(_run_handle([sentence], language="en"))
    assert len(texts) == 1, f"Expected 1 TTS call, got {len(texts)}: {texts}"
    assert texts[0] == sentence.strip()


if __name__ == "__main__":
    # Run directly without pytest
    tests = [
        test_single_sentence_no_split,
        test_two_sentences_no_split,
        test_commas_not_split,
        test_tts_call_count_equals_sentence_count,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
