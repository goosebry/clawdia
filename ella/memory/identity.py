"""Ella's identity layer — loaded from ~/Ella/*.md at startup.

The files in ~/Ella/ define *who Ella is* and *how she relates to the user*.
They are distinct from The Knowledge (Qdrant) which stores what she has learned.
These files shape her behaviour at a fundamental level:

  Identity.md      — basic facts: name, age, timezone, background
  Soul.md          — personality, tone, emotional style, values
  Role.md          — how she relates to this specific user (role, address, dynamic)
  Personality.json — numeric emotion engine traits (resilience, volatility, etc.)
  Personality.md   — narrative description of traits, injected into LLM prompt

They are loaded once at startup, cached as a compiled system prompt block, and
injected into every conversation prompt as a fixed system layer — sitting below
the persona but above the conversation goal.

When any file changes, the identity cache is refreshed AND all in-memory
conversation state (sessions + goal refs) is wiped so Ella starts fresh with
her new sense of self.  The Qdrant long-term knowledge is NOT cleared — that
is what she remembers, not who she is.

Callers that own resettable state (e.g. TelegramPoller) register a reset
callback via register_reset_callback().  All callbacks are called on every
identity reload.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, NamedTuple

if TYPE_CHECKING:
    from ella.emotion.models import PersonalityTraits

logger = logging.getLogger(__name__)

# Default location — ~/Ella/
ELLA_DIR = Path.home() / "Ella"

_IDENTITY_FILE     = ELLA_DIR / "Identity.md"
_SOUL_FILE         = ELLA_DIR / "Soul.md"
_USER_FILE         = ELLA_DIR / "User.md"
_PERSONALITY_JSON  = ELLA_DIR / "Personality.json"
_PERSONALITY_MD    = ELLA_DIR / "Personality.md"


class IdentityContext(NamedTuple):
    """Parsed content of the identity files, ready to inject into prompts."""
    identity: str      # raw content of Identity.md (includes relationship with user)
    soul: str          # raw content of Soul.md
    user: str          # raw content of User.md
    prompt_block: str  # pre-compiled system prompt fragment
    personality_narrative: str   # raw content of Personality.md (injected into prompt when emotion engine is on)
    personality_traits_raw: dict  # parsed Personality.json (consumed by emotion engine)


def _read_file(path: Path) -> str:
    """Read a markdown file, stripping comment blocks (<!-- ... -->)."""
    try:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        return text.strip()
    except FileNotFoundError:
        logger.warning("Identity file not found: %s", path)
        return ""
    except Exception:
        logger.exception("Failed to read identity file: %s", path)
        return ""


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict on failure."""
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("Failed to read JSON file: %s", path)
        return {}


def _compile_prompt(identity: str, soul: str, user: str, personality_narrative: str) -> str:
    """Compile the identity files into a single system prompt block."""
    parts: list[str] = []
    if identity:
        parts.append(f"[Who Ella is — Identity & Relationship]\n{identity}")
    if soul:
        parts.append(f"[Ella's personality and soul]\n{soul}")
    if personality_narrative:
        parts.append(f"[Ella's emotional personality]\n{personality_narrative}")
    if user:
        parts.append(f"[Who the user is]\n{user}")

    # Relationship-aware tone instruction — derived from Identity.md's
    # "Relationship with User" section so the LLM calibrates warmth accordingly.
    if identity or soul:
        parts.append(
            "[Tone]\n"
            "Let the relationship described in your Identity file guide your entire tone. "
            "A close, intimate relationship (e.g. girlfriend, best friend) calls for warmth "
            "and playfulness. A professional or formal relationship calls for a measured tone. "
            "Never use emojis. Match the mood of the moment."
        )

    return "\n\n".join(parts) if parts else ""


def load_identity() -> IdentityContext:
    """Load and compile all identity files from ~/Ella/."""
    from ella.config import get_settings
    settings = get_settings()

    identity = _read_file(_IDENTITY_FILE)
    soul     = _read_file(_SOUL_FILE)
    user     = _read_file(_USER_FILE)

    # Personality files — only loaded when emotion engine is enabled
    personality_narrative: str = ""
    personality_traits_raw: dict = {}
    if settings.emotion_enabled:
        personality_narrative  = _read_file(_PERSONALITY_MD)
        personality_traits_raw = _read_json(_PERSONALITY_JSON)
        if personality_narrative:
            logger.info("Personality.md loaded (%d chars)", len(personality_narrative))
        if personality_traits_raw:
            logger.info("Personality.json loaded: %s", list(personality_traits_raw.keys()))

    block = _compile_prompt(identity, soul, user, personality_narrative)

    if block:
        logger.info(
            "Identity loaded from %s (%d chars — Identity:%d Soul:%d User:%d Personality:%d)",
            ELLA_DIR, len(block), len(identity), len(soul), len(user), len(personality_narrative),
        )
    else:
        logger.warning("No identity files found in %s — using persona defaults.", ELLA_DIR)

    return IdentityContext(
        identity=identity,
        soul=soul,
        user=user,
        prompt_block=block,
        personality_narrative=personality_narrative,
        personality_traits_raw=personality_traits_raw,
    )


# ── Personality trait helper ─────────────────────────────────────────────────

def get_personality_traits() -> "PersonalityTraits":
    """Return the parsed PersonalityTraits from the cached identity context.

    Returns defaults (from Personality.json dataclass defaults) if the file
    is not loaded or the emotion engine is disabled.
    """
    from ella.emotion.models import EcsWeights, PersonalityTraits
    ctx = get_identity()
    raw = ctx.personality_traits_raw
    if not raw:
        return PersonalityTraits()
    ecs_raw = raw.get("ecs", {})
    ecs = EcsWeights(
        happiness=float(ecs_raw.get("happiness", 0.60)),
        love=float(ecs_raw.get("love", 0.65)),
        fear=float(ecs_raw.get("fear", 0.35)),
        anger=float(ecs_raw.get("anger", 0.30)),
        sadness=float(ecs_raw.get("sadness", 0.50)),
    )
    return PersonalityTraits(
        resilience=float(raw.get("resilience", 0.5)),
        volatility=float(raw.get("volatility", 0.4)),
        expressiveness=float(raw.get("expressiveness", 0.65)),
        optimism_bias=float(raw.get("optimismBias", 0.1)),
        dominance_base=float(raw.get("dominanceBase", 0.55)),
        ecs=ecs,
    )


# ── Singleton ────────────────────────────────────────────────────────────────

_identity: IdentityContext | None = None


def get_identity() -> IdentityContext:
    """Return the cached identity context, loading it on first call."""
    global _identity
    if _identity is None:
        _identity = load_identity()
    return _identity


# ── Reset callbacks ───────────────────────────────────────────────────────────
# Owners of resettable in-memory state register an async callback here.
# Every callback is awaited when identity reloads, clearing stale state.

_reset_callbacks: list[Callable[[], Awaitable[None]]] = []


def register_reset_callback(cb: Callable[[], Awaitable[None]]) -> None:
    """Register an async callable to be invoked on every identity reload.

    The callback should clear all in-memory state that was shaped by the
    previous identity (session history, goal references, etc.).
    """
    _reset_callbacks.append(cb)


async def _run_reset_callbacks() -> None:
    for cb in _reset_callbacks:
        try:
            await cb()
        except Exception:
            logger.exception("Identity reset callback failed: %s", cb)


# ── File watcher ─────────────────────────────────────────────────────────────

async def watch_identity() -> None:
    """Async background task: reload identity and reset conversation state
    when any file in ~/Ella/ changes.

    Designed to run as a long-lived asyncio task alongside the main agent loop.
    """
    try:
        from watchfiles import awatch
    except ImportError:
        logger.warning("watchfiles not installed — identity hot-reload disabled.")
        return

    if not ELLA_DIR.exists():
        logger.warning("~/Ella/ directory not found — identity hot-reload disabled.")
        return

    # Only the 5 identity definition files should trigger a reload.
    # ~/Ella/ also contains browser_profiles/ (Chrome) and downloads/ which
    # change constantly and must not cause resets.
    _WATCHED_NAMES = {
        "Identity.md", "Soul.md", "User.md",
        "Personality.md", "Personality.json",
    }

    logger.info("Identity watcher started on: %s (watching %s only)", ELLA_DIR, _WATCHED_NAMES)
    async for changes in awatch(str(ELLA_DIR)):
        changed_files = {Path(p).name for _, p in changes}
        relevant = changed_files & _WATCHED_NAMES
        if not relevant:
            # Chrome profiles, downloads, etc. — ignore completely
            continue
        logger.info(
            "Identity files changed (%s) — reloading and resetting conversation state.",
            ", ".join(relevant),
        )
        # 1. Refresh the identity cache
        global _identity
        _identity = load_identity()
        # 2. Re-embed updated identity into Qdrant long-term memory
        try:
            from ella.memory.knowledge import refresh_identity_knowledge
            await refresh_identity_knowledge()
        except Exception:
            logger.exception("Failed to refresh identity knowledge after file change.")
        # 3. Wipe all in-memory state derived from the old identity
        await _run_reset_callbacks()
