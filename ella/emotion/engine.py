"""Emotion engine — contagion, self-update, and decay.

Implements the Bosse / Pereira-Paiva Generic Contagion Model (ACII 2011)
across all three PAD dimensions (Pleasure-Arousal-Dominance), with:
  - Per-emotion-category susceptibility from Doherty's ECS (1997)
  - Dominance preservation via dominanceBase + resilience
  - Emotion label resolved by nearest-neighbour in PAD space (no hardcoded mappings)
  - Decay toward baseline (calmness) every 4 hours via _decay_loop in main.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ella.emotion.models import (
    BASELINE,
    EMOTION_REGISTRY,
    AgentState,
    PersonalityTraits,
    UserState,
    nearest_emotion,
    resolve_ecs_susceptibility,
)
from ella.emotion.store import get_emotion_store

logger = logging.getLogger(__name__)

# Decay rate per cycle (4-hour background loop).
# At 0.12, the state covers ~88% of the distance to baseline per 24h.
_DECAY_RATE = 0.12


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def read_agent_state(chat_id: int) -> AgentState:
    """Load Ella's current emotional state for this chat, or return baseline."""
    store = get_emotion_store()
    state = await store.read_agent_state(chat_id)
    if state is None:
        state = AgentState(
            valence=BASELINE.valence,
            energy=BASELINE.energy,
            dominance=BASELINE.dominance,
            emotion=BASELINE.emotion,
            intensity=BASELINE.intensity,
            momentum=BASELINE.momentum,
        )
    return state


async def read_user_state(chat_id: int) -> UserState:
    """Load the last detected user emotional state, or return neutral default."""
    store = get_emotion_store()
    state = await store.read_user_state(chat_id)
    if state is None:
        state = UserState()
    return state


async def apply_contagion(
    chat_id: int,
    user_state: UserState,
    personality: PersonalityTraits,
) -> AgentState:
    """Apply user emotion contagion to Ella's state (Bosse/Pereira-Paiva model).

    Updates all three PAD dimensions and resolves a new emotion label by
    nearest-neighbour lookup. Persists the updated state to MySQL.
    """
    store = get_emotion_store()
    agent = await read_agent_state(chat_id)

    # 1. Resolve ECS susceptibility for the user's emotion category
    s_ecs = resolve_ecs_susceptibility(user_state.emotion, personality.ecs)
    alpha = personality.expressiveness   # sender expressiveness (Bosse α)

    # 2. Base pull — S_ecs × volatility × user intensity
    pull = s_ecs * personality.volatility * user_state.intensity

    # 3. Valence update — resilience attenuates negative pulls only
    val_delta = user_state.valence - agent.valence
    if val_delta < 0:
        effective_pull = pull * (1.0 - personality.resilience * 0.8)
    else:
        effective_pull = pull
    new_valence = _clamp(
        agent.valence + alpha * effective_pull * val_delta + personality.optimism_bias,
        -1.0, 1.0,
    )

    # 4. Energy update (weaker coupling than valence — 0.7 factor)
    eng_delta = user_state.energy - agent.energy
    new_energy = _clamp(
        agent.energy + alpha * (pull * 0.7) * eng_delta,
        0.0, 1.0,
    )

    # 5. Dominance update — resilience and dominanceBase resist collapse
    dom_pull = pull * 0.5                              # weaker coupling than valence
    dom_protection = personality.resilience * 0.6
    new_dominance = _clamp(
        agent.dominance
        + alpha * dom_pull * (1.0 - dom_protection) * (user_state.dominance - agent.dominance)
        + 0.05 * (personality.dominance_base - agent.dominance),  # drift back toward base
        0.0, 1.0,
    )

    # 6. Intensity — blend toward user intensity, modulated by pull
    new_intensity = _clamp(
        agent.intensity + pull * (user_state.intensity - agent.intensity),
        0.0, 1.0,
    )

    # 7. Resolve emotion label by nearest-neighbour in PAD space
    new_emotion = nearest_emotion(new_valence, new_energy, new_dominance)

    # 8. Momentum from the new emotion profile
    new_momentum = EMOTION_REGISTRY[new_emotion].momentum if new_emotion in EMOTION_REGISTRY else 0.5

    new_state = AgentState(
        valence=new_valence,
        energy=new_energy,
        dominance=new_dominance,
        emotion=new_emotion,
        intensity=new_intensity,
        momentum=new_momentum,
        updated_at=_now_str(),
        session_count=agent.session_count,
    )

    await store.upsert_agent_state(chat_id, new_state)
    await store.append_history(
        chat_id, "contagion", new_state,
        trigger=user_state.emotion,
        note=f"user: {user_state.emotion} v={user_state.valence:.2f} "
             f"e={user_state.energy:.2f} i={user_state.intensity:.2f}",
    )

    logger.info(
        "[Emotion] contagion chat_id=%d: %s → %s (v=%.2f e=%.2f d=%.2f)",
        chat_id, agent.emotion, new_emotion, new_valence, new_energy, new_dominance,
    )
    return new_state


async def apply_self_update(
    chat_id: int,
    emotion_label: str,
    trigger: str = "",
) -> AgentState:
    """Update Ella's state to reflect the LLM's self-assessed emotion for this turn.

    Looks up the EmotionProfile for the given label and uses those PAD coordinates
    directly — weighted blend with the existing state to avoid jarring jumps.
    """
    store = get_emotion_store()
    agent = await read_agent_state(chat_id)

    profile = EMOTION_REGISTRY.get(emotion_label)
    if profile is None:
        logger.warning("[Emotion] Unknown emotion label %r — skipping self-update", emotion_label)
        return agent

    # Blend: 60% new profile, 40% existing state — prevents sudden discontinuities
    blend = 0.6
    new_valence   = _clamp(blend * profile.valence   + (1 - blend) * agent.valence,   -1.0, 1.0)
    new_energy    = _clamp(blend * profile.energy    + (1 - blend) * agent.energy,     0.0, 1.0)
    new_dominance = _clamp(blend * profile.dominance + (1 - blend) * agent.dominance,  0.0, 1.0)
    new_intensity = _clamp(blend * profile.intensity + (1 - blend) * agent.intensity,  0.0, 1.0)

    # Resolve nearest label for the blended coordinates (may differ from input label
    # if the blend shifted the PAD point toward a neighbouring emotion)
    resolved_emotion = nearest_emotion(new_valence, new_energy, new_dominance)

    new_state = AgentState(
        valence=new_valence,
        energy=new_energy,
        dominance=new_dominance,
        emotion=resolved_emotion,
        intensity=new_intensity,
        momentum=profile.momentum,
        updated_at=_now_str(),
        session_count=agent.session_count + 1,
    )

    await store.upsert_agent_state(chat_id, new_state)
    await store.append_history(
        chat_id, "self", new_state,
        trigger=emotion_label,
        note=trigger[:255] if trigger else "",
    )

    logger.info(
        "[Emotion] self-update chat_id=%d: %s → %s (v=%.2f e=%.2f d=%.2f)",
        chat_id, agent.emotion, resolved_emotion, new_valence, new_energy, new_dominance,
    )
    return new_state


async def apply_decay(
    chat_id: int,
    personality: PersonalityTraits,
) -> AgentState:
    """Drift Ella's emotional state toward baseline (calmness).

    Called by the 4-hour background loop in main.py. The rate of drift is
    scaled by (1 - momentum) so high-momentum emotions (calmness, nostalgia)
    resist decay while low-momentum ones (surprise, horror) decay quickly.
    """
    store = get_emotion_store()
    agent = await read_agent_state(chat_id)

    # Momentum-scaled decay rate: high momentum → slower decay
    effective_rate = _DECAY_RATE * (1.0 - agent.momentum * 0.6)

    new_valence   = _clamp(agent.valence   + effective_rate * (BASELINE.valence   - agent.valence),   -1.0, 1.0)
    new_energy    = _clamp(agent.energy    + effective_rate * (BASELINE.energy    - agent.energy),     0.0, 1.0)
    new_dominance = _clamp(
        agent.dominance + effective_rate * (personality.dominance_base - agent.dominance),
        0.0, 1.0,
    )
    new_intensity = _clamp(agent.intensity + effective_rate * (BASELINE.intensity - agent.intensity), 0.0, 1.0)

    # Add optimism bias during decay so Ella doesn't drift into negativity
    new_valence = _clamp(new_valence + personality.optimism_bias * effective_rate, -1.0, 1.0)

    new_emotion = nearest_emotion(new_valence, new_energy, new_dominance)
    new_momentum = EMOTION_REGISTRY[new_emotion].momentum if new_emotion in EMOTION_REGISTRY else 0.5

    new_state = AgentState(
        valence=new_valence,
        energy=new_energy,
        dominance=new_dominance,
        emotion=new_emotion,
        intensity=new_intensity,
        momentum=new_momentum,
        updated_at=_now_str(),
        session_count=agent.session_count,
    )

    await store.upsert_agent_state(chat_id, new_state)
    await store.append_history(
        chat_id, "decay", new_state,
        trigger="decay",
        note=f"rate={effective_rate:.3f}",
    )

    logger.debug(
        "[Emotion] decay chat_id=%d: %s → %s (v=%.2f e=%.2f d=%.2f)",
        chat_id, agent.emotion, new_emotion, new_valence, new_energy, new_dominance,
    )
    return new_state
