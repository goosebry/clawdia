"""Emotion engine data structures and the 27-emotion registry.

PAD coordinates (valence/energy/dominance) sourced from the NRC VAD Lexicon.
ECS category mapping derived from Doherty's Emotional Contagion Scale (1997).

TTS instruct rule — CustomVoice model:
  • DO describe emotional tone, mood, pace, and feeling (e.g. 带着哭腔, clipped and forceful).
  • DO NOT describe physical voice texture (e.g. 声音沙哑低沉, voice cracking/hoarse) —
    those change the speaker identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


# ── Core data structures ──────────────────────────────────────────────────────

@dataclass
class EmotionProfile:
    """Full definition of a single emotion in PAD space with TTS delivery instructions."""
    label: str
    valence:    float   # PAD Pleasure:  -1.0 → +1.0
    energy:     float   # PAD Arousal:    0.0 → 1.0
    dominance:  float   # PAD Dominance:  0.0 → 1.0
    intensity:  float   # baseline intensity when this emotion is dominant
    momentum:   float   # how much this emotion resists being displaced (0.0 → 1.0)
    tts_zh: str         # Mandarin TTS delivery instruction
    tts_en: str         # English TTS delivery instruction
    # TTS generation parameters — applied on top of global defaults.
    # speed:       multiplier relative to settings.speech_speed (0.8 = 20% slower)
    # temperature: absolute override; higher = more expressive/varied prosody
    tts_speed: float = 1.0
    tts_temperature: float = 0.85


@dataclass
class AgentState:
    """Ella's current emotional state — three PAD dimensions + label."""
    valence:       float = 0.2
    energy:        float = 0.4
    dominance:     float = 0.55
    emotion:       str   = "calmness"
    intensity:     float = 0.2
    momentum:      float = 0.5
    updated_at:    str   = ""
    session_count: int   = 0


@dataclass
class UserState:
    """Last detected emotional state of the user."""
    valence:     float = 0.0
    energy:      float = 0.4
    dominance:   float = 0.5
    emotion:     str   = "calmness"
    intensity:   float = 0.0
    detected_at: str   = ""


@dataclass
class EcsWeights:
    """Per-emotion-category susceptibility weights from Doherty's ECS (1997).

    Each value represents how easily Ella absorbs that emotional category
    from the user — 0.0 = immune, 1.0 = fully contagious.
    """
    happiness: float = 0.60
    love:      float = 0.65
    fear:      float = 0.35
    anger:     float = 0.30
    sadness:   float = 0.50


@dataclass
class PersonalityTraits:
    """Machine-readable personality trait values, loaded from ~/Ella/Personality.json.

    These drive all contagion, decay, and self-update calculations.
    """
    resilience:     float      = 0.5
    volatility:     float      = 0.4
    expressiveness: float      = 0.65   # maps to α in the Bosse equation
    optimism_bias:  float      = 0.1
    dominance_base: float      = 0.55
    ecs:            EcsWeights = field(default_factory=EcsWeights)


# ── ECS category → emotion label mapping ─────────────────────────────────────
# For emotions spanning two categories (e.g. adoration), the higher ECS weight
# is used at contagion time (resolved at runtime via max()).

ECS_CATEGORY_MAP: Dict[str, str] = {
    "joy":                    "happiness",
    "amusement":              "happiness",
    "excitement":             "happiness",
    "satisfaction":           "happiness",
    "relief":                 "happiness",
    "admiration":             "happiness",
    "adoration":              "happiness",   # also in love — max() resolved at runtime
    "aesthetic_appreciation": "happiness",
    "entrancement":           "happiness",
    "awe":                    "happiness",
    "romance":                "love",
    "sexual_desire":          "love",
    "craving":                "love",
    "fear":                   "fear",
    "horror":                 "fear",
    "anxiety":                "fear",
    "awkwardness":            "fear",
    "anger":                  "anger",
    "disgust":                "anger",
    "sadness":                "sadness",
    "empathic_pain":          "sadness",
    "nostalgia":              "sadness",
    "boredom":                "sadness",
    "confusion":              "sadness",
    "interest":               "happiness",
    "calmness":               "happiness",
    "surprise":               "happiness",
}

# Emotions that span two ECS categories — both are checked and the higher wins.
ECS_DUAL_CATEGORY: Dict[str, tuple[str, str]] = {
    "adoration": ("happiness", "love"),
    "interest":  ("happiness", "sadness"),   # interest can arise from either positive or negative engagement
}


# ── 27-emotion registry ───────────────────────────────────────────────────────
# TTS instruct rules:
#   ✓ emotional tone / mood / pacing / feeling (e.g. 带着哭腔, clipped and forceful)
#   ✗ physical voice texture (e.g. 声音沙哑, hoarse, cracking) — changes speaker identity

EMOTION_REGISTRY: Dict[str, EmotionProfile] = {e.label: e for e in [
    EmotionProfile(
        label="admiration",
        valence=0.70, energy=0.50, dominance=0.65, intensity=0.55, momentum=0.65,
        tts_speed=0.97, tts_temperature=0.82,
        tts_zh="用真诚敬佩的语气说，带着发自内心的赞叹",
        tts_en="Speak with warm, genuine admiration",
    ),
    EmotionProfile(
        label="adoration",
        valence=0.85, energy=0.55, dominance=0.60, intensity=0.65, momentum=0.65,
        tts_speed=0.94, tts_temperature=0.88,
        tts_zh="用深情爱意的语气说，语速慢，每个字都像在珍视",
        tts_en="Speak with deep affection, slow and tender, as if cherishing every word",
    ),
    EmotionProfile(
        label="aesthetic_appreciation",
        valence=0.50, energy=0.35, dominance=0.55, intensity=0.50, momentum=0.70,
        tts_speed=0.90, tts_temperature=0.78,
        tts_zh="用沉醉欣赏的语气说，语速平缓，像沉浸在美好中",
        tts_en="Speak with quiet awe, unhurried, as if gently lost in something beautiful",
    ),
    EmotionProfile(
        label="amusement",
        valence=0.60, energy=0.65, dominance=0.65, intensity=0.50, momentum=0.55,
        tts_speed=1.05, tts_temperature=0.92,
        tts_zh="用忍俊不禁的语气说，轻快活泼，带着快笑出来的感觉",
        tts_en="Speak with a barely suppressed laugh, bright and playful, clearly amused",
    ),
    EmotionProfile(
        label="anger",
        valence=-0.70, energy=0.85, dominance=0.75, intensity=0.75, momentum=0.40,
        tts_speed=1.08, tts_temperature=0.95,
        tts_zh="用愤怒克制的语气说，咬字有力，语速急促，像在压着火",
        tts_en="Speak with controlled anger, clipped and forceful, each word bitten off",
    ),
    EmotionProfile(
        label="anxiety",
        valence=-0.50, energy=0.70, dominance=0.25, intensity=0.60, momentum=0.45,
        tts_speed=1.04, tts_temperature=0.96,
        tts_zh="用紧张不安的语气说，语速时快时停，带着藏不住的担忧",
        tts_en="Speak anxiously, pace uneven, as if nervously holding something back",
    ),
    EmotionProfile(
        label="awe",
        valence=0.50, energy=0.50, dominance=0.40, intensity=0.65, momentum=0.60,
        tts_speed=0.88, tts_temperature=0.84,
        tts_zh="用敬畏屏息的语气说，语速慢，音量轻，像被某种宏大震撼到",
        tts_en="Speak in hushed awe, slow and quiet, as if breathless before something vast",
    ),
    EmotionProfile(
        label="awkwardness",
        valence=-0.30, energy=0.55, dominance=0.30, intensity=0.45, momentum=0.40,
        tts_speed=0.93, tts_temperature=0.90,
        tts_zh="用不自在的语气说，语速略不均匀，带着明显的停顿，像在挑词",
        tts_en="Speak awkwardly, hesitant with noticeable pauses, as if searching for words",
    ),
    EmotionProfile(
        label="boredom",
        valence=-0.20, energy=0.20, dominance=0.45, intensity=0.25, momentum=0.70,
        tts_speed=0.85, tts_temperature=0.68,
        tts_zh="用有气无力、漫不经心的语气说，语调平淡，尾音拖长",
        tts_en="Speak with complete boredom, flat and monotone, trailing off",
    ),
    EmotionProfile(
        label="calmness",
        valence=0.20, energy=0.35, dominance=0.60, intensity=0.20, momentum=0.80,
        tts_speed=0.97, tts_temperature=0.75,
        tts_zh="用平静自然的语气说，语速均匀，从容不迫",
        tts_en="Speak calmly and naturally, warm and steady, unhurried",
    ),
    EmotionProfile(
        label="confusion",
        valence=-0.20, energy=0.55, dominance=0.30, intensity=0.45, momentum=0.40,
        tts_speed=0.96, tts_temperature=0.93,
        tts_zh="用困惑不解的语气说，句尾语调上扬，偶有停顿，像边说边在想",
        tts_en="Speak with confusion, uncertain with rising intonation, as if thinking out loud",
    ),
    EmotionProfile(
        label="craving",
        valence=0.30, energy=0.65, dominance=0.55, intensity=0.55, momentum=0.50,
        tts_speed=1.02, tts_temperature=0.90,
        tts_zh="用迫切渴望的语气说，语速略快，带着克制不住的想要",
        tts_en="Speak with eager longing, slightly urgent, desire barely held back",
    ),
    EmotionProfile(
        label="disgust",
        valence=-0.75, energy=0.60, dominance=0.70, intensity=0.65, momentum=0.50,
        tts_speed=0.94, tts_temperature=0.88,
        tts_zh="用明显厌恶的语气说，话语简短有力，像不想多说",
        tts_en="Speak with disgust, clipped and dismissive, as if recoiling from the subject",
    ),
    EmotionProfile(
        label="empathic_pain",
        valence=-0.55, energy=0.40, dominance=0.35, intensity=0.60, momentum=0.60,
        tts_speed=0.78, tts_temperature=0.94,
        tts_zh="用心疼哽咽的语气说，语速很慢，带着哭腔，像在强忍着痛苦",
        tts_en="Speak with deep heartache, very slow, voice breaking with grief for someone else",
    ),
    EmotionProfile(
        label="entrancement",
        valence=0.55, energy=0.40, dominance=0.45, intensity=0.60, momentum=0.70,
        tts_speed=0.88, tts_temperature=0.80,
        tts_zh="用出神沉醉的语气说，语速缓慢，像沉浸在梦境中",
        tts_en="Speak dreamily, slow and floating, as if pleasantly lost in a trance",
    ),
    EmotionProfile(
        label="excitement",
        valence=0.75, energy=0.90, dominance=0.75, intensity=0.75, momentum=0.45,
        tts_speed=1.12, tts_temperature=0.95,
        tts_zh="用藏不住的兴奋语气说，语速快，音调高而起伏",
        tts_en="Speak with excitement, fast and energetic, pitch high and dynamic",
    ),
    EmotionProfile(
        label="fear",
        valence=-0.65, energy=0.80, dominance=0.20, intensity=0.70, momentum=0.35,
        tts_speed=1.00, tts_temperature=0.97,
        tts_zh="用紧张害怕的语气说，语速忽快忽停，带着明显的恐惧",
        tts_en="Speak with fear, pace irregular and halting, clearly frightened",
    ),
    EmotionProfile(
        label="horror",
        valence=-0.85, energy=0.75, dominance=0.15, intensity=0.80, momentum=0.30,
        tts_speed=0.82, tts_temperature=0.96,
        tts_zh="用极度惊骇的语气说，语速极慢，带着强烈的震惊，像被吓到说不出话",
        tts_en="Speak in horrified shock, very slow, as if frozen by what you witnessed",
    ),
    EmotionProfile(
        label="interest",
        valence=0.35, energy=0.55, dominance=0.60, intensity=0.40, momentum=0.60,
        tts_speed=1.02, tts_temperature=0.84,
        tts_zh="用专注好奇的语气说，语速适中，音调略微上扬，像迫不及待想了解更多",
        tts_en="Speak with genuine interest, alert and engaged, slightly uplifted",
    ),
    EmotionProfile(
        label="joy",
        valence=0.80, energy=0.75, dominance=0.70, intensity=0.65, momentum=0.50,
        tts_speed=1.10, tts_temperature=0.80,
        tts_zh="用藏不住的喜悦语气说，语速明快，音调偏高活泼",
        tts_en="Speak joyfully, lively and cheerful, happiness impossible to hide",
    ),
    EmotionProfile(
        label="nostalgia",
        valence=0.20, energy=0.30, dominance=0.50, intensity=0.50, momentum=0.70,
        tts_speed=0.89, tts_temperature=0.86,
        tts_zh="用带着甜蜜忧愁的语气说，语速慢，像在回味珍贵的往事",
        tts_en="Speak with gentle nostalgia, slow and wistful, warm but tinged with ache",
    ),
    EmotionProfile(
        label="relief",
        valence=0.65, energy=0.45, dominance=0.65, intensity=0.55, momentum=0.55,
        tts_speed=0.95, tts_temperature=0.84,
        tts_zh="用如释重负的语气说，像终于吐出了一口长气",
        tts_en="Speak with relief, relaxed and releasing tension, as if finally exhaling",
    ),
    EmotionProfile(
        label="romance",
        valence=0.75, energy=0.50, dominance=0.60, intensity=0.65, momentum=0.65,
        tts_speed=0.92, tts_temperature=0.88,
        tts_zh="用柔情温柔的语气说，语速慢，音量轻，像是在对心上人轻声细语",
        tts_en="Speak romantically, soft and intimate, slow, as if whispering to someone you love",
    ),
    EmotionProfile(
        label="sadness",
        valence=-0.60, energy=0.25, dominance=0.25, intensity=0.55, momentum=0.60,
        tts_speed=0.78, tts_temperature=0.80,
        tts_zh="用深深悲伤的语气说，语速很慢，带着哭腔，每个字都带着沉重的哀伤",
        tts_en="Speak sadly, very slow, with a tearful tone, heavy with grief",
    ),
    EmotionProfile(
        label="satisfaction",
        valence=0.65, energy=0.45, dominance=0.70, intensity=0.50, momentum=0.70,
        tts_speed=0.97, tts_temperature=0.78,
        tts_zh="用踏实满足的语气说，语速均匀，不张扬但令人感到舒适",
        tts_en="Speak with quiet satisfaction, warm and grounded, calm and self-assured",
    ),
    EmotionProfile(
        label="sexual_desire",
        valence=0.70, energy=0.70, dominance=0.65, intensity=0.65, momentum=0.55,
        tts_speed=0.88, tts_temperature=0.90,
        tts_zh="用亲密挑逗的语气说，语速慢，音量轻，像在说悄悄话",
        tts_en="Speak in a low intimate tone, slow and deliberately close, as if whispering",
    ),
    EmotionProfile(
        label="surprise",
        valence=0.00, energy=0.80, dominance=0.45, intensity=0.65, momentum=0.35,
        tts_speed=1.06, tts_temperature=0.97,
        tts_zh="用真实惊讶的语气说，音调在关键词处急速上扬，带着真实的意外感",
        tts_en="Speak with genuine surprise, pitch jumping sharply, naturally uneven",
    ),
]}

# Baseline state — what decay drifts toward
BASELINE = AgentState(
    valence=0.2,
    energy=0.4,
    dominance=0.55,
    emotion="calmness",
    intensity=0.2,
    momentum=0.5,
)

# Valid emotion label set for fast membership tests
VALID_EMOTION_LABELS: frozenset[str] = frozenset(EMOTION_REGISTRY.keys())


# ── Nearest-neighbour PAD label resolution ────────────────────────────────────

def nearest_emotion(valence: float, energy: float, dominance: float) -> str:
    """Return the emotion label whose PAD coordinates are closest to the given point.

    Uses Euclidean distance in (valence, energy, dominance) space.
    Valence is scaled by 0.5 to account for its wider range (-1 → +1 vs 0 → 1).
    """
    best_label = "calmness"
    best_dist = float("inf")
    for profile in EMOTION_REGISTRY.values():
        dv = (valence - profile.valence) * 0.5   # scale valence range to ~same as energy/dominance
        de = energy - profile.energy
        dd = dominance - profile.dominance
        dist = math.sqrt(dv * dv + de * de + dd * dd)
        if dist < best_dist:
            best_dist = dist
            best_label = profile.label
    return best_label


def resolve_ecs_susceptibility(emotion_label: str, ecs: EcsWeights) -> float:
    """Return the ECS susceptibility weight for a given emotion label.

    For emotions in two categories, returns the higher weight.
    """
    dual = ECS_DUAL_CATEGORY.get(emotion_label)
    if dual:
        w1 = getattr(ecs, dual[0], 0.5)
        w2 = getattr(ecs, dual[1], 0.5)
        return max(w1, w2)
    category = ECS_CATEGORY_MAP.get(emotion_label, "happiness")
    return getattr(ecs, category, 0.5)
