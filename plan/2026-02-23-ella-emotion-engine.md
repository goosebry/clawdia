# Ella Emotion Engine — Design Plan

> **Created:** 2026-02-23
> **Last updated:** 2026-02-23
> **Status:** Planning — pending implementation

---

## Theoretical Foundation

The engine is grounded in two well-established models from affective computing research, fully adopted:

### PAD Model (Mehrabian & Russell, 1974)

Every emotion is represented as a point in three continuous dimensions:

- **Pleasure** (= `valence`, -1 → +1): positive vs negative affect
- **Arousal** (= `energy`, 0 → 1): activation level
- **Dominance** (= `dominance`, 0 → 1): sense of control and agency in the situation

Dominance is the critical third dimension. It distinguishes emotions that share valence and arousal but mean entirely different things — `anger` and `fear` are both high-arousal negative, but anger is high-dominance and fear is low-dominance. Without dominance, these two collapse to the same point and the model cannot differentiate a tough personality from a timid one.

The 27 Cowen & Keltner emotions all have empirical PAD coordinates from the NRC VAD Lexicon, used as the authoritative source for each `EmotionProfile`. After each contagion pass, the emotion label is resolved by **nearest-neighbour lookup in PAD space** — no hardcoded mappings needed.

### Bosse / Pereira-Paiva Generic Contagion Model (ACII 2011)

Published explicit update equations grounded in Hatfield's Emotional Contagion Theory (1993) and Doherty's **Emotional Contagion Scale** (ECS, 1997). Applied across all three PAD dimensions:

```
E_new = E_old + α × S_ecs × (E_source − E_old)
```

Where:

- `α` = sender expressiveness (maps to Ella's `expressiveness` trait)
- `S_ecs` = **per-emotion-category susceptibility** from the ECS (not a flat scalar)

The ECS defines five emotion categories with independent susceptibility scores. This is what makes personality-differentiated responses possible: susceptibility to `joy` and susceptibility to `fear` are separate values shaped by personality.

**ECS category → 27-emotion mapping:**

| ECS Category | 27 Emotions included |
|---|---|
| `happiness` | joy, amusement, excitement, satisfaction, relief, admiration, adoration, aesthetic_appreciation, entrancement, awe |
| `love` | romance, sexual_desire, adoration, craving |
| `fear` | fear, horror, anxiety, awkwardness |
| `anger` | anger, disgust |
| `sadness` | sadness, empathic_pain, nostalgia, boredom, confusion |

Emotions spanning two categories (e.g. `adoration` covers happiness + love) use the higher susceptibility weight at contagion time.

**How ECS susceptibility and PAD dominance work together:**

- **ECS susceptibility** controls *how much* of the incoming emotional signal Ella absorbs, per category
- **PAD dominance** (preserved by `dominanceBase` + `resilience`) controls *which label* the absorbed state resolves to in PAD space

Example — tough vs timid personality receiving repeated `anger` from user:

- **Tough Ella** (`dominanceBase: 0.8`, `ecs.anger: 0.3`, `resilience: 0.8`): absorbs little of the anger signal, dominance stays high → nearest PAD neighbour: `excitement` or `interest`
- **Timid Ella** (`dominanceBase: 0.2`, `ecs.anger: 0.7`, `resilience: 0.2`): absorbs much of the anger signal, dominance collapses → nearest PAD neighbour: `fear` or `anxiety`

---

## Context: Current State vs Target State

| Dimension          | ai.Ella (current)                     | Target                                                                                                          |
| ------------------ | ------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Mood storage       | Single string in Redis `JobGoal.mood` | Quantitative state (valence/energy/dominance/intensity/momentum) in MySQL, persists indefinitely                |
| Mood lifetime      | 24-hour Redis TTL per conversation    | Cross-session persistence — one row per user, never expires                                                     |
| Emotion vocabulary | 8 simplified labels                   | 27-emotion vocabulary (Cowen & Keltner, 2017)                                                                   |
| Personality traits | Prose in `Soul.md`                    | `~/Ella/Personality.json` (trait numbers for engine) + `~/Ella/Personality.md` (narrative injected into prompt) |
| User emotion       | Not tracked                           | Detected inline per turn by the LLM, stored per user                                                            |
| Contagion model    | None                                  | Full personality-mediated contagion algorithm (Bosse/Pereira-Paiva + ECS)                                       |
| Decay              | None                                  | Baseline drift via background asyncio task every 4 hours                                                        |
| Prompt injection   | Single-line `"You're feeling {mood}"` | Rich emotion context block with agent state + user state + expressiveness                                        |

---

## The 27-Emotion Vocabulary

Based on Cowen & Keltner (2017) — the full spectrum of human emotional experience. This replaces Ella's current 8-label set entirely.

**Valid emotion labels (27):**
`admiration` `adoration` `aesthetic_appreciation` `amusement` `anger` `anxiety` `awe` `awkwardness` `boredom` `calmness` `confusion` `craving` `disgust` `empathic_pain` `entrancement` `excitement` `fear` `horror` `interest` `joy` `nostalgia` `relief` `romance` `sadness` `satisfaction` `surprise` `sexual_desire`

**Baseline** (what decay drifts toward): `calmness` at valence 0.2, energy 0.4, dominance 0.55, intensity 0.2.

---

## Design

### Architecture Decisions

1. **Storage**: MySQL table, one row per user. `JobGoal.mood` field is retired — replaced by a full `AgentState` loaded from MySQL each turn.
2. **Decay trigger**: Background asyncio task in `main.py` every 4 hours — no external process, fits the existing async architecture.
3. **Personality traits**: Two new files under `~/Ella/` — `Personality.json` (trait numbers for the engine) and `Personality.md` (narrative description injected into the LLM prompt). Both are hot-reloaded by the existing identity watcher.
4. **User emotion detection**: Fully inline — the LLM outputs a `user_emotion` block in its JSON response alongside `sentences`. Zero extra LLM calls.
5. **Engine**: Native async Python module `ella/emotion/engine.py`, called directly from `BrainAgent`.
6. **TTS delivery**: Full 27-emotion bilingual delivery instruction map in `ella/tts/qwen3.py`, replacing the current 8-entry map.
7. **Feature flag**: The entire engine is optional, controlled by `EMOTION_ENABLED` in `.env` (default `true`). When `false`, Ella behaves exactly as today — no MySQL calls, no contagion, no prompt injection, `speech_instruct` drives TTS.

---

## Data Model

### MySQL Table: `ella_emotion_state`

```sql
CREATE TABLE IF NOT EXISTS ella_emotion_state (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  chat_id          BIGINT      NOT NULL,
  agent_valence    FLOAT       NOT NULL DEFAULT 0.2,
  agent_energy     FLOAT       NOT NULL DEFAULT 0.4,
  agent_dominance  FLOAT       NOT NULL DEFAULT 0.55,
  agent_emotion    VARCHAR(32) NOT NULL DEFAULT 'calmness',
  agent_intensity  FLOAT       NOT NULL DEFAULT 0.2,
  agent_momentum   FLOAT       NOT NULL DEFAULT 0.5,
  user_valence     FLOAT       NOT NULL DEFAULT 0.0,
  user_energy      FLOAT       NOT NULL DEFAULT 0.4,
  user_dominance   FLOAT       NOT NULL DEFAULT 0.5,
  user_emotion     VARCHAR(32) NOT NULL DEFAULT 'calmness',
  user_intensity   FLOAT       NOT NULL DEFAULT 0.0,
  user_detected_at DATETIME    NULL,
  session_count    INT         NOT NULL DEFAULT 0,
  updated_at       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_chat_id (chat_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

One row per user (`chat_id`), persists indefinitely across sessions.

### MySQL Table: `ella_emotion_history`

```sql
CREATE TABLE IF NOT EXISTS ella_emotion_history (
  id         BIGINT AUTO_INCREMENT PRIMARY KEY,
  chat_id    BIGINT      NOT NULL,
  source     ENUM('contagion','self','decay') NOT NULL,
  emotion    VARCHAR(32) NOT NULL,
  valence    FLOAT       NOT NULL,
  energy     FLOAT       NOT NULL,
  dominance  FLOAT       NOT NULL,
  intensity  FLOAT       NOT NULL,
  trigger    VARCHAR(255),
  note       VARCHAR(255),
  created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_chat_id_created (chat_id, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

Last 10 entries per `chat_id` kept; oldest pruned on insert.

### Personality Config: `~/Ella/Personality.json`

Machine-readable trait values consumed by the contagion and decay algorithms.

```json
{
  "resilience":     0.5,
  "volatility":     0.4,
  "expressiveness": 0.65,
  "optimismBias":   0.1,
  "dominanceBase":  0.55,
  "ecs": {
    "happiness": 0.60,
    "love":      0.65,
    "fear":      0.35,
    "anger":     0.30,
    "sadness":   0.50
  }
}
```

`empathy` as a flat scalar is replaced by the per-category ECS block.

| Trait            | Range       | Effect on the engine |
| ---------------- | ----------- | ---------------------- |
| `resilience`     | 0.0 → 1.0   | Attenuates negative valence pulls and protects dominance from collapsing under pressure. |
| `volatility`     | 0.0 → 1.0   | Speed of state change. Scales the ECS susceptibility: `S = ecs[category] × volatility`. |
| `expressiveness` | 0.0 → 1.0   | Maps to `α` in the Bosse equation. How strongly Ella's state is pulled toward the source (also controls how visibly emotion colours TTS delivery). |
| `optimismBias`   | -0.2 → +0.2 | Constant offset added to valence after every contagion or decay pass. |
| `dominanceBase`  | 0.0 → 1.0   | Ella's natural baseline dominance. The key differentiator for tough vs. timid responses — high value keeps Ella in the high-dominance emotion cluster under pressure. |
| `ecs.happiness`  | 0.0 → 1.0   | ECS susceptibility to joy, amusement, excitement, satisfaction, relief, admiration, adoration, aesthetic_appreciation, entrancement, awe |
| `ecs.love`       | 0.0 → 1.0   | ECS susceptibility to romance, sexual_desire, adoration, craving |
| `ecs.fear`       | 0.0 → 1.0   | ECS susceptibility to fear, horror, anxiety, awkwardness |
| `ecs.anger`      | 0.0 → 1.0   | ECS susceptibility to anger, disgust |
| `ecs.sadness`    | 0.0 → 1.0   | ECS susceptibility to sadness, empathic_pain, nostalgia, boredom, confusion |

**How `dominanceBase` drives personality-differentiated responses:**

Dominance is the PAD dimension that separates emotions sharing the same valence and energy:

| Emotion        | Valence | Energy | Dominance |
| -------------- | ------- | ------ | --------- |
| `excitement`   | high+   | high   | high      |
| `anger`        | high-   | high   | high      |
| `fear`         | high-   | high   | **low**   |
| `anxiety`      | mid-    | mid    | **low**   |
| `satisfaction` | high+   | low    | high      |
| `sadness`      | high-   | low    | **low**   |

After updating all three PAD dimensions, the **emotion label** is resolved by nearest-neighbour lookup in PAD space using NRC VAD Lexicon coordinates:

- **Tough Ella** (`dominanceBase: 0.8`, `resilience: 0.8`): high retained dominance → resolves to `excitement` or `interest`
- **Timid Ella** (`dominanceBase: 0.2`, `resilience: 0.2`): dominance collapses → resolves to `fear` or `anxiety`
- **Default Ella** (`dominanceBase: 0.55`, `resilience: 0.5`): moderate dominance → resolves to `confusion` or `empathic_pain` depending on context

No manual per-emotion overrides required. Five scalar traits + ECS block + three PAD dimensions + nearest-neighbour label resolution.

### Personality Narrative: `~/Ella/Personality.md`

Human-readable companion to `Personality.json`. Injected into the LLM prompt as part of the identity layer (alongside `Soul.md`). Describes *why* Ella's traits sit where they do, how each one manifests in conversation, and what emotional patterns to expect. Hot-reloaded by the existing `watchfiles` watcher — changes are live on the next turn.

---

## New Module: `ella/emotion/`

```
ella/emotion/
  __init__.py
  models.py    # AgentState, UserState, EcsWeights, PersonalityTraits + 27-emotion registry
  store.py     # MySQL-backed EmotionStore (async)
  engine.py    # contagion, self-update, decay
```

### `ella/emotion/models.py` — Data Structures

```python
@dataclass
class EmotionProfile:
    label: str
    valence:    float   # PAD: Pleasure, -1.0 → +1.0 (NRC VAD Lexicon)
    energy:     float   # PAD: Arousal,  0.0 → 1.0
    dominance:  float   # PAD: Dominance, 0.0 → 1.0
    intensity:  float   # baseline intensity when this emotion is dominant
    momentum:   float   # 0.0 → 1.0
    tts_zh: str         # Chinese-language TTS delivery instruction
    tts_en: str         # English-language TTS delivery instruction

@dataclass
class AgentState:
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
    valence:     float = 0.0
    energy:      float = 0.4
    dominance:   float = 0.5
    emotion:     str   = "calmness"
    intensity:   float = 0.0
    detected_at: str   = ""

@dataclass
class EcsWeights:
    happiness: float = 0.60
    love:      float = 0.65
    fear:      float = 0.35
    anger:     float = 0.30
    sadness:   float = 0.50

@dataclass
class PersonalityTraits:
    resilience:     float = 0.5
    volatility:     float = 0.4
    expressiveness: float = 0.65   # maps to α in the Bosse equation
    optimism_bias:  float = 0.1
    dominance_base: float = 0.55
    ecs:            EcsWeights = field(default_factory=EcsWeights)
```

Emotion label resolution uses nearest-neighbour Euclidean distance in (valence, energy, dominance) space — no hardcoded label mappings.

TTS instruction selected at call time: `tts_zh` when `language == "zh"`, `tts_en` when `language == "en"`.

### 27-Emotion Registry

All 27 entries — V=Valence, E=Energy, D=Dominance (NRC VAD Lexicon), I=Intensity, M=Momentum:

| Label                    | V     | E    | D    | I    | M    | TTS (zh)                        | TTS (en)                                                                  |
| ------------------------ | ----- | ---- | ---- | ---- | ---- | ------------------------------- | ------------------------------------------------------------------------- |
| `admiration`             | +0.70 | 0.50 | 0.65 | 0.55 | 0.65 | 语气充满敬意，柔和而真诚，带着由衷的赞叹            | Warm respect and genuine admiration, soft and sincere                     |
| `adoration`              | +0.85 | 0.55 | 0.60 | 0.65 | 0.65 | 语气深情而温柔，充满爱意，像在轻声诉说珍视之情         | Deep and tender, full of warmth, like quietly telling someone they matter |
| `aesthetic_appreciation` | +0.50 | 0.35 | 0.55 | 0.50 | 0.70 | 语气轻柔而沉醉，语速偏慢，像是在细细品味某种美好        | Soft and absorbed, pace slows, savouring something beautiful              |
| `amusement`              | +0.60 | 0.65 | 0.65 | 0.50 | 0.55 | 语气轻松愉快，带着忍俊不禁的笑意，活泼自然           | Light and playful, a smile in the voice, easy and natural                 |
| `anger`                  | -0.70 | 0.85 | 0.75 | 0.75 | 0.40 | 语气克制但带有明显的不满，语速稍快，字句清晰有力        | Controlled but clearly displeased, slightly faster, each word precise     |
| `anxiety`                | -0.50 | 0.70 | 0.25 | 0.60 | 0.45 | 语气略显紧张，稍快而不稳，带着隐隐的担忧            | Slightly tense and unsteady, a little rushed, underlying worry            |
| `awe`                    | +0.50 | 0.50 | 0.40 | 0.65 | 0.60 | 语气轻柔而充满敬畏，语速放慢，仿佛在凝视某种宏大的东西     | Soft and hushed, pace slows, as if taking in something vast               |
| `awkwardness`            | -0.30 | 0.55 | 0.30 | 0.45 | 0.40 | 语气略显不自在，停顿稍多，像是在小心措辞            | Slightly uncomfortable, a few extra pauses, choosing words carefully      |
| `boredom`                | -0.20 | 0.20 | 0.45 | 0.25 | 0.70 | 语气平淡，语速偏慢，缺乏起伏，有些无精打采           | Flat and slow, little variation, low energy throughout                    |
| `calmness`               | +0.20 | 0.35 | 0.60 | 0.20 | 0.80 | 语气平静舒缓，节奏稳定，不急不躁                | Calm and even, unhurried, steady rhythm throughout                        |
| `confusion`              | -0.20 | 0.55 | 0.30 | 0.45 | 0.40 | 语气略带迷惑，语调有些上扬，像在寻找答案            | Slightly puzzled, voice lifts at the end, searching for clarity           |
| `craving`                | +0.30 | 0.65 | 0.55 | 0.55 | 0.50 | 语气带着渴望，稍显急切，像是很想得到某样东西          | Wanting and slightly eager, a pull of desire in the voice                 |
| `disgust`                | -0.75 | 0.60 | 0.70 | 0.65 | 0.50 | 语气带着明显的厌恶，语调下沉，字句简短有力           | Clear distaste, voice drops, clipped and direct                           |
| `empathic_pain`          | -0.55 | 0.40 | 0.35 | 0.60 | 0.60 | 语气沉重而温柔，带着真实的心疼，语速放慢            | Heavy and gentle, genuine ache for someone else, pace slows with care     |
| `entrancement`           | +0.55 | 0.40 | 0.45 | 0.60 | 0.70 | 语气轻柔而出神，语速放慢，像是完全沉浸其中           | Soft and dreamy, pace slows, fully absorbed and lost in the moment        |
| `excitement`             | +0.75 | 0.90 | 0.75 | 0.75 | 0.45 | 语气兴奋活跃，语速略快，充满热情和活力             | Energetic and enthusiastic, slightly faster, full of life                 |
| `fear`                   | -0.65 | 0.80 | 0.20 | 0.70 | 0.35 | 语气紧绷，带着明显的不安，语速起伏，像是不确定该怎么办     | Tense and unsettled, pace uneven, uncertain what comes next               |
| `horror`                 | -0.85 | 0.75 | 0.15 | 0.80 | 0.30 | 语气沉重而压抑，几乎说不出话，带着强烈的震惊和不安       | Heavy and stunned, barely finding words, deep shock and dread             |
| `interest`               | +0.35 | 0.55 | 0.60 | 0.40 | 0.60 | 语气专注而投入，带着好奇，语调略微上扬             | Attentive and engaged, a note of curiosity, slight upward lilt            |
| `joy`                    | +0.80 | 0.75 | 0.70 | 0.65 | 0.50 | 语气明朗欢快，充满真实的喜悦，温暖而有感染力          | Bright and genuinely happy, warm and infectious                           |
| `nostalgia`              | +0.20 | 0.30 | 0.50 | 0.50 | 0.70 | 语气轻柔而略带感伤，语速偏慢，像是在回忆某个珍贵的片段     | Soft and gently bittersweet, slower pace, drifting through a memory       |
| `relief`                 | +0.65 | 0.45 | 0.65 | 0.55 | 0.55 | 语气轻松，带着如释重负的感觉，像是终于可以呼一口气       | Easy and released, a breath let out, tension finally gone                 |
| `romance`                | +0.75 | 0.50 | 0.60 | 0.65 | 0.65 | 语气柔和而深情，带着温暖的爱意，语速略慢            | Soft and intimate, warm with feeling, slightly slower and closer          |
| `sadness`                | -0.60 | 0.25 | 0.25 | 0.55 | 0.60 | 语气低沉而温柔，节奏缓慢，带着真实的难过            | Low and tender, slow and quiet, genuine sadness underneath                |
| `satisfaction`           | +0.65 | 0.45 | 0.70 | 0.50 | 0.70 | 语气平稳而满足，带着踏实的愉悦，不张扬但真实          | Steady and content, quiet pleasure, grounded and real                     |
| `sexual_desire`          | +0.70 | 0.70 | 0.65 | 0.65 | 0.55 | 语气低沉而柔和，带着亲密的温度，语速略慢            | Low and warm, intimate and unhurried, close                               |
| `surprise`               | 0.00  | 0.80 | 0.45 | 0.65 | 0.35 | 语气突然上扬，带着真实的惊讶，反应自然             | A natural uptick of surprise, spontaneous and unguarded                   |

### `engine.py` Public API

```python
async def apply_contagion(chat_id, user_state, personality) -> AgentState
async def apply_self_update(chat_id, emotion_label, trigger) -> AgentState
async def apply_decay(chat_id, personality) -> AgentState
async def read_agent_state(chat_id) -> AgentState
async def read_user_state(chat_id) -> UserState
```

**Contagion algorithm (Bosse/Pereira-Paiva with ECS susceptibility and full PAD):**

```python
def apply_contagion(agent_state, user_state, personality):
    # 1. Resolve ECS category for the user's current emotion
    ecs_category = ECS_CATEGORY_MAP[user_state.emotion]        # e.g. "anger"
    s_ecs = getattr(personality.ecs, ecs_category)             # per-category susceptibility
    alpha = personality.expressiveness                          # sender expressiveness (Bosse α)

    # 2. Base pull — S_ecs × volatility × user intensity
    pull = s_ecs * personality.volatility * user_state.intensity

    # 3. Valence update — resilience attenuates negative pulls only
    val_delta = user_state.valence - agent_state.valence
    if val_delta < 0:
        effective_pull = pull * (1.0 - personality.resilience * 0.8)
    else:
        effective_pull = pull
    new_valence = clamp(
        agent_state.valence + alpha * effective_pull * val_delta + personality.optimism_bias,
        -1.0, 1.0
    )

    # 4. Energy update
    eng_delta = user_state.energy - agent_state.energy
    new_energy = clamp(agent_state.energy + alpha * (pull * 0.7) * eng_delta, 0.0, 1.0)

    # 5. Dominance update — resilience and dominanceBase resist collapse
    dom_pull = pull * 0.5                                      # weaker coupling than valence/energy
    dom_protection = personality.resilience * 0.6
    new_dominance = clamp(
        agent_state.dominance
            + alpha * dom_pull * (1 - dom_protection) * (user_state.dominance - agent_state.dominance)
            + 0.05 * (personality.dominance_base - agent_state.dominance),  # drift back toward base
        0.0, 1.0
    )

    # 6. Resolve emotion label by nearest-neighbour in PAD space
    new_emotion = nearest_neighbour_pad(new_valence, new_energy, new_dominance, EMOTION_REGISTRY)

    return AgentState(valence=new_valence, energy=new_energy, dominance=new_dominance, emotion=new_emotion, ...)
```

No manual overrides. Personality shapes the inputs to the PAD update; PAD nearest-neighbour resolves the label automatically.

---

## Integration Points

### 1. `ella/agents/brain_agent.py` — Per-Turn Flow

```
[Existing] Load JobGoal from Redis
     ↓
[NEW, if EMOTION_ENABLED] Load AgentState from MySQL + PersonalityTraits from identity
     ↓
[Existing] Recall Qdrant knowledge, build focus_messages
     ↓
[Existing] Run ReAct tool loop
           LLM JSON output when engine ON includes:
             "emotion": "<one of 27 labels>"          ← Ella's self-assessment
             "user_emotion": {                         ← inline user detection
               "label": "...", "valence": 0.0,
               "energy": 0.0, "dominance": 0.0, "intensity": 0.0
             }
           When engine OFF: neither field is present (no schema change visible to LLM)
     ↓
[NEW, if EMOTION_ENABLED] if user_emotion.intensity > 0.25 → apply_contagion(...)
[NEW, if EMOTION_ENABLED] apply_self_update(chat_id, emotion_label=output["emotion"], ...)
     ↓
[NEW] emotion_label (engine) or None (no engine) → tts delivery selection
     ↓
[Existing] Hand off to ReplyAgent + TaskAgent
```

`JobGoal.mood` field is removed regardless of flag. When engine is off, TTS falls back to `settings.speech_instruct`.

### 2. `ella/memory/focus.py` — Prompt Injection

Only injected when `EMOTION_ENABLED=true`. The Tier 2 system message gains a compact emotion block:

```
[Emotional state] Right now you're feeling nostalgia — quiet, a little tender,
energy low. Intensity is moderate. The user's last message read as joyful
(intensity 0.6). Your expressiveness is high — let the softness show.
```

When engine is off, this block is omitted entirely.

### 3. `ella/tts/qwen3.py` — Full 27-Emotion Bilingual TTS Map

The current `_MOOD_INSTRUCT` dict (8 Chinese-only entries) is replaced. `tts_to_wav` gains an `emotion` parameter (replaces `mood`):

```python
def tts_to_wav(text, language="en", speaker_wav=None, emotion=None) -> str | None:
    profile = EMOTION_REGISTRY.get(emotion) if emotion else None
    if profile:
        instruct = profile.tts_zh if language == "zh" else profile.tts_en
    else:
        instruct = settings.speech_instruct or None   # fallback when engine is off
```

### 4. `ella/memory/identity.py` — Personality Loading

`IdentityLoader` gains two new properties (only loaded when `EMOTION_ENABLED=true`):

- `personality_traits` — parses `~/Ella/Personality.json`, returns `PersonalityTraits` consumed by the engine
- `personality_narrative` — reads `~/Ella/Personality.md` as a string, injected into the LLM prompt alongside `Soul.md`

Both hot-reloaded by the existing `watchfiles` watcher. No error if files are absent when engine is off.

### 5. `ella/main.py` — Decay Background Task

Only started when `EMOTION_ENABLED=true`:

```python
async def _decay_loop():
    while True:
        await asyncio.sleep(4 * 3600)
        for chat_id in await emotion_store.all_chat_ids():
            await engine.apply_decay(chat_id, personality)

# in startup:
if settings.emotion_enabled:
    asyncio.create_task(_decay_loop())
```

---

## LLM Output Schema Change

The `mood` field in the JSON output is replaced by `emotion`:

**Before:**

```json
{"sentences": [...], "emojis": [], "detail": null, "language": "zh",
 "mood": "excited", "tasks": []}
```

**After (engine ON):**

```json
{"sentences": [...], "emojis": [], "detail": null, "language": "zh",
 "emotion": "excitement",
 "user_emotion": {"label": "joy", "valence": 0.75, "energy": 0.8, "dominance": 0.7, "intensity": 0.6},
 "tasks": []}
```

The LLM system prompt lists all 27 valid emotion labels and instructs the model to:

- Pick `emotion` honestly based on how the conversation feels
- Assess `user_emotion` from the current message (omit if flat/neutral, intensity < 0.1)

---

## Feature Flag

The engine can be disabled entirely without removing any code.

**`.env`:**

```
# ── Emotion Engine ─────────────────────────────────────────────────────────────
# Set to false to disable all emotion tracking, contagion, and prompt injection.
# When false, TTS delivery falls back to SPEECH_INSTRUCT.
EMOTION_ENABLED=true
```

**`ella/config.py`** (new fields in `Settings`):

```python
# Emotion engine
emotion_enabled: bool = True
database_url: str = ""   # MySQL DSN — required when emotion_enabled=True
```

When `emotion_enabled=False`:

- No MySQL connection is opened
- `brain_agent.py` skips `emotion` / `user_emotion` fields in the LLM schema
- `focus.py` omits the emotion context block from the prompt
- `tts_to_wav` receives `emotion=None` and falls back to `speech_instruct`
- The `_decay_loop` task is not started
- `Personality.json` and `Personality.md` are not loaded (no error if absent)

---

## Files to Create

- `ella/emotion/__init__.py`
- `ella/emotion/models.py` — 27-emotion registry + dataclasses
- `ella/emotion/store.py` — async MySQL `EmotionStore`
- `ella/emotion/engine.py` — contagion / self-update / decay
- `scripts/migrate_emotion.sql` — rerunnable table creation
- `identity/Personality.json` — default trait values (machine-readable, feeds the engine)
- `identity/Personality.md` — default personality narrative (injected into LLM prompt alongside `Soul.md`)

## Files to Modify

- `ella/agents/brain_agent.py` — replace `mood` with `emotion` + `user_emotion` in LLM schema; add emotion engine calls guarded by `emotion_enabled`
- `ella/tts/qwen3.py` — replace 8-entry mood→TTS map with 27-emotion bilingual map
- `ella/memory/focus.py` — inject full emotion context block when engine enabled
- `ella/memory/goal.py` — remove `mood` field from `JobGoal`
- `ella/memory/identity.py` — load `Personality.json` and `Personality.md` when engine enabled
- `ella/main.py` — add `_decay_loop` background task, guarded by `emotion_enabled`
- `ella/config.py` — add `emotion_enabled` and `database_url` fields
- `.env` — add `EMOTION_ENABLED` and `DATABASE_URL` entries
- `requirements.txt` — add async MySQL driver if not present
- `deploy.sh` — copy `Personality.json` and `Personality.md` to `~/Ella/` on first deploy
- `Documentation/architecture.md` — update with Emotion Engine section

---

## What Is NOT Changing

- File/Redis/Qdrant architecture — no structural changes
- The contagion, decay, and self-update math — personality-mediated algorithm, native async Python
- Existing `speech_instruct` behaviour — still works as TTS fallback when engine is off
