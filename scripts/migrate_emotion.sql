-- Ella Emotion Engine — Database Migration
-- Rerunnable: uses CREATE TABLE IF NOT EXISTS and ALTER TABLE IF NOT EXISTS patterns.
-- Run against the database configured in DATABASE_URL.

-- ── Emotion state — one row per user (chat_id) ───────────────────────────────
-- Persists Ella's emotional state and the last detected user emotion across all
-- sessions. Never expires — the emotion engine is a long-lived memory layer.

CREATE TABLE IF NOT EXISTS ella_emotion_state (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  chat_id          BIGINT      NOT NULL,

  -- Ella's current emotional state (PAD: Pleasure-Arousal-Dominance)
  agent_valence    FLOAT       NOT NULL DEFAULT 0.2,   -- Pleasure: -1.0 → +1.0
  agent_energy     FLOAT       NOT NULL DEFAULT 0.4,   -- Arousal:  0.0 → 1.0
  agent_dominance  FLOAT       NOT NULL DEFAULT 0.55,  -- Dominance: 0.0 → 1.0
  agent_emotion    VARCHAR(32) NOT NULL DEFAULT 'calmness',
  agent_intensity  FLOAT       NOT NULL DEFAULT 0.2,   -- 0.0 → 1.0
  agent_momentum   FLOAT       NOT NULL DEFAULT 0.5,   -- 0.0 → 1.0

  -- Last detected user emotional state
  user_valence     FLOAT       NOT NULL DEFAULT 0.0,
  user_energy      FLOAT       NOT NULL DEFAULT 0.4,
  user_dominance   FLOAT       NOT NULL DEFAULT 0.5,
  user_emotion     VARCHAR(32) NOT NULL DEFAULT 'calmness',
  user_intensity   FLOAT       NOT NULL DEFAULT 0.0,
  user_detected_at DATETIME    NULL,

  -- Metadata
  session_count    INT         NOT NULL DEFAULT 0,
  updated_at       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP,

  UNIQUE KEY uq_chat_id (chat_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ── Emotion history — rolling log of state changes ───────────────────────────
-- Tracks each contagion, self-update, and decay event.
-- Oldest entries pruned on insert (keep last 10 per chat_id — enforced in app).

CREATE TABLE IF NOT EXISTS ella_emotion_history (
  id         BIGINT AUTO_INCREMENT PRIMARY KEY,
  chat_id    BIGINT      NOT NULL,
  source     ENUM('contagion','self','decay') NOT NULL,
  emotion    VARCHAR(32) NOT NULL,
  valence    FLOAT       NOT NULL,
  energy     FLOAT       NOT NULL,
  dominance  FLOAT       NOT NULL DEFAULT 0.5,
  intensity  FLOAT       NOT NULL,
  trigger_emotion VARCHAR(255) NULL,   -- emotion label or event that caused this change
  note            VARCHAR(255) NULL,   -- free-form annotation (e.g. "user: joy 0.8")
  created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,

  INDEX idx_chat_id_created (chat_id, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
