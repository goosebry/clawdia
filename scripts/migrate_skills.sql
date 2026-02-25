-- Ella Skill System — MySQL migration
-- Rerunnable: uses IF NOT EXISTS and ON DUPLICATE KEY UPDATE patterns.
-- Run with: mysql -u <user> -p <database> < scripts/migrate_skills.sql

-- ── ella_skill_runs ───────────────────────────────────────────────────────────
-- Permanent audit log and checkpoint backup for all skill executions.
-- Redis is the fast working copy during execution; this table is the source of
-- truth for "what happened?" and the fallback when the Redis volume is lost.
CREATE TABLE IF NOT EXISTS ella_skill_runs (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id           VARCHAR(36)  NOT NULL UNIQUE,       -- UUID, matches Redis ella:skill:{run_id}
    chat_id          BIGINT       NOT NULL,
    skill_name       VARCHAR(64)  NOT NULL,
    goal             TEXT         NOT NULL,
    status           ENUM('running','paused','completed','failed','cancelled')
                     NOT NULL DEFAULT 'running',
    phase            VARCHAR(64)  DEFAULT NULL,          -- last saved phase (e.g. 'analyse')
    cycle            TINYINT      NOT NULL DEFAULT 0,    -- last completed cycle number (1-indexed)
    stored_points    INT          NOT NULL DEFAULT 0,    -- Qdrant knowledge entries written so far
    sources_done     JSON         DEFAULT NULL,          -- JSON array of processed URLs/paths
    notes_snapshot   LONGTEXT     DEFAULT NULL,          -- JSON array of accumulated notes (fallback)
    summary          TEXT         DEFAULT NULL,          -- final synthesis (set on completion)
    started_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP,
    completed_at     DATETIME     DEFAULT NULL,

    INDEX idx_skill_runs_chat_status (chat_id, status),
    INDEX idx_skill_runs_status      (status),
    INDEX idx_skill_runs_updated     (updated_at)
);

-- ── ella_skill_open_questions ─────────────────────────────────────────────────
-- Knowledge gaps that remained unresolved after the maximum number of research
-- cycles. Stored for human review and potential future learning sessions.
CREATE TABLE IF NOT EXISTS ella_skill_open_questions (
    id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id     VARCHAR(36) NOT NULL,
    question   TEXT        NOT NULL,
    created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_soq_run_id (run_id),
    CONSTRAINT fk_soq_run
        FOREIGN KEY (run_id) REFERENCES ella_skill_runs(run_id)
        ON DELETE CASCADE
);
