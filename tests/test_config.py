"""Tests for ella/config.py"""
import os
import pytest
from ella.config import Settings, get_settings


def test_defaults_without_env(monkeypatch):
    """All non-required fields have sensible defaults."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123")
    s = Settings()
    assert s.telegram_bot_token == "test-token-123"
    assert s.mlx_chat_model == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert s.mlx_vl_model == "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    assert s.mlx_whisper_model == "mlx-community/whisper-small"
    assert s.embed_model == "paraphrase-multilingual-MiniLM-L12-v2"
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.max_tool_rounds == 5
    assert s.goal_ttl_seconds == 86400
    assert s.knowledge_recall_top_k == 5


def test_env_overrides(monkeypatch):
    """Env vars override defaults."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "10")
    monkeypatch.setenv("KNOWLEDGE_RECALL_TOP_K", "3")
    monkeypatch.setenv("MLX_VL_MODEL", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    s = Settings()
    assert s.max_tool_rounds == 10
    assert s.knowledge_recall_top_k == 3
    assert s.mlx_vl_model == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


def test_get_settings_singleton(monkeypatch):
    """get_settings() returns the same instance on repeated calls."""
    import ella.config as cfg
    cfg._settings = None
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "singleton-test")
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    cfg._settings = None  # reset after test
