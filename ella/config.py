from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str

    # MLX models
    mlx_chat_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    mlx_vl_model: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    mlx_whisper_model: str = "mlx-community/whisper-small-mlx"

    # TTS (Qwen3-TTS via mlx-audio)
    # HuggingFace repo id for the Qwen3-TTS MLX model.
    # Base model supports voice cloning; CustomVoice supports named voices + emotion.
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    # Path to a reference WAV/M4A for voice cloning (≥3s of clean speech).
    # When the file exists it takes priority over tts_voice.
    speaker_wav_path: str = "assets/speaker.wav"
    # Transcript of the reference audio clip — used by the Base model to separate
    # voice identity from speaking style. Leave empty to let the model auto-transcribe.
    speaker_ref_text: str = ""
    # Named built-in Qwen3-TTS voice used when no reference WAV is present.
    # CustomVoice voices: Chelsie, Ethan, Vivian  (Base model ignores this).
    # Leave empty to use the hardcoded default (Chelsie).
    tts_voice: str = "Chelsie"
    # Playback speed multiplier: 1.0 = normal, >1.0 = faster, <1.0 = slower.
    speech_speed: float = 1.0
    # Natural-language style/emotion instruction passed to Qwen3-TTS.
    # Describes how to speak — works on all model variants.
    # Examples: "用温柔亲切的语气说" / "speak warmly and naturally, like chatting with a close friend"
    # Leave empty to use the model's default speaking style.
    speech_instruct: str = ""
    # Sampling temperature: higher = more expressive/varied, lower = more stable/flat.
    # Used as the global default; each emotion can override this via EmotionProfile.tts_temperature.
    # 0.7–0.9 is a good range; 0.0 = fully deterministic (greedy).
    tts_temperature: float = 0.85
    # Top-k sampling — limits token pool to the k most likely choices.
    tts_top_k: int = 50
    # Top-p (nucleus) sampling — keeps the smallest token set whose cumulative probability ≥ p.
    # 0.95 is a good starting point; 1.0 disables it (all tokens eligible).
    # Lower values (e.g. 0.90) constrain sampling and reduce random artefacts.
    tts_top_p: float = 0.95
    # Minimum-p sampling — discards tokens whose probability is below min_p × max_token_prob.
    # Complement to top-p; 0.02–0.05 removes low-probability garbage while preserving diversity.
    # 0.0 disables it.
    tts_min_p: float = 0.02
    # Repetition penalty — prevents stuttering or looping audio artefacts.
    tts_repetition_penalty: float = 1.05

    # Embeddings
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"

    # External APIs
    openai_api_key: str = ""

    # Tool system
    tools_custom_dir: str = "ella/tools/custom"
    max_tool_rounds: int = 5

    # Memory
    goal_ttl_seconds: int = 86400
    knowledge_recall_top_k: int = 5
    # How many minutes back Qdrant conversation recall looks. Only exchanges
    # within this window are surfaced — keeps the LLM focused on the current
    # topic and prevents old conversations from bleeding into today's context.
    # Set to 0 to disable the filter entirely (recall all history).
    knowledge_conv_recall_minutes: int = 15

    # Skill system
    # How many days before learned knowledge in ella_topic_knowledge is considered stale.
    # Stale results are returned with an annotation in Tier 3 recall.
    # LearnSkill also checks this before starting a new research cycle.
    knowledge_freshness_days: int = 30
    # How many hours before a cached webpage .md file in ~/Ella/downloads/ is re-fetched.
    # PDFs are cached permanently (no time check — PDFs don't change once downloaded).
    fetch_cache_hours: int = 24

    # Emotion engine
    # Set EMOTION_ENABLED=false to disable all emotion tracking, contagion, and
    # prompt injection. TTS delivery will fall back to SPEECH_INSTRUCT.
    emotion_enabled: bool = True
    # MySQL DSN — required when emotion_enabled=True.
    # Format: mysql+aiomysql://user:password@host:3306/database
    database_url: str = ""


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
