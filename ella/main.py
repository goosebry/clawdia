"""Ella AI Agent — entry point.

Startup sequence:
  1. Load configuration from .env
  2. Initialise Qdrant collections (idempotent)
  3. Load Qwen3-TTS singleton (warm-up, ~2 GB)
  4. Load built-in tools into ToolRegistry
  5. Start ToolRegistry file watcher (asyncio background task)
  6. Wire up agent pipeline: IngestionAgent → BrainAgent → ReplyAgent + TaskAgent
  7. Start Telegram long-poll loop

To also run Celery workers (separate process):
    celery -A ella.tasks.celery_app worker --loglevel=info --concurrency=1
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ella.main")


async def _decay_loop() -> None:
    """Background task: drift all users' emotional state toward baseline every 4 hours."""
    from ella.emotion import engine as emotion_engine
    from ella.emotion.store import get_emotion_store
    from ella.memory.identity import get_personality_traits

    while True:
        await asyncio.sleep(4 * 3600)
        try:
            personality = get_personality_traits()
            store = get_emotion_store()
            chat_ids = await store.all_chat_ids()
            for chat_id in chat_ids:
                try:
                    await emotion_engine.apply_decay(chat_id, personality)
                except Exception:
                    logger.exception("Decay failed for chat_id=%d", chat_id)
            logger.info("[Emotion] Decay pass complete — %d user(s) processed", len(chat_ids))
        except Exception:
            logger.exception("[Emotion] Decay loop error")


async def main() -> None:
    from ella.config import get_settings
    from ella.memory.identity import get_identity, watch_identity
    from ella.memory.knowledge import ensure_collections, refresh_identity_knowledge
    from ella.tools.registry import get_registry
    from ella.tts.qwen3 import get_tts

    from ella.agents.reply_agent import ReplyAgent
    from ella.agents.task_agent import TaskAgent
    from ella.agents.brain_agent import BrainAgent
    from ella.agents.ingestion_agent import IngestionAgent
    from ella.communications.telegram.poller import TelegramPoller
    from ella.communications.telegram.sender import get_sender

    settings = get_settings()

    # ── Step 0: Load identity files from ~/Ella/ ─────────────────────────────
    get_identity()   # warm the cache; logs what was loaded
    identity_watcher_task = asyncio.create_task(
        watch_identity(), name="identity-watcher"
    )

    # ── Step 0b: Emotion engine decay background task ────────────────────────
    decay_task = None
    if settings.emotion_enabled:
        if settings.database_url:
            decay_task = asyncio.create_task(_decay_loop(), name="emotion-decay")
            logger.info("Emotion engine enabled — decay loop started (every 4 hours)")
        else:
            logger.warning(
                "EMOTION_ENABLED=true but DATABASE_URL is not set — emotion engine disabled."
            )

    # ── Step 1: Qdrant collections ───────────────────────────────────────────
    logger.info("Initialising Qdrant collections...")
    try:
        await ensure_collections()
        logger.info("Qdrant ready.")
    except Exception:
        logger.exception("Qdrant init failed — Knowledge (Tier 3) will be unavailable.")

    # ── Step 1b: Embed identity files into long-term memory ──────────────────
    logger.info("Refreshing identity knowledge in Qdrant...")
    try:
        await refresh_identity_knowledge()
    except Exception:
        logger.exception("Identity knowledge refresh failed — continuing without it.")

    # ── Step 2: Qwen3-TTS warm-up ────────────────────────────────────────────
    logger.info("Loading Qwen3-TTS (warm-up)...")
    tts = get_tts()
    if tts is None:
        logger.warning("Qwen3-TTS failed to load. Replies will be sent as text.")
    else:
        logger.info("Qwen3-TTS ready.")

    # ── Step 3: ToolRegistry ─────────────────────────────────────────────────
    registry = get_registry()

    # Load built-in tools
    builtin_dir = Path(__file__).parent / "tools" / "builtin"
    registry.load_directory(builtin_dir)
    logger.info("Loaded %d built-in tool(s).", len(registry.get_schemas()))

    # Load custom tools (user-dropped)
    custom_dir = Path(settings.tools_custom_dir)
    registry.load_directory(custom_dir)

    # Start hot-reload watcher as a background task
    watcher_task = asyncio.create_task(
        registry.watch(builtin_dir, custom_dir),
        name="tool-registry-watcher",
    )
    logger.info("ToolRegistry watcher started on: %s, %s", builtin_dir, custom_dir)

    # ── Step 3b: SkillRegistry ────────────────────────────────────────────────
    skill_watcher_task = None
    try:
        from ella.skills.registry import get_skill_registry

        skill_registry = get_skill_registry()
        skills_builtin_dir = Path(__file__).parent / "skills" / "builtin"
        skills_custom_dir = Path(__file__).parent / "skills" / "custom"
        skill_registry.load_directory(skills_builtin_dir)
        skill_registry.load_directory(skills_custom_dir)
        logger.info("Loaded %d built-in skill(s): %s", len(skill_registry.all_names()), skill_registry.all_names())

        skill_watcher_task = asyncio.create_task(
            skill_registry.watch(skills_builtin_dir, skills_custom_dir),
            name="skill-registry-watcher",
        )
        logger.info("SkillRegistry watcher started.")
    except Exception:
        logger.exception("SkillRegistry init failed — skill system will be unavailable.")

    # ── Step 3c: SkillExecutionRegistry — auto-resume interrupted runs ────────
    # On restart (machine reboot, crash, OOM), automatically resume any skill
    # executions that were running or that failed within the last 24 hours.
    # Ella picks up exactly where she left off — no user re-trigger needed.
    try:
        from ella.skills.execution import get_execution_registry as _get_exec_reg
        from ella.skills.checkpoint import get_checkpoint_store as _get_cp_store
        from ella.communications.telegram.sender import get_sender as _get_sender
        from ella.agents.protocol import SessionContext  # noqa: F401

        if settings.database_url:
            exec_reg = await _get_exec_reg()
            cp_store  = await _get_cp_store()
            sender    = _get_sender()

            # Collect all resumable runs: paused + recently failed (within 24h)
            all_resumable = await cp_store.list_resumable(chat_id=None, max_age_hours=24)

            # Only resume runs that have meaningful notes — zero-note failed runs
            # are stale and should be re-triggered fresh by the user.
            all_resumable = [cp for cp in all_resumable if cp.notes]

            # list_resumable returns newest-first; take only the single most recent
            # run to avoid hammering GPU with back-to-back LLM loads on startup.
            to_resume = all_resumable[:1]

            if all_resumable and len(all_resumable) > 1:
                skipped = [cp.run_id[:8] for cp in all_resumable[1:]]
                logger.info(
                    "[Skills] %d resumable run(s) found — resuming only the most recent. "
                    "Skipped (user can re-trigger): %s",
                    len(all_resumable), skipped,
                )

            if to_resume:
                logger.info("[Skills] Found %d resumable skill execution(s) at startup — auto-resuming sequentially", len(to_resume))

                from ella.tools.registry import get_registry as _get_tool_reg
                tool_reg = _get_tool_reg()

                async def _resume_all_sequentially(runs: list) -> None:
                    """Resume each interrupted skill one at a time to avoid GPU OOM."""
                    for cp in runs:
                        _chat_id = cp.chat_id
                        _run_id  = cp.run_id

                        try:
                            await sender.send_message(
                                _chat_id,
                                f"🔄 I'm back! Resuming '{cp.goal[:60]}' "
                                f"from where I left off ({len(cp.notes)} notes already collected)…"
                            )
                        except Exception:
                            logger.warning("[Skills] Could not notify chat_id=%d for run %s", _chat_id, _run_id)

                        _session = SessionContext(chat_id=_chat_id)

                        async def _send_update(msg: str, cid: int = _chat_id) -> None:
                            try:
                                await sender.send_message(cid, msg)
                            except Exception:
                                pass

                        async def _ask_user(prompt: str, cid: int = _chat_id, rid: str = _run_id) -> str | None:
                            from ella.skills.checkpoint import get_checkpoint_store as _gcs
                            _store = await _gcs()
                            await _store.set_pending_reply(cid, rid, prompt, ttl=120)
                            try:
                                await sender.send_message(cid, prompt)
                            except Exception:
                                pass
                            elapsed = 0
                            while elapsed < 120:
                                await asyncio.sleep(3)
                                elapsed += 3
                                slot = await _store.get_pending_reply(cid)
                                if slot is None:
                                    return None
                                if slot.get("answer") is not None:
                                    await _store.clear_pending_reply(cid)
                                    return slot["answer"]
                            await _store.clear_pending_reply(cid)
                            return None

                        logger.info(
                            "[Skills] Auto-resuming run_id=%s skill=%s goal=%r chat_id=%d notes=%d",
                            _run_id, cp.skill_name, cp.goal[:60], _chat_id, len(cp.notes),
                        )
                        try:
                            # Await each resume — only start next after this one completes/fails
                            await exec_reg.resume(
                                run_id=_run_id,
                                session=_session,
                                tool_executor=tool_reg,
                                send_update=_send_update,
                                ask_user=_ask_user,
                            )
                        except Exception:
                            logger.exception("[Skills] Auto-resume failed for run_id=%s", _run_id)

                # Run the sequential resume chain as a background task so startup completes
                asyncio.create_task(_resume_all_sequentially(to_resume), name="startup-skill-resume")
        else:
            logger.info("[Skills] DATABASE_URL not set — skill checkpointing disabled.")
    except Exception:
        logger.exception("SkillExecutionRegistry startup scan failed — continuing without it.")

    # ── Step 4: Wire agent pipeline ──────────────────────────────────────────
    reply_agent = ReplyAgent()
    task_agent = TaskAgent()
    brain_agent = BrainAgent(reply_agent=reply_agent, task_agent=task_agent)
    ingestion_agent = IngestionAgent(brain_agent=brain_agent)

    # ── Step 5: Telegram poller ──────────────────────────────────────────────
    poller = TelegramPoller(ingestion_agent=ingestion_agent)
    poller_task = asyncio.create_task(poller.run(), name="telegram-poller")
    logger.info("Ella is running. Bot token: ...%s", settings.telegram_bot_token[-8:])

    # ── Graceful shutdown ────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("Shutdown signal received.")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await shutdown_event.wait()

    logger.info("Shutting down...")
    identity_watcher_task.cancel()
    watcher_task.cancel()
    poller_task.cancel()
    if decay_task is not None:
        decay_task.cancel()
    if skill_watcher_task is not None:
        skill_watcher_task.cancel()

    tasks_to_gather = [identity_watcher_task, watcher_task, poller_task]
    if decay_task is not None:
        tasks_to_gather.append(decay_task)
    if skill_watcher_task is not None:
        tasks_to_gather.append(skill_watcher_task)
    await asyncio.gather(*tasks_to_gather, return_exceptions=True)
    await poller.close()
    logger.info("Ella stopped.")


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
