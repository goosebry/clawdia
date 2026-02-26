"""Celery task worker: routes tasks to Cursor CLI or OpenAI Codex.

Each task carries: {task_id, job_id, task_type, description, chat_id, priority}

Workflow:
  1. Read JobGoal from Redis (Tier 2) to reconstruct context
  2. Load Qwen2.5-7B on-demand → LLM routing decision → unload
  3. Dispatch to appropriate executor (Cursor CLI or OpenAI Codex)
  4. Write StepSummary back to JobGoal in Redis
  5. Send progress/result via Telegram
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from typing import Any

from ella.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="ella.tasks.worker.execute_task",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def execute_task(
    self,
    task_id: str,
    job_id: str,
    task_type: str,
    description: str,
    chat_id: int,
    priority: int = 1,
) -> dict[str, Any]:
    """Main Celery task: route and execute a single task."""
    logger.info(
        "execute_task start: task_id=%s type=%s chat=%d", task_id, task_type, chat_id
    )

    # Each Celery task runs in the same forked worker process but needs its own
    # event loop.  Async clients (Redis, httpx) hold connections tied to the
    # loop that was active when they first connected.  We must:
    #   1. Reset all async singletons BEFORE creating the new loop, so they are
    #      re-created and bound to the new loop on first use inside _run_task.
    #   2. Gracefully close those clients BEFORE closing the loop, so their
    #      open transports/connections are cleanly torn down on the correct loop.
    #   3. Reset singletons again AFTER closing, so the next task starts fresh.
    import ella.communications.telegram.sender as _sender_mod
    import ella.memory.goal as _goal_mod

    # Step 1: discard any singletons left over from the previous task
    _sender_mod._sender = None
    _goal_mod._goal_store = None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _run_task(self, task_id, job_id, task_type, description, chat_id, priority)
        )
    finally:
        # Step 2: close async clients on the still-open loop before closing it
        loop.run_until_complete(_close_async_clients(_sender_mod, _goal_mod))
        loop.close()
        # Step 3: discard closed singletons so the next task gets fresh ones
        _sender_mod._sender = None
        _goal_mod._goal_store = None


async def _close_async_clients(sender_mod: Any, goal_mod: Any) -> None:
    """Gracefully close all async singletons on the current loop before it shuts down.

    Open asyncio transports (Redis StreamWriter, httpx connection pool) must be
    closed on the same loop they were created on.  Closing them here — while the
    loop is still running — prevents 'Event loop is closed' errors on the next task.
    """
    sender = getattr(sender_mod, "_sender", None)
    if sender is not None:
        try:
            await sender.close()
        except Exception:
            pass

    goal_store = getattr(goal_mod, "_goal_store", None)
    if goal_store is not None:
        try:
            await goal_store._redis.aclose()
        except Exception:
            pass


async def _run_task(
    celery_task: Any,
    task_id: str,
    job_id: str,
    task_type: str,
    description: str,
    chat_id: int,
    priority: int,
) -> dict[str, Any]:
    from ella.config import get_settings
    from ella.memory.goal import StepSummary, get_goal_store
    from ella.communications.telegram.sender import get_sender

    settings = get_settings()
    sender = get_sender()
    goal_store = get_goal_store()

    # Read goal context from Redis
    goal = await goal_store.read(job_id)
    context = ""
    if goal:
        context = f"Objective: {goal.objective}\n"
        if goal.steps_done:
            context += "Prior steps: " + "; ".join(
                s.summary[:100] for s in goal.steps_done
            )

    # Route by task_type first (fast, no GPU). LLM routing is used only as
    # a tiebreaker for ambiguous "other" types.
    route = _rule_based_route(task_type)
    if route == "llm":
        route = await _llm_route(description, task_type, context, settings)
    logger.info("Task %s routed to: %s (type=%s)", task_id, route, task_type)

    # Guard: skip routes that require unavailable external services so we
    # don't spam the user with error messages for unsupported task types.
    if route == "codex" and not settings.openai_api_key:
        logger.warning(
            "Task %s skipped — type=%s routes to codex but OPENAI_API_KEY is not set. "
            "Use a skill instead for research/learning tasks.",
            task_id, task_type,
        )
        return {"task_id": task_id, "status": "skipped", "reason": "codex_unavailable"}

    result_text = ""
    success = False

    try:
        if route == "cursor":
            result_text = await _run_cursor(description)
        elif route == "codex":
            result_text = await _run_codex(description, settings)
        elif route == "web_search":
            result_text = await _run_web_search(description)
        else:
            result_text = await _run_shell_task(description)
        success = True
    except Exception as exc:
        logger.exception("Task execution failed: %s", task_id)
        result_text = f"Error: {exc}"
        try:
            await sender.send_message(
                chat_id=chat_id,
                text=f"❌ Task failed ({task_type}): {description[:80]}\n{result_text[:200]}",
            )
        except Exception:
            pass

        # Write failure to goal
        try:
            await goal_store.append_step(
                job_id,
                StepSummary(
                    step_index=len(goal.steps_done) if goal else 0,
                    agent="CeleryWorker",
                    summary=f"FAILED {task_type}: {description[:100]} — {result_text[:100]}",
                ),
            )
        except Exception:
            pass
        raise celery_task.retry(exc=exc)

    # Write success to goal
    try:
        await goal_store.append_step(
            job_id,
            StepSummary(
                step_index=len(goal.steps_done) if goal else 0,
                agent="CeleryWorker",
                summary=f"DONE {task_type} via {route}: {description[:100]}",
            ),
        )
    except Exception:
        logger.exception("Failed to update JobGoal")

    # Summarise raw result through LLM so the reply is natural, not a data dump.
    # This also handles the "no results found" case gracefully.
    reply_text = await _summarise_result(description, task_type, result_text, settings)
    logger.info("Task %s summarised reply: %s chars", task_id, len(reply_text))

    # Detect language for TTS
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", reply_text))
    language: str = "zh" if has_chinese else "en"

    # Send as voice sentences (same path as ReplyAgent) so the task result
    # arrives as natural speech, then send the full raw result as text if needed.
    await _send_voice_reply(chat_id, reply_text, language, sender)

    # Full detail text as follow-up text message(s) — only when there's
    # meaningful additional content beyond the spoken summary.
    if result_text and result_text.strip() != reply_text.strip():
        await _send_detail_text(chat_id, result_text, sender)

    return {"task_id": task_id, "route": route, "success": success, "output": result_text}


def _rule_based_route(task_type: str) -> str:
    """Fast deterministic routing by task_type — no LLM needed.

    Returns 'llm' to signal that LLM routing should be used as a fallback
    for ambiguous types.
    """
    t = task_type.lower()
    if t in ("web_search", "search", "lookup", "research"):
        return "web_search"
    if t in ("coding", "code", "debug", "refactor", "programming"):
        return "cursor"
    if t in ("document", "write", "summarise", "summarize", "draft"):
        # These require an external LLM (Codex/OpenAI). If not available,
        # the codex guard above will skip gracefully. Fall to LLM routing
        # first so it can be remapped to a supported route if possible.
        return "codex"
    if t in ("shell", "command", "file", "system"):
        return "shell"
    # Ambiguous — let LLM decide
    return "llm"


def _keyword_route(description: str, task_type: str) -> str:
    """Keyword-based fallback when LLM routing is unavailable."""
    text = (description + " " + task_type).lower()
    if any(w in text for w in ("search", "look up", "find", "查找", "搜索", "查询")):
        return "web_search"
    if any(w in text for w in ("code", "script", "function", "debug", "refactor", "program")):
        return "cursor"
    if any(w in text for w in ("write", "document", "summarise", "summarize", "draft", "report")):
        return "codex"
    return "shell"


async def _run_web_search(description: str) -> str:
    """Execute a web search task using the built-in web_search tool."""
    try:
        from ella.tools.builtin.web_search import web_search
        result = web_search(query=description)
        return str(result)
    except Exception as exc:
        logger.exception("Web search failed")
        return f"Web search error: {exc}"


async def _summarise_result(
    original_request: str,
    task_type: str,
    raw_result: str,
    settings: Any,
) -> str:
    """Pass the raw task result through the LLM to produce a natural reply.

    Falls back to a formatted plain-text summary if LLM is unavailable.
    """
    # Determine language from original request
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", original_request))
    lang_hint = "Reply in Chinese." if has_chinese else "Reply in English."

    no_result_phrases = ("no results found", "no search results", "search failed")
    is_empty = not raw_result or any(p in raw_result.lower() for p in no_result_phrases)

    if is_empty:
        system = (
            f"You are Ella, a helpful AI assistant. {lang_hint}\n"
            "The user asked for a web search but no results were found. "
            "Apologise briefly and suggest they try rephrasing the query or ask you something else. "
            "Keep your reply concise (2-3 sentences)."
        )
        user_content = f"Original request: {original_request}"
    else:
        system = (
            f"You are Ella, a helpful AI assistant. {lang_hint}\n"
            "Summarise the following search results in a natural, conversational way. "
            "Highlight the key findings relevant to the user's request. "
            "Be concise (3-5 sentences). Do NOT list raw URLs."
        )
        user_content = (
            f"User request: {original_request}\n\n"
            f"Search results:\n{raw_result[:2000]}"
        )

    try:
        import mlx.core as mx
        from mlx_lm import load, generate

        model, tokenizer = load(settings.mlx_chat_model)
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            output = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
            return output.strip() if isinstance(output, str) else str(output).strip()
        finally:
            del model, tokenizer
            try:
                mx.metal.clear_cache()
            except Exception:
                pass
    except Exception:
        logger.warning("LLM summarisation unavailable, sending formatted result")
        if is_empty:
            if has_chinese:
                return f"抱歉，没有找到关于「{original_request[:60]}」的搜索结果。请尝试换一种方式描述您的问题。"
            return f"Sorry, no results were found for '{original_request[:60]}'. Please try rephrasing your query."
        # Plain formatted fallback
        header = f"🔍 Search results for: {original_request[:80]}\n\n"
        return header + raw_result[:600]


async def _llm_route(
    description: str,
    task_type: str,
    context: str,
    settings: Any,
) -> str:
    """Use Qwen2.5-7B on-demand to decide the routing target."""
    model = None
    tokenizer = None
    try:
        import mlx.core as mx
        from mlx_lm import load, generate

        logger.info("Loading routing LLM on-demand: %s", settings.mlx_chat_model)
        model, tokenizer = load(settings.mlx_chat_model)

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a task router. Given a task description, decide the best executor.\n"
                    "Reply with ONLY one of: cursor, codex, shell\n"
                    "cursor = coding tasks (write/modify code, debug, refactor)\n"
                    "codex = document creation, writing, summarisation\n"
                    "shell = file operations, system commands, data processing"
                ),
            },
            {
                "role": "user",
                "content": f"Task type: {task_type}\nContext: {context}\nDescription: {description}",
            },
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        output = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
        output = output.strip().lower() if isinstance(output, str) else "shell"

        if "cursor" in output:
            return "cursor"
        if "codex" in output:
            return "codex"
        return "shell"

    except ImportError:
        logger.warning("mlx-lm not available, using keyword-based route fallback")
        return _keyword_route(description, task_type)
    except Exception as exc:
        # Metal GPU is unavailable in Celery forked workers — fall back silently
        if "MTLCompiler" in str(exc) or "metal" in str(exc).lower() or "loop" in str(exc).lower():
            logger.warning("LLM routing unavailable in worker context, using keyword fallback")
        else:
            logger.exception("LLM routing failed, using keyword fallback")
        return _keyword_route(description, task_type)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


async def _run_cursor(description: str) -> str:
    """Execute a coding task via Cursor headless CLI."""
    try:
        result = subprocess.run(
            ["cursor", "--headless", "--eval", description],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"Cursor exited {result.returncode}: {stderr[:200]}")
        return output or "(cursor completed with no output)"
    except FileNotFoundError:
        raise RuntimeError(
            "Cursor CLI not found. Ensure `cursor` is in PATH and headless mode is enabled."
        )


async def _run_codex(description: str, settings: Any) -> str:
    """Execute a document task via OpenAI Codex (GPT-4o)."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Cannot use Codex routing.")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a document and writing assistant. Complete the task thoroughly.",
            },
            {"role": "user", "content": description},
        ],
        max_tokens=2048,
    )
    return response.choices[0].message.content or "(empty response)"


async def _send_voice_reply(chat_id: int, text: str, language: str, sender: Any) -> None:
    """Send *text* as per-sentence voice messages, mirroring ReplyAgent behaviour.

    Each sentence is synthesised to WAV, converted to OGG/Opus, and sent as a
    separate voice message.  Emoji-only tokens are sent as plain text in order.
    Falls back to a single text message if every TTS attempt fails.
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path
    from ella.tts.qwen3 import is_emoji_only, split_into_sentences, tts_to_wav

    def _wav_to_ogg(wav_path: str) -> str | None:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
            tmp.close()
            ogg_path = tmp.name
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-c:a", "libopus", "-b:a", "48k", "-vbr", "on",
                 "-compression_level", "10", "-application", "voip", ogg_path],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                os.unlink(ogg_path)
                return None
            return ogg_path
        except Exception:
            return None

    sentences = split_into_sentences(text)
    any_voice_sent = False

    for token in sentences:
        if is_emoji_only(token):
            try:
                await sender.send_message(chat_id=chat_id, text=token)
            except Exception:
                pass
            continue

        try:
            await sender.send_chat_action(chat_id, action="record_voice")
        except Exception:
            pass

        wav_path = tts_to_wav(token, language=language)
        ogg_path: str | None = None
        if wav_path and Path(wav_path).exists():
            ogg_path = _wav_to_ogg(wav_path)
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        voice_path = ogg_path or wav_path
        if voice_path and Path(voice_path).exists():
            try:
                await sender.send_voice(chat_id=chat_id, voice_path=voice_path, caption=None)
                any_voice_sent = True
                logger.info("Task voice sentence sent to chat_id=%d: %s", chat_id, token[:60])
            except Exception:
                logger.exception("Failed to send task voice sentence")
            finally:
                try:
                    os.unlink(voice_path)
                except OSError:
                    pass
        else:
            logger.warning("TTS failed for task sentence: %s", token[:60])

    if not any_voice_sent:
        try:
            await sender.send_message(chat_id=chat_id, text=text)
        except Exception:
            logger.exception("Failed to send task text fallback")


async def _send_detail_text(chat_id: int, text: str, sender: Any) -> None:
    """Send full result text split into Telegram-safe chunks (≤ 4096 chars)."""
    _TG_MAX = 4096
    text = text.strip()
    if not text:
        return

    # Simple splitter — prefer newlines, fall back to hard cut
    chunks: list[str] = []
    while text:
        if len(text) <= _TG_MAX:
            chunks.append(text)
            break
        idx = text.rfind("\n", 0, _TG_MAX)
        cut = idx + 1 if idx > _TG_MAX // 2 else _TG_MAX
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        label = f"({i}/{total})\n" if total > 1 else ""
        try:
            await sender.send_message(chat_id=chat_id, text=label + chunk)
        except Exception:
            logger.exception("Failed to send task detail chunk %d/%d", i, total)


async def _run_shell_task(description: str) -> str:
    """Fallback: attempt to execute the description as a shell command."""
    try:
        result = subprocess.run(
            description,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"Exit {result.returncode}: {stderr[:200]}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Shell task timed out after 60s."
    except Exception as exc:
        return f"Shell error: {exc}"
