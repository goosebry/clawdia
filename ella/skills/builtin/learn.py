"""LearnSkill — gap-driven Research → Read → Analyse → Accumulate learning loop.

This is the main learning orchestrator. It:
  1. Pre-checks if sufficiently fresh knowledge already exists (skips re-research if so)
  2. Loops: Research → Analyse → fill gaps → repeat until no gaps or max cycles
  3. Accumulates validated knowledge into ella_topic_knowledge (Qdrant)
  4. Tags each knowledge chunk with a user-chosen sensitivity level
  5. Returns a synthesis summary for the BrainAgent final reply

The Analyse phase is the ONLY place the LLM is invoked — all sourcing and
storage are deterministic.
"""
from __future__ import annotations

import logging
from typing import Any

from ella.skills.base import BaseSkill, SkillContext, SkillResult
from ella.skills.registry import ella_skill
from ella.agents.protocol import LLMMessage

logger = logging.getLogger(__name__)

MAX_LEARN_CYCLES = 3
CHUNK_SIZE_CHARS = 2000  # ~512 tokens at ~4 chars/token


@ella_skill(
    name="learn",
    description=(
        "Deeply research a topic using web search, PDF documents, and Rednote social media posts. "
        "Runs multiple Research → Analyse cycles until all knowledge gaps are filled, "
        "then stores the findings permanently in Ella's long-term knowledge base. "
        "Use when the user wants Ella to learn about or study a topic thoroughly. "
        "Do NOT use for simple one-off questions — this is for deep learning sessions."
    ),
)
class LearnSkill(BaseSkill):
    name = "learn"
    description = (
        "Deeply research a topic using web search, PDF documents, and Rednote social media posts. "
        "Runs multiple Research → Analyse cycles until all knowledge gaps are filled, "
        "then stores the findings permanently in Ella's long-term knowledge base."
    )

    async def run(self, goal: str, context: SkillContext) -> SkillResult:
        run_id = context.run_id
        is_resume = bool(context.notes)  # notes already loaded from checkpoint

        if is_resume:
            logger.info(
                "[LearnSkill] run_id=%s RESUMING goal=%r — %d notes from previous run, starting at cycle %d",
                run_id, goal[:80], len(context.notes), context.cycle,
            )
            await context.send_update(
                f"🔄 Resuming learning session: {goal[:80]}\n"
                f"   ({len(context.notes)} notes already collected — skipping to analysis)"
            )
        else:
            logger.info("[LearnSkill] run_id=%s STARTED goal=%r", run_id, goal[:80])
            await context.send_update(f"📚 Starting learning session: {goal[:80]}")

        # ── Pre-check: do we already have fresh knowledge? (skip on resume) ──
        if not is_resume:
            existing = await _check_existing_knowledge(goal, context)
            if existing:
                await context.send_update(
                    f"💡 I already have some knowledge about this topic "
                    f"(learned {existing['learned_at'][:10]}). "
                    f"I'll research what's new and fill any gaps."
                )

        # ── Research → Analyse loop ───────────────────────────────────────────
        stored_points = 0
        final_summary = ""

        # When resuming, start_cycle is the already-completed cycle so we enter
        # the loop body and go straight to Analyse (research is skipped because
        # context.notes is already populated and non-empty).
        start_cycle = context.cycle if is_resume else 1

        for cycle in range(start_cycle, MAX_LEARN_CYCLES + 1):
            context.cycle = cycle
            await context.checkpoint("research")
            await context.send_update(f"🔄 Learning cycle {cycle}/{MAX_LEARN_CYCLES}…")

            # RESEARCH: skip if we already have notes from this cycle (resume)
            if is_resume and cycle == start_cycle:
                logger.info("[LearnSkill] run_id=%s Skipping research for cycle %d (resuming with existing notes)", run_id, cycle)
            else:
                # RESEARCH: invoke the research sub-skill
                await context.invoke_skill("research", goal)
                await context.checkpoint("read")

            if not context.notes:
                await context.send_update("⚠️ No content found. Trying a different search…")
                await context.invoke_skill("research", f"introduction to {goal}")

            if not context.notes:
                break

            # ANALYSE: LLM synthesises notes and identifies gaps
            await context.send_update("🧠 Analysing what I've learned…")
            await context.checkpoint("analyse")

            analysis = await _analyse(goal, context)
            new_questions = analysis.get("questions", [])
            summary_so_far = analysis.get("summary", "")

            logger.info(
                "[LearnSkill] run_id=%s cycle=%d: %d notes, %d new questions, questions=%s",
                run_id, cycle, len(context.notes), len(new_questions), new_questions[:3],
            )

            if new_questions:
                context.questions.extend(new_questions)

            # GAP RESOLUTION
            remaining_questions = list(context.questions)
            if remaining_questions and cycle < MAX_LEARN_CYCLES:
                # Auto-research each sub-question
                for question in remaining_questions[:3]:  # cap to avoid runaway loops
                    await context.send_update(f"🔍 Looking into: {question[:60]}…")
                    await context.invoke_skill("research", question)
                    context.questions = [q for q in context.questions if q != question]

                # Re-analyse after filling gaps
                analysis = await _analyse(goal, context)
                new_questions = analysis.get("questions", [])
                summary_so_far = analysis.get("summary", "")
                context.questions = new_questions

            # If still gaps, ask the user
            if context.questions and cycle < MAX_LEARN_CYCLES:
                gap_text = "\n".join(f"• {q}" for q in context.questions[:3])
                await context.send_update(
                    f"🤔 I have some knowledge gaps:\n{gap_text}\n"
                    f"Do you know anything about these? (Reply or say 'skip' to continue)"
                )
                user_answer = await context.ask_user(
                    f"I have gaps in my research about '{goal}'. Can you help fill them in?"
                )
                if user_answer and str(user_answer).lower().strip() not in ("skip", "no", "不知道", "不清楚", ""):
                    context.notes.append(f"[User provided context]\n{user_answer}")
                # Whether the user answered, skipped, or timed out — clear the
                # remaining questions so the loop proceeds to Accumulate rather
                # than repeating the ask_user call on the next cycle.
                context.questions = []

            final_summary = summary_so_far

            # If no more questions, exit early
            if not context.questions:
                logger.info("[LearnSkill] run_id=%s No remaining questions after cycle %d — exiting loop", run_id, cycle)
                break

        # ── ACCUMULATE: chunk and store knowledge ─────────────────────────────
        await context.send_update("💾 Storing what I've learned…")
        await context.checkpoint("accumulate")

        # Ask user for sensitivity level — non-blocking with a short 60s window.
        # Defaults to "internal" so learning is never blocked waiting for a tag.
        sensitivity = await _ask_sensitivity(goal, context, len(context.notes))

        stored_points = await _store_knowledge(goal, context, sensitivity)

        # ── FINAL SYNTHESIS ──────────────────────────────────────────────────
        if not final_summary:
            final_summary = await _synthesise(goal, context)

        await context.checkpoint("done")
        await context.send_update(
            f"✅ Learning complete! Stored {stored_points} knowledge chunks about '{goal}'."
        )

        # Unload so BrainAgent's reply turn can reload fresh without OOM.
        _unload_llm()

        return SkillResult(
            summary=final_summary,
            stored_points=stored_points,
            artifacts=list(context.artifacts),
            open_questions=list(context.questions),
        )


# ── LLM helpers ──────────────────────────────────────────────────────────────
# LearnSkill owns a module-level LLM singleton for the duration of its run.
# The running-skill guard in BrainAgent prevents any chat turn from loading
# the model concurrently — so no lock is needed. The model is explicitly
# unloaded at the end of run() so BrainAgent's post-skill reply turn can load
# it fresh without OOM.

_llm_model = None
_llm_tokenizer = None


def _load_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is not None:
        return _llm_model, _llm_tokenizer
    try:
        from mlx_lm import load
        from ella.config import get_settings
        settings = get_settings()
        logger.info("[LearnSkill] Loading LLM: %s", settings.mlx_chat_model)
        _llm_model, _llm_tokenizer = load(settings.mlx_chat_model)
        return _llm_model, _llm_tokenizer
    except Exception as exc:
        logger.warning("[LearnSkill] LLM load failed: %s", exc)
        return None, None


def _unload_llm() -> None:
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        return
    try:
        del _llm_model
        del _llm_tokenizer
        import mlx.core as mx
        mx.clear_cache()
        logger.info("[LearnSkill] LLM unloaded, GPU cache cleared")
    except Exception as exc:
        logger.warning("[LearnSkill] LLM unload failed: %s", exc)
    finally:
        _llm_model = None
        _llm_tokenizer = None


_ANALYSE_CHUNK = 12000   # ~3000 tokens — safe single-call window
_ANALYSE_SYSTEM = (
    "You are an expert research analyst. Your task is to:\n"
    "1. Synthesise the provided research notes into a concise summary\n"
    "2. Identify specific knowledge gaps or questions that need further research\n\n"
    "Respond ONLY with a JSON object on one line:\n"
    '{"summary": "...", "questions": ["question1", "question2"]}\n'
    "Keep summary under 300 words. List at most 3 specific questions. "
    "If the research is comprehensive, return an empty questions array."
)


def _chunk_notes(notes: list[str], chunk_size: int = _ANALYSE_CHUNK) -> list[str]:
    """Split the joined notes into non-overlapping chunks of ≤ chunk_size chars.

    Chunks are split at note boundaries (the separator) so individual notes are
    never truncated mid-sentence.
    """
    sep = "\n\n---\n\n"
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for note in notes:
        piece = (sep if current else "") + note
        if current_len + len(piece) > chunk_size and current:
            chunks.append(sep.join(current))
            current = [note]
            current_len = len(note)
        else:
            current.append(note)
            current_len += len(piece)
    if current:
        chunks.append(sep.join(current))
    return chunks


async def _analyse(goal: str, context: SkillContext) -> dict[str, Any]:
    """Synthesise notes and identify knowledge gaps using the skill LLM.

    Returns {"summary": "...", "questions": ["...", ...]}
    """
    import json
    import re
    from ella.memory.focus import _call_llm_plain

    model, tokenizer = _load_llm()
    if model is None:
        logger.warning("[LearnSkill] LLM unavailable — skipping analysis")
        return {"summary": "", "questions": []}

    chunks = _chunk_notes(context.notes)
    total_chars = sum(len(c) for c in chunks)
    logger.info(
        "[LearnSkill] _analyse: %d notes → %d chars across %d chunk(s)",
        len(context.notes), total_chars, len(chunks),
    )

    all_questions: list[str] = []
    last_summary = ""

    for idx, chunk in enumerate(chunks):
        chunk_label = f"chunk {idx + 1}/{len(chunks)}"
        messages = [
            LLMMessage(role="system", content=_ANALYSE_SYSTEM),
            LLMMessage(
                role="user",
                content=(
                    f"Topic: {goal}\n\n"
                    f"Research notes ({chunk_label}):\n{chunk}\n\n"
                    "Synthesise and identify gaps."
                ),
            ),
        ]
        try:
            raw = _call_llm_plain(model, tokenizer, messages)
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                last_summary = parsed.get("summary", last_summary)
                for q in parsed.get("questions", []):
                    if q and q not in all_questions:
                        all_questions.append(q)
        except Exception as e:
            logger.warning("[LearnSkill] Analyse LLM call failed for %s: %s", chunk_label, e)

    return {"summary": last_summary, "questions": all_questions[:3]}


async def _synthesise(goal: str, context: SkillContext) -> str:
    """Produce a final plain-text summary of everything learned."""
    import re
    from ella.memory.focus import _call_llm_plain

    model, tokenizer = _load_llm()
    if model is None:
        return f"Research completed on '{goal}'. {len(context.notes)} sources collected."

    notes_block = "\n\n---\n\n".join(context.notes)[:10000]
    messages = [
        LLMMessage(
            role="system",
            content="Synthesise the research notes into a clear, engaging summary of 2-4 paragraphs. Plain text only.",
        ),
        LLMMessage(
            role="user",
            content=f"Topic: {goal}\n\nResearch notes:\n{notes_block}",
        ),
    ]
    try:
        result = _call_llm_plain(model, tokenizer, messages)
        result = re.sub(r"<think>[\s\S]*?</think>", "", result, flags=re.IGNORECASE).strip()
        return result
    except Exception:
        return f"Research completed on '{goal}'."


async def _ask_sensitivity(goal: str, context: SkillContext, note_count: int) -> str:
    """Ask the user to tag the sensitivity of the learned knowledge.

    Gives the user 60 seconds to reply; defaults to 'internal' so learning
    is never indefinitely blocked waiting for a sensitivity tag.
    """
    answer = await context.ask_user(
        f"📚 I've finished researching '{goal}' ({note_count} sources). "
        f"How sensitive is this knowledge?\n\n"
        f"  🟢 public  🔵 internal  🟡 private  🔴 secret\n\n"
        f"(Defaulting to 🔵 internal in 60 seconds if no reply)"
    )
    if answer:
        a = str(answer).lower().strip()
        if "public" in a or "🟢" in a:
            return "public"
        if "internal" in a or "🔵" in a:
            return "internal"
        if "private" in a or "🟡" in a:
            return "private"
        if "secret" in a or "🔴" in a:
            return "secret"
    return "internal"


async def _store_knowledge(goal: str, context: SkillContext, sensitivity: str) -> int:
    """Chunk notes and store each chunk in ella_topic_knowledge (Qdrant)."""
    from ella.memory.knowledge import get_knowledge_store, SENSITIVITY_DEFAULT

    store = get_knowledge_store()
    stored = 0

    for note in context.notes:
        # Chunk at CHUNK_SIZE_CHARS boundaries, breaking on newlines
        chunks = _chunk_text(note, CHUNK_SIZE_CHARS)
        source_url = _extract_source_url(note)
        source_type = _infer_source_type(note)

        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                await store.store_topic_knowledge(
                    topic=goal,
                    chunk_text=chunk,
                    source_url=source_url,
                    source_type=source_type,
                    sensitivity=sensitivity,
                    learned_by_chat_id=context.chat_id,
                )
                stored += 1
            except Exception as e:
                logger.warning("[LearnSkill] Failed to store chunk: %s", e)

    logger.info("[LearnSkill] Stored %d knowledge chunks with sensitivity=%s (from %d notes)", stored, sensitivity, len(context.notes))
    return stored


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks of approximately `size` characters, breaking on newlines."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > size and current:
            chunks.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _extract_source_url(note: str) -> str:
    """Try to extract the source URL from a note's header line."""
    import re
    m = re.search(r"https?://[^\s\]]+", note[:200])
    return m.group(0) if m else ""


def _infer_source_type(note: str) -> str:
    """Infer the source type from the note's header marker."""
    first_line = note[:80].lower()
    if "rednote" in first_line:
        return "rednote"
    if "pdf" in first_line:
        return "pdf"
    if "user provided" in first_line:
        return "user_input"
    return "web"


async def _check_existing_knowledge(goal: str, context: SkillContext) -> dict | None:
    """Return the most recent existing knowledge chunk for this topic, if any."""
    try:
        from ella.memory.knowledge import get_knowledge_store
        store = get_knowledge_store()
        results = await store.recall_topic_knowledge(goal, top_k=1)
        if results:
            return results[0]
    except Exception:
        pass
    return None


