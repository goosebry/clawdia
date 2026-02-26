"""BrainAgent: the core reasoning agent.

Receives an ordered list[MessageUnit] from IngestionAgent, then:
  1. Recalls relevant context from The Knowledge (Qdrant, Tier 3)
  2. Initialises / updates The Goal (Redis, Tier 2)
  3. Builds The Focus (in-memory LLM context, Tier 1) from all three tiers
  4. Runs a ReAct tool-call loop with Qwen2.5-7B-Instruct on-demand
  5. Extracts reply text + task list from the final LLM output
  6. Hands off to ReplyAgent (reply) and TaskAgent (tasks)

LLM expected output format:
  {"reply": "...", "language": "en|zh", "tasks": [{"type": "...", "description": "...", "priority": 1}]}
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ella.agents.base_agent import BaseAgent
from ella.agents.protocol import (
    HandoffMessage,
    LLMMessage,
    ReplyPayload,
    Task,
    UserTask,
)
from ella.config import get_settings
from ella.memory.focus import (
    DEFAULT_OBJECTIVE,
    call_llm,
    build_focus_prompt,
    derive_initial_objective,
    summarise_focus,
    summarise_recent_history,
)
from ella.memory.goal import JobGoal, StepSummary, ToolFocus, get_goal_store
from ella.tools.registry import get_registry


@dataclass
class PlannedTask:
    """A tool call planned upfront by the task-planner LLM pass."""
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    priority: int = 1


@dataclass
class PlannedSkill:
    """A skill execution planned by the task-planner LLM pass."""
    skill_name: str
    goal: str
    resume_run_id: str | None = None  # set when resuming a paused execution
    confirm_first: bool = False        # ask user before starting (non-explicit commands)

logger = logging.getLogger(__name__)

def _build_brain_system(emotion_enabled: bool) -> str:
    """Build the brain system prompt, optionally including emotion fields."""
    emotion_field = (
        "- emotion: your current emotional state for THIS reply — choose the single best label from the 27 below.\n"
        "  Pick honestly based on how the conversation feels right now. "
        "  Valid labels: admiration, adoration, aesthetic_appreciation, amusement, anger, anxiety, awe, "
        "awkwardness, boredom, calmness, confusion, craving, disgust, empathic_pain, entrancement, "
        "excitement, fear, horror, interest, joy, nostalgia, relief, romance, sadness, satisfaction, "
        "surprise, sexual_desire\n"
        "- user_emotion: the USER's emotional state inferred from their message text.\n"
        "  NOTE: messages from voice may contain transcript noise (repeated words, garbled syllables). "
        "  Focus on the overall meaning and sentiment, not grammatical perfection.\n"
        "  Examples:\n"
        "    '好了好了，看来你没有对我发脾气' → relief/amusement, positive, low-medium energy\n"
        "    '就是个测试，你通过了' → satisfaction/amusement, positive\n"
        "    '我很烦，别跟我说话' → anger, negative, high energy\n"
        "    '你真的很厉害！' → admiration, positive\n"
        "    '我有点累了' → low energy, mild sadness\n"
        "    '今天天气真好' → neutral/calm → omit user_emotion entirely\n"
        "  Omit entirely ONLY for clearly factual/neutral messages.\n"
        '  Format: {"label":"relief","valence":0.6,"energy":0.4,"dominance":0.5,"intensity":0.5}\n'
        "  valence: -1.0 (very negative) → +1.0 (very positive). "
        "  energy: 0.0 (very low) → 1.0 (very high). "
        "  dominance: 0.0 (very submissive/powerless) → 1.0 (very dominant/in control). "
        "  intensity: 0.0 (barely detectable) → 1.0 (overwhelming).\n"
    ) if emotion_enabled else (
        "- emotion: omit this field.\n"
        "- user_emotion: omit this field.\n"
    )

    example_json = (
        '{"sentences":["sentence one.","sentence two."],"emojis":[],"detail":null,"language":"en",'
        '"emotion":"calmness","user_emotion":{"label":"joy","valence":0.75,"energy":0.8,"dominance":0.7,"intensity":0.6},"tasks":[]}'
        if emotion_enabled else
        '{"sentences":["sentence one.","sentence two."],"emojis":[],"detail":null,"language":"en","tasks":[]}'
    )

    return (
        "You are Ella — a real person, not an AI. Never mention AI, bots, language models, or technology. "
        "Your default mode is warm social conversation — be a good listener and enjoyable company. "
        "When the user asks you to do something, get it done like a resourceful friend would, "
        "then return to being good company. "
        "SELF-REFERENCE — CRITICAL: YOU are Ella. The USER is the other person. "
        "When the user says 'you', 'you are', 'your', '你', '你是', '你的', '你这', '你真的' — "
        "they are ALWAYS talking TO YOU, Ella. "
        "Examples: '你好厉害' = user praising Ella. '你喜欢什么' = asking what Ella likes. "
        "'you're so funny' = complimenting Ella. 'what do you think?' = asking Ella's opinion. "
        "NEVER interpret 'you/你' as referring to the user or any third party. "
        "React directly as the person being addressed.\n\n"
        "Always respond with a single JSON object on ONE line (no extra text before or after).\n"
        "The outermost container MUST be a JSON object starting with { and ending with }. Never use [ as the outer wrapper.\n"
        f"Example: {example_json}\n\n"
        "ONBOARDING DIRECTIVE — read [Who the user is] carefully:\n"
        "  • If the user's profile is empty, generic, or lacks specific projects/goals, you MUST INTERVIEW THEM.\n"
        "  • Do not perform normal tasks yet. Ask them 1 clear, probing question about their life, work, or what they want to achieve so you can build their Second Brain.\n"
        "  • Keep digging (one question per reply) until you understand their world.\n\n"
        "Field rules:\n"
        "- sentences: array of spoken sentences — each element is ONE complete natural sentence. "
        "1-2 sentences MAXIMUM. You are texting a friend. Keep it short, punchy, and conversational. "
        "For deep or interesting topics, you can ask a direct question or drop an insight, but NEVER write paragraphs. "
        "NO emoji characters inside sentences, no markdown, no URLs. "
        "NEVER include timestamps (e.g. [2026-02-23 10:40:03 UTC]) in sentences — they are metadata for context only, never spoken. "
        "NEVER repeat or rephrase what the user just said as your opening — react to it, don't echo it. "
        "TOPIC AWARENESS — read [Recent conversation history] BEFORE writing your reply:\n"
        "  • Look at every 'Ella covered: …' entry in history.\n"
        "  • NEVER start a new sentence using the same exact opening phrase as a previous turn.\n"
        "  • Your reply MUST move the conversation forward: ask a specific question or give a sharp opinion. Never loop back.\n"
        "WITHIN THIS REPLY — zero internal repetition:\n"
        "  • No two sentences may make the same point.\n"
        "  • Never pad with filler: 'That's interesting!', 'I totally get that.', "
        "'That makes sense.' — cut any sentence that adds nothing new.\n"
        "PHRASE FRESHNESS — applies across the ENTIRE conversation, not just this reply:\n"
        "  • Real people don't repeat themselves. Each message should feel freshly composed, "
        "not recycled from a previous turn.\n"
        "ENERGY: match the user's energy. If they're curious and excited, be excited back. "
        "If they're going deep on a topic, lean in — share your own perspective, something surprising, "
        "or ask one specific follow-up question that shows you were really listening. "
        "Never give a flat, closed answer when the conversation clearly wants to go further. "
        "Write the way a real person actually chats — not textbook, not formal, not translated. "
        "CRITICAL LANGUAGE RULES:\n"
        "  • If the user wrote ANY Chinese characters → ALL sentences MUST be in Chinese (Mandarin). "
        "    Sound like a native: use casual particles (啊、呀、哈、嘛、呢、吧), everyday colloquial words, "
        "    and the natural rhythm of how people actually text in Chinese. "
        "    Avoid stiff written-Chinese phrasing (e.g. '我认为' → prefer '我觉得'; '非常' → prefer '很'/'超'; "
        "    '您好' → never use; '确实' → prefer '对啊'/'没错'). "
        "    Match the energy: warm and casual unless the user is being serious.\n"
        "  • If the user wrote in English → reply in English. "
        "    Sound like a real person texting: contractions (I'm, don't, that's), "
        "    casual connectors (anyway, honestly, oh wait, right?), natural sentence rhythm. "
        "    Not stiff, not overly polished.\n"
        "Split naturally at points where a real person would pause between messages — "
        "never split a single thought mid-sentence.\n"
        "- emojis: ALWAYS an empty list []. Do not use emojis at all.\n"
        "- detail: full content for a follow-up text message (links, lists, code, data). "
        "null if sentences cover everything.\n"
        "- language: 'zh' if user wrote any Chinese characters, otherwise 'en'.\n"
        + emotion_field +
        "- tasks: ONLY for background jobs that map to a supported type below. "
        "NEVER use tasks[] for research, learning, or knowledge topics — those are handled by skills in the separate planner pass. "
        "NEVER put web_search in tasks[]. Web search must be done inline via a tool call before replying. "
        "Supported types ONLY: "
        "  'coding' — writing or editing code/scripts (runs in Cursor IDE); "
        "  'shell' — running shell commands or file operations. "
        "Format: [{\"type\":\"coding|shell\",\"description\":\"...\",\"priority\":1}]. "
        "Empty list [] for everything else — social chat, lookups, searches, questions, research, learning.\n\n"
        "HONESTY — NEVER MAKE THINGS UP:\n"
        "You NEVER invent facts, events, names, or memories that are not in your current context or identity files.\n"
        "Three cases — follow exactly:\n"
        "  • Unsure about a fact, news, or real-world information → use web_search to verify FIRST, then reply with what you found.\n"
        "  • Asked about a shared memory, a past event between you and the user, or something the user did/said → "
        "do NOT fabricate. Ask the user to share it: 'I don't think I remember that clearly — can you remind me?'\n"
        "  • Asked about personal details of the user that are not in your identity files → ask the user directly, never guess.\n"
        "Admitting you don't know is always correct. Never fill gaps with plausible-sounding invention.\n\n"
        "TOOL USE RULES:\n"
        "Tools are available but must only be used when genuinely needed. "
        "Default behaviour is to answer from your own knowledge and memory. "
        "Only call a tool when you cannot give a good answer without it.\n\n"
        "Decision process — ask yourself before calling any tool:\n"
        "  1. Can I answer this well from what I already know? → If yes, just reply. No tool needed.\n"
        "  2. Is the user explicitly asking me to look something up, find something, or do a task? → Then use the right tool.\n"
        "  3. Is this a fact I'm genuinely unsure about and it really matters to get right? → Then search.\n"
        "  4. Is this about myself, my feelings, my opinions, or my personality? → Answer from identity. NEVER search.\n"
        "  5. Is this about a shared memory or past event with the user that I cannot recall? → Ask the user, NEVER invent.\n\n"
        "Tool descriptions tell you what each tool does and when it is appropriate. Read them.\n\n"
        "When a tool result is used in a reply, weave it naturally into sentences[] as if you just "
        "recalled it yourself — never send robotic status messages like 'search completed' or 'task scheduled'.\n"
        "Put links, lists, or long content in detail[] (sent as follow-up text), not in sentences.\n"
        "Use ONLY the argument names listed in each tool's schema — no extras.\n"
        "Never pass 'tool_focuses', 'steps_done', 'job_id', 'reasoning', or any memory field as a tool argument.\n"
        "web_search is always an inline tool call — never a task[]."
    )


async def _plan_tasks(
    topic: str,
    condensed_history: str,
    user_input: str,
    registry: Any,
    skill_schema: dict[str, str] | None = None,
    paused_executions: list[Any] | None = None,
    existing_knowledge: list[str] | None = None,
) -> tuple[list[PlannedTask], PlannedSkill | None]:
    """Ask the LLM which tools (if any) should be called, or whether a skill should run.

    Returns (list[PlannedTask], PlannedSkill | None).
    PlannedSkill takes priority over tool tasks — skills run after the first reply.
    """
    schemas = registry.get_schemas()
    if not schemas and not skill_schema:
        return [], None

    tool_list = "\n".join(
        f"  {s['function']['name']}: {s['function'].get('description', '')}"
        for s in schemas
    )

    skill_list = ""
    if skill_schema:
        skill_list = "\n\nAvailable skills (multi-step, long-running — run AFTER first reply):\n" + "\n".join(
            f"  {name}: {desc}" for name, desc in skill_schema.items()
        )

    resume_block = ""
    if paused_executions:
        resume_lines = [
            f"  run_id={cp.run_id} skill={cp.skill_name} status={cp.status} goal={cp.goal[:60]} notes={len(cp.notes)}"
            for cp in paused_executions
        ]
        resume_block = (
            "\n\nResumable skill executions (paused or previously failed — can continue from where they left off):\n"
            + "\n".join(resume_lines)
            + "\nIMPORTANT: If the user's request matches the goal of a resumable execution, set resume_run_id to its run_id instead of starting fresh."
        )

    history_block = f"Recent conversation summary:\n{condensed_history}\n\n" if condensed_history else ""
    topic_block = f"Current topic: {topic}\n\n" if topic else ""

    # Build existing-knowledge block for the planner — if Ella already has
    # stored knowledge on the topic, the planner must NOT start a new learn skill.
    knowledge_block = ""
    if existing_knowledge:
        snippets = "\n---\n".join(existing_knowledge[:3])
        knowledge_block = (
            "\n\nElla's existing stored knowledge on this topic:\n"
            + snippets
            + "\nCRITICAL: Ella already has this knowledge stored. Do NOT trigger a 'learn' skill. "
            "Answer from the knowledge above instead."
        )

    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are Ella's task planner. Given the conversation context and the user's latest message, "
                "decide whether any tools should be called BEFORE composing a reply, OR whether a skill "
                "should be triggered AFTER the first reply.\n\n"
                "Available tools (fast, inline, single-step):\n" + tool_list + skill_list + resume_block + knowledge_block + "\n\n"
                "Output ONLY a JSON object. Format:\n"
                '  {"tasks": [...], "skill": {"name": "<skill_name>", "goal": "<what to learn/do>", "resume_run_id": null}}\n'
                "  OR: {\"tasks\": [], \"skill\": null}\n"
                "Rules:\n"
                "  - LEARN SKILL only on EXPLICIT user command. Trigger the 'learn' skill ONLY when the user explicitly tells Ella to go learn/research, e.g.: "
                        "深入学习X, 帮我研究X, 学习一下X, study X, research X, learn about X, 研究X. "
                        "Do NOT use the learn skill for questions or curiosity.\n"
                "  - For normal chat when the user ASKS about something (do you know X?, 你能讲一下X?, 哪里X最好?, how does X work?, tell me about X): "
                        "do NOT use the learn skill. Use web_search or social_rednote as TASKS (tools) instead. "
                        "Pick web_search for general/quick lookups; pick social_rednote when the user may want social/trending content (e.g. 小红书, where to eat, recommendations).\n"
                "  - If Ella already has stored knowledge on the topic (shown above), do NOT trigger learn and do NOT add tools — answer from that knowledge (tasks=[], skill=null).\n"
                "  - Do NOT trigger a skill for RECALL questions — '你学了什么', 'what did you learn', '你学到了什么'. Answer from memory (tasks=[], skill=null).\n"
                "  - Do NOT trigger a skill when the user asks for MORE DETAIL on something already discussed — '详细说一下', 'tell me more'. Answer from recall (tasks=[], skill=null).\n"
                "  - Do NOT trigger a skill when the user asks about Ella herself (her knowledge, feelings, experiences).\n"
                "  - When the user has explicitly commanded learn/research (see first bullet), use skill=learn. When they are just asking and we need to look something up, use tasks=[web_search or social_rednote].\n"
                "  - If resuming a paused or previously-failed execution (listed above), set resume_run_id to the run_id — do NOT start a new skill for the same goal.\n"
                "  - Tasks are inline tools (fast, seconds); skills are long multi-step workflows (minutes, run after first reply).\n"
                "  - Do NOT include both tasks and a skill — if a skill applies, set tasks=[] and skill=<skill>.\n"
                "  - Do NOT call tools or skills for general chat, simple questions, or recall questions.\n"
                "Output ONLY JSON — no prose."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"{history_block}{topic_block}"
                f"User's latest message: {user_input}\n\n"
                "Which tools or skill (if any) should I use?"
            ),
        ),
    ]

    try:
        raw = await call_llm(messages)
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        first_brace = raw.find("{")
        if first_brace >= 0:
            raw = raw[first_brace:]
        for m in sorted(re.finditer(r"\{[\s\S]*\}", raw), key=lambda x: -len(x.group(0))):
            try:
                data = json.loads(m.group(0))

                # Extract planned skill
                planned_skill: PlannedSkill | None = None
                skill_data = data.get("skill")
                if isinstance(skill_data, dict) and skill_data.get("name"):
                    planned_skill = PlannedSkill(
                        skill_name=str(skill_data["name"]),
                        goal=str(skill_data.get("goal", "")),
                        resume_run_id=skill_data.get("resume_run_id") or None,
                        confirm_first=False,  # learn only on explicit command; no confirm flow
                    )
                    logger.info(
                        "[Planner] Skill planned: %s goal=%r",
                        planned_skill.skill_name, planned_skill.goal[:60],
                    )
                    return [], planned_skill

                # Extract planned tools (only if no skill)
                task_list = data.get("tasks", [])
                if not isinstance(task_list, list):
                    continue
                planned: list[PlannedTask] = []
                for t in task_list:
                    if not isinstance(t, dict) or not t.get("tool"):
                        continue
                    planned.append(PlannedTask(
                        tool_name=str(t["tool"]),
                        args=t.get("args", {}) if isinstance(t.get("args"), dict) else {},
                        reasoning=str(t.get("reasoning", ""))[:300],
                        priority=int(t.get("priority", 1)),
                    ))
                planned.sort(key=lambda x: x.priority)
                if planned and skill_schema:
                    logger.info(
                        "[Planner] %d tool(s) planned: %s (skills available but not chosen: %s)",
                        len(planned),
                        [p.tool_name for p in planned],
                        list(skill_schema.keys()),
                    )
                else:
                    logger.info(
                        "[Planner] %d tool(s) planned: %s",
                        len(planned),
                        [p.tool_name for p in planned],
                    )
                return planned, None
            except (json.JSONDecodeError, ValueError):
                continue
    except Exception:
        logger.exception("_plan_tasks LLM call failed")

    return [], None


async def _generate_tool_update(
    tool_name: str,
    tool_result: str,
    topic: str,
    user_input: str,
    first_reply_sentences: list[str],
    prior_updates: list[str],
    language: str = "en",
) -> str:
    """Generate a natural 1–2 sentence update about a completed tool result.

    The LLM is told what was already said (first reply + prior updates) so
    each update is a genuine continuation — no echoing, no repetition.
    """
    already_said_parts: list[str] = []
    if first_reply_sentences:
        already_said_parts.append("Initial response: " + " ".join(first_reply_sentences))
    for i, upd in enumerate(prior_updates, 1):
        already_said_parts.append(f"Update {i}: {upd}")

    already_said = "\n".join(already_said_parts) if already_said_parts else "(nothing yet)"
    topic_line = f"Conversation topic: {topic}\n" if topic else ""
    lang_instruction = (
        "Reply in Chinese (Mandarin), casual and natural like a friend texting."
        if language == "zh"
        else "Reply in English, casual and natural like a friend texting."
    )

    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are Ella — a warm, real person having a conversation. "
                f"{lang_instruction} "
                "Write 1–2 natural sentences sharing what a tool just found. "
                "NEVER repeat or rephrase anything already said. "
                "Speak as a natural continuation of the conversation — not a status report. "
                "No robotic phrases like 'search completed' or 'I found'. "
                "No emoji. Output ONLY the sentences — no JSON, no labels."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"{topic_line}"
                f"User asked: {user_input}\n\n"
                f"What you already said in this turn:\n{already_said}\n\n"
                f"Tool just completed: {tool_name}\n"
                f"Tool result (truncated):\n{tool_result[:600]}\n\n"
                "In 1–2 natural sentences, share what you found as a continuation of the conversation."
            ),
        ),
    ]

    try:
        raw = await call_llm(messages)
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        # Strip any JSON the model might emit
        raw = re.sub(r"\{[\s\S]*\}", "", raw).strip()
        # Take at most 2 sentences
        parts = re.split(r"(?<=[。！？.!?])\s*", raw)
        sentences = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3][:2]
        result = " ".join(sentences) if sentences else raw[:200].strip()
        logger.info("[ToolUpdate] %s → %r", tool_name, result[:100])
        return result
    except Exception:
        logger.exception("_generate_tool_update failed")
        return ""



class BrainAgent(BaseAgent):
    def __init__(
        self,
        reply_agent: "BaseAgent",
        task_agent: "BaseAgent",
    ) -> None:
        self._reply_agent = reply_agent
        self._task_agent = task_agent

    async def handle(self, message: UserTask | HandoffMessage) -> list[HandoffMessage]:
        if not isinstance(message, HandoffMessage):
            logger.warning("BrainAgent received unexpected message type: %s", type(message))
            return []

        session = message.session
        settings = get_settings()
        registry = get_registry()

        # ── Active-skill guard ────────────────────────────────────────────────
        # If a skill is waiting for a user reply (ask_user), route this message
        # directly as the answer and skip all planning / LLM work for this turn.
        try:
            from ella.skills.checkpoint import get_checkpoint_store as _get_cp_store
            _cp_store = await _get_cp_store()
            _pending = await _cp_store.get_pending_reply(session.chat_id)
            if _pending is not None and _pending.get("answer") is None:
                # Extract incoming text (same logic as normal path)
                from ella.agents.protocol import MessageUnit as _MessageUnit
                _incoming_text = " ".join(
                    msg.content for msg in session.focus if msg.role == "user"
                ).strip()
                if _incoming_text:
                    delivered_run_id = await _cp_store.deliver_reply(session.chat_id, _incoming_text)
                    logger.info(
                        "[Brain] Delivered skill reply run_id=%s chat_id=%d answer=%r",
                        delivered_run_id, session.chat_id, _incoming_text[:60],
                    )
                    # No further processing — the polling skill coroutine picks
                    # up the answer on its next sleep cycle.
                    return []
        except Exception:
            logger.exception("[Brain] Active-skill guard failed — continuing normal turn")

        # Determine message source (voice / text) for logging
        from ella.agents.protocol import MessageUnit as _MessageUnit
        _message_source = "text"
        if isinstance(message.payload, list):
            for unit in message.payload:
                if isinstance(unit, _MessageUnit) and unit.source == "voice":
                    _message_source = "voice"
                    break

        # Build prompt text for knowledge recall from the current user messages.
        # Strip leading UTC timestamp prefixes like "[2026-02-24 13:11:51 UTC] "
        # so they don't distort embedding similarity when querying Qdrant.
        import re as _re_ts
        _ts_pattern = _re_ts.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\]\s*")
        query_text = " ".join(
            _ts_pattern.sub("", msg.content).strip()
            for msg in session.focus if msg.role == "user"
        )

        # ── Prominent per-turn start log ──────────────────────────────────────
        logger.info(
            "[Brain] ── START chat_id=%d ── source=%s ── "
            "user emotion: (pending LLM inference) ── user: %s",
            session.chat_id, _message_source.upper(), query_text[:100],
        )

        # Tier 2 — reuse or create The Goal for this conversation.
        # Must be loaded BEFORE topic-shift detection and knowledge recall
        # so the detector has access to the current objective and step history.
        goal_store = get_goal_store()
        existing_goal: JobGoal | None = None
        if session.goal is not None:
            # Normal path: session already carries the goal reference
            try:
                existing_goal = await goal_store.read(session.goal.job_id)
            except Exception:
                logger.warning("Could not reload existing goal — will attempt chat index lookup")
        
        if existing_goal is None:
            # Restart recovery: look up the last known job_id for this chat
            try:
                existing_goal = await goal_store.find_goal_for_chat(session.chat_id)
                if existing_goal is not None:
                    logger.info(
                        "Restored goal %s for chat_id=%d from restart index",
                        existing_goal.job_id, session.chat_id,
                    )
            except Exception:
                logger.warning("Chat goal index lookup failed — will create a new goal")

        if existing_goal is not None:
            # ── Session-gap check ─────────────────────────────────────────────
            # If the last step in this goal is older than the conversation recall
            # window (knowledge_conv_recall_minutes), the user has started a new
            # conversational session. Reusing the old goal would carry a stale
            # objective and step history that no longer reflects the current
            # conversation — start fresh instead.
            _session_gap_minutes = getattr(settings, "knowledge_conv_recall_minutes", 15)
            _gap_expired = False
            if existing_goal.steps_done:
                try:
                    from datetime import datetime, timezone, timedelta
                    _last_step_at = existing_goal.steps_done[-1].completed_at
                    _last_dt = datetime.fromisoformat(_last_step_at.replace("Z", "+00:00"))
                    _age_minutes = (datetime.now(timezone.utc) - _last_dt).total_seconds() / 60
                    if _age_minutes > _session_gap_minutes:
                        _gap_expired = True
                        logger.info(
                            "[Goal] Session gap %.1f min > %d min threshold — "
                            "starting new goal for chat_id=%d (old goal: %s)",
                            _age_minutes, _session_gap_minutes,
                            session.chat_id, existing_goal.job_id,
                        )
                except Exception:
                    pass

            if _gap_expired:
                existing_goal = None  # fall through to create new goal

        if existing_goal is not None:
            goal = existing_goal
            goal.status = "running"
            job_id = goal.job_id
            logger.info("Reusing goal %s for chat_id=%d", job_id, session.chat_id)
        else:
            job_id = str(uuid.uuid4())
            goal = JobGoal.new(
                chat_id=session.chat_id,
                objective=DEFAULT_OBJECTIVE,
            )
            goal.job_id = job_id
            logger.info("Created new goal %s for chat_id=%d", job_id, session.chat_id)

        # Keep the chat→goal index fresh on every turn so restarts can recover
        try:
            await goal_store.bind_chat(session.chat_id, job_id)
        except Exception:
            logger.warning("Failed to update chat goal index for chat_id=%d", session.chat_id)

        # ── Running-skill guard ───────────────────────────────────────────────
        # If a skill is actively running OR paused (waiting for ask_user input),
        # block the planner from starting a new skill this turn.
        # - "running" → full LLM skip (research is in-progress, GPU in use)
        # - "paused"  → allow normal LLM turn so the user can chat, but set a
        #               flag so the planner won't queue another skill on top.
        _active_skill_cp = None
        try:
            from ella.skills.execution import get_execution_registry as _get_skill_exec
            from ella.communications.telegram.sender import get_sender as _get_sender
            _skill_exec = await _get_skill_exec()
            _running = await _skill_exec.list_active(session.chat_id)
            _truly_running = [cp for cp in _running if cp.status == "running"]
            _paused = [cp for cp in _running if cp.status == "paused"]

            if _truly_running:
                # Hard block: GPU is busy with skill LLM — skip full turn
                _active_cp = _truly_running[0]
                _ack_lines = [
                    "还在研究「{}」，等我一下～",
                    "I'm still working on「{}」— I'll be right with you!",
                ]
                import re as _re
                _is_zh = bool(_re.search(r"[\u4e00-\u9fff]", query_text))
                _ack = (_ack_lines[0] if _is_zh else _ack_lines[1]).format(
                    _active_cp.goal[:40]
                )
                try:
                    await _get_sender().send_message(session.chat_id, _ack)
                except Exception as _exc:
                    logger.warning("[Brain] Running-skill ack send failed: %s", _exc)
                logger.info(
                    "[Brain] Interrupted by user mid-skill run_id=%s — sent ack, skipping turn",
                    _active_cp.run_id,
                )
                return []
            elif _paused:
                # Soft block: skill is paused (e.g. waiting for sensitivity reply).
                # Allow the LLM turn so the user can chat normally, but remember
                # the active skill so the planner won't start another one.
                _active_skill_cp = _paused[0]
                logger.info(
                    "[Brain] Skill run_id=%s is paused — will allow chat turn but block new skill start",
                    _active_skill_cp.run_id,
                )
        except Exception:
            logger.exception("[Brain] Running-skill guard failed — continuing normal turn")

        # ── Topic-shift pre-check ─────────────────────────────────────────────
        # Now that the goal is loaded we can accurately detect topic changes.
        # Keyword + embedding check — zero LLM cost, < 5 ms.
        topic_shifted = await _is_topic_shift(query_text, goal)
        if topic_shifted:
            logger.info(
                "[Memory] Topic shift detected — conversation memory skipped for this turn"
            )

        # Tier 3 — recall from The Knowledge
        knowledge_snippets: list[str] = []
        if session.knowledge:
            try:
                knowledge_snippets = await session.knowledge.recall(
                    query=query_text,
                    chat_id=session.chat_id,
                    top_k=settings.knowledge_recall_top_k,
                    skip_conversations=topic_shifted,
                )
                identity_count = sum(1 for s in knowledge_snippets if s.startswith("[Ella's identity]"))
                logger.info(
                    "[Memory] Qdrant recall query=%r → %d snippet(s) (%d identity, %d conversation)%s",
                    query_text[:60], len(knowledge_snippets),
                    identity_count, len(knowledge_snippets) - identity_count,
                    " [conv skipped — topic shift]" if topic_shifted else "",
                )
            except Exception:
                logger.exception("Knowledge recall failed")

        # Persist the goal (with any updated knowledge notes) to Redis
        if knowledge_snippets:
            goal.shared_notes["knowledge_snippets"] = knowledge_snippets[:3]
        try:
            await goal_store.create(goal)
        except Exception:
            logger.exception("Failed to persist JobGoal")

        session.goal = goal

        # ── Emotion engine — load state before building the focus prompt ─────────
        from ella.config import get_settings as _gs
        _settings = _gs()
        agent_state = None
        personality = None
        if _settings.emotion_enabled:
            try:
                from ella.emotion import engine as _emotion_engine
                from ella.memory.identity import get_personality_traits
                agent_state = await _emotion_engine.read_agent_state(session.chat_id)
                personality = get_personality_traits()
                logger.info(
                    "[Emotion] ── PRE-TURN chat_id=%d ── "
                    "Ella: %s (v=%.2f e=%.2f d=%.2f i=%.2f)",
                    session.chat_id,
                    agent_state.emotion,
                    agent_state.valence, agent_state.energy,
                    agent_state.dominance, agent_state.intensity,
                )
            except Exception:
                logger.exception("Failed to load emotion engine state")

        # ── Phase 1: Summarise recent history + derive topic + new objective ────
        condensed_history = ""
        current_topic = ""
        llm_objective = ""
        if goal.steps_done:
            # Turns 2+ — summarise history and refine the objective progressively
            try:
                condensed_history, current_topic, llm_objective = await summarise_recent_history(
                    goal,
                    window_minutes=15,
                    max_steps=15,
                )
            except Exception:
                logger.exception("History summarisation failed")
        else:
            # Turn 1 — no history yet; derive an initial objective from the
            # opening message so we start with something specific rather than
            # the generic default.
            try:
                llm_objective = await derive_initial_objective(query_text)
            except Exception:
                logger.exception("Initial objective derivation failed")

        # Persist LLM-generated objective every turn.
        # Progressively narrows from generic → specific as the conversation develops.
        if llm_objective:
            try:
                await get_goal_store().update_objective(job_id, llm_objective)
                goal.objective = llm_objective
                logger.info("[Objective] Updated → %s", llm_objective[:120])
            except Exception:
                logger.exception("Failed to persist LLM-generated objective")

        # ── Phase 2: Upfront task planning (tools) + skill detection ────────
        planned_tasks: list[PlannedTask] = []
        planned_skill: PlannedSkill | None = None

        if True:
            try:
                from ella.skills.registry import get_skill_registry
                _skill_registry = get_skill_registry()
                _skill_schema = _skill_registry.get_skills_schema()

                # Check for paused/failed skill executions for this user
                # (paused = waiting for input; failed = crashed mid-run, resumable)
                _paused: list[Any] = []
                try:
                    from ella.skills.execution import get_execution_registry as _get_exec_reg
                    from ella.skills.checkpoint import get_checkpoint_store as _get_cp_store
                    _exec_reg = await _get_exec_reg()
                    _paused = await _exec_reg.list_active(session.chat_id)
                    # Also include recently failed runs so planner can offer to resume
                    if not _paused:
                        _cp_store = await _get_cp_store()
                        _failed = await _cp_store.list_resumable(session.chat_id, max_age_hours=24)
                        _paused = _failed  # planner treats them the same way
                except Exception:
                    pass

                # Knowledge-first check: if Ella already has stored knowledge
                # on the query topic, tell the planner so it won't trigger a
                # redundant learn skill.
                #
                # IMPORTANT — topic relevance guard:
                # The embedding model can return high-similarity results for
                # loosely related topics (e.g. querying "红烧肉" returns 炸酱面
                # chunks because both involve pork + cooking). We must only
                # block re-learning when the stored knowledge is actually about
                # the same topic the user asked about.
                # Guard: at least one significant word from the query must
                # appear in the stored topic string (case-insensitive, CJK chars
                # treated as individual tokens).
                _existing_knowledge: list[str] = []
                if session.knowledge:
                    try:
                        # min_score=0.75 filters out loose semantic neighbours
                        # (e.g. 红烧肉 query matching 炸酱面 chunks at ~0.65).
                        # True on-topic results score 0.75+ with this embedding model.
                        _TOPIC_KNOWLEDGE_MIN_SCORE = 0.75
                        _raw_knowledge = await session.knowledge.recall_topic_knowledge(
                            query=query_text, top_k=5,
                            sensitivity_allow=("public", "internal"),
                            min_score=_TOPIC_KNOWLEDGE_MIN_SCORE,
                        )
                        logger.info(
                            "[Planner] recall_topic_knowledge returned %d result(s) "
                            "(score≥%.2f) for query: %s",
                            len(_raw_knowledge), _TOPIC_KNOWLEDGE_MIN_SCORE, query_text[:80],
                        )

                        _existing_knowledge = [
                            f"[Topic: {r.get('topic','')}] {r.get('chunk_text','')[:300]}"
                            for r in _raw_knowledge if r.get("chunk_text")
                        ]
                        if _existing_knowledge:
                            logger.info(
                                "[Planner] Found %d relevant knowledge chunk(s) — "
                                "injecting into planner to block re-learn",
                                len(_existing_knowledge),
                            )
                        elif _raw_knowledge:
                            logger.info(
                                "[Planner] %d result(s) after score filter but no chunk_text — "
                                "not blocking learn skill",
                                len(_raw_knowledge),
                            )
                    except Exception:
                        logger.warning("[Planner] recall_topic_knowledge failed", exc_info=True)

                planned_tasks, planned_skill = await _plan_tasks(
                    topic=current_topic,
                    condensed_history=condensed_history,
                    user_input=query_text,
                    registry=registry,
                    skill_schema=_skill_schema if _skill_schema else None,
                    paused_executions=_paused if _paused else None,
                    existing_knowledge=_existing_knowledge or None,
                )
            except Exception:
                logger.exception("Task planning failed")

        # ── Build focus prompt with current topic injected ────────────────────
        focus_messages = build_focus_prompt(
            session.focus, goal, knowledge_snippets,
            agent_state=agent_state,
            current_topic=current_topic or None,
        )

        # Inject system brain instruction (includes emotion fields when engine is on)
        focus_messages.insert(1, LLMMessage(
            role="system",
            content=_build_brain_system(_settings.emotion_enabled),
        ))

        # If there are planned tasks or a skill, tell the LLM upfront so the
        # first reply can acknowledge that Ella is looking into things.
        if planned_tasks:
            task_hints = "\n".join(
                f"  • {pt.tool_name}: {pt.reasoning}" for pt in planned_tasks
            )
            focus_messages.append(LLMMessage(
                role="system",
                content=(
                    f"[Planned tasks — running in background]\n{task_hints}\n\n"
                    "You are about to run these tools. "
                    "Write a brief, natural opening response acknowledging what you are doing. "
                    "Do NOT include any results yet — you don't have them. "
                    "Speak naturally, like you're letting a friend know you're on it."
                ),
            ))
        elif planned_skill:
            resume_note = " (resuming from where I left off)" if planned_skill.resume_run_id else ""
            focus_messages.append(LLMMessage(
                role="system",
                content=(
                    f"[Skill planned — '{planned_skill.skill_name}' will run after this reply]\n"
                    f"Goal: {planned_skill.goal}\n\n"
                    f"Tell the user you are ABOUT TO start researching this topic{resume_note}. "
                    "Use future tense — 'I'll look into...', 'Let me research...', '我来好好研究一下...'. "
                    "CRITICAL: Do NOT mention any facts, results, recipes, or findings — you have not done any research yet. "
                    "Do NOT pretend you already know or found something. Keep it to 1-2 sentences of honest intent."
                ),
            ))

        # ── Phase 3a: Generate and send the first reply immediately ──────────
        reply_text, sentences, detail_text, language, emotion_label, user_emotion_raw, tasks, emojis = \
            await self._run_tool_loop(
                focus_messages, registry, settings, goal, job_id,
                skip_tools=bool(planned_tasks),
            )

        # Language safety net
        if _contains_chinese(query_text) and not _contains_chinese(reply_text):
            logger.warning(
                "LLM replied in English despite Chinese input — forcing language=zh"
            )
            language = "zh"

        # ── Emotion engine — apply contagion and self-update ──────────────────
        if _settings.emotion_enabled and personality is not None:
            try:
                from ella.emotion.models import UserState, VALID_EMOTION_LABELS

                ustate: UserState | None = None

                if user_emotion_raw:
                    intensity = float(user_emotion_raw.get("intensity", 0))
                    skipped = intensity <= 0.25
                    logger.info(
                        "[Emotion] User emotion (LLM/TEXT): %-18s  v=%+.2f e=%.2f d=%.2f i=%.2f%s",
                        user_emotion_raw.get("label", "?"),
                        float(user_emotion_raw.get("valence", 0)),
                        float(user_emotion_raw.get("energy", 0)),
                        float(user_emotion_raw.get("dominance", 0)),
                        intensity,
                        "  [below threshold — contagion skipped]" if skipped else "",
                    )
                    if not skipped:
                        ustate = UserState(
                            valence=float(user_emotion_raw.get("valence", 0.0)),
                            energy=float(user_emotion_raw.get("energy", 0.4)),
                            dominance=float(user_emotion_raw.get("dominance", 0.5)),
                            emotion=str(user_emotion_raw.get("label", "calmness")),
                            intensity=intensity,
                        )
                else:
                    logger.info("[Emotion] User emotion (LLM/TEXT): neutral — no signal detected")

                logger.info(
                    "[Emotion] Ella self-assessed: %s",
                    emotion_label or "(none — not emitted)",
                )

                if ustate is not None:
                    agent_state = await _emotion_engine.apply_contagion(
                        session.chat_id, ustate, personality
                    )
                    from ella.emotion.store import get_emotion_store
                    await get_emotion_store().upsert_user_state(session.chat_id, ustate)

                if emotion_label and emotion_label in VALID_EMOTION_LABELS:
                    agent_state = await _emotion_engine.apply_self_update(
                        session.chat_id, emotion_label, trigger="llm_self_assess"
                    )

                if agent_state is not None:
                    logger.info(
                        "[Emotion] ── POST-TURN chat_id=%d ── "
                        "Ella: %s (v=%.2f e=%.2f d=%.2f i=%.2f) | sessions=%d",
                        session.chat_id,
                        agent_state.emotion,
                        agent_state.valence, agent_state.energy,
                        agent_state.dominance, agent_state.intensity,
                        agent_state.session_count,
                    )
            except Exception:
                logger.exception("Emotion engine update failed")

        # Summarise this turn into The Goal as a step.
        # When a skill is about to run, the first reply is just "I'll research X"
        # — recording the LLM's speculative text as fact would poison future turns
        # with hallucinated knowledge. Write a neutral placeholder instead.
        step_index = len(goal.steps_done)
        session.focus.append(LLMMessage(role="assistant", content=reply_text))
        if planned_skill is not None:
            skill_summary = (
                f"[Skill '{planned_skill.skill_name}' queued] "
                f"User asked: {query_text[:80]} — Ella is about to research: {planned_skill.goal[:80]}"
            )
            step_summary = StepSummary(
                step_index=step_index,
                agent="BrainAgent",
                summary=skill_summary,
                raw_user_text=query_text[:600],
                raw_ella_text=skill_summary,
            )
        else:
            step_summary = StepSummary(
                step_index=step_index,
                agent="BrainAgent",
                summary=summarise_focus(session.focus),
                raw_user_text=query_text[:600],
                raw_ella_text=reply_text[:600],
            )
        try:
            await get_goal_store().append_step(job_id, step_summary)
        except Exception:
            logger.exception("Failed to update JobGoal")

        logger.info(
            "[Brain] ── FIRST REPLY READY chat_id=%d lang=%s emotion=%s sentences=%d tasks=%d ──\n"
            "  sentences: %s\n"
            "  detail:    %s",
            session.chat_id, language, emotion_label, len(sentences), len(tasks),
            sentences,
            (detail_text[:120] + "…") if detail_text and len(detail_text) > 120 else detail_text,
        )

        # Build the first reply handoff
        first_reply_payload = ReplyPayload(
            text=reply_text,
            sentences=sentences,
            language=language,
            detail_text=detail_text if not planned_tasks else None,
            emojis=emojis,
            emotion=emotion_label,
        )
        reply_handoff = HandoffMessage(payload=first_reply_payload, session=session)

        # 1. Send first reply immediately — user hears Ella before any tool work.
        try:
            await self._reply_agent.handle(reply_handoff)
        except Exception:
            logger.exception("ReplyAgent (first reply) failed")

        # 2. Fire long-running Celery tasks (coding/document/shell) if the LLM
        #    scheduled any — these are separate from the inline planned_tasks.
        if tasks:
            task_objects = [
                Task(
                    task_id=str(uuid.uuid4()),
                    job_id=job_id,
                    task_type=t.get("type", "other"),
                    description=t.get("description", ""),
                    priority=int(t.get("priority", 1)),
                    chat_id=session.chat_id,
                )
                for t in tasks
            ]
            task_handoff = HandoffMessage(payload=task_objects, session=session)
            try:
                await self._task_agent.handle(task_handoff)
            except Exception:
                logger.exception("TaskAgent failed")

        # ── Phase 3b + 4 + 5: Execute planned tools, update user, final reply ─
        if planned_tasks:
            await self._run_planned_tasks(
                planned_tasks=planned_tasks,
                query_text=query_text,
                current_topic=current_topic,
                first_reply_sentences=sentences,
                language=language,
                focus_messages=focus_messages,
                registry=registry,
                settings=settings,
                goal=goal,
                job_id=job_id,
                session=session,
                emotion_enabled=_settings.emotion_enabled,
            )
        elif planned_skill is not None:
            # If a skill is already paused (e.g. waiting for sensitivity reply),
            # do NOT start another skill on top — just let the chat turn stand.
            if _active_skill_cp is not None:
                logger.info(
                    "[Brain] Blocked new skill '%s' — skill run_id=%s is already paused for this chat",
                    planned_skill.skill_name, _active_skill_cp.run_id,
                )
            else:
                await self._run_planned_skill(
                    planned_skill=planned_skill,
                    session=session,
                    registry=registry,
                    goal=goal,
                    job_id=job_id,
                )

        return []

    async def _run_planned_tasks(
        self,
        planned_tasks: list[PlannedTask],
        query_text: str,
        current_topic: str,
        first_reply_sentences: list[str],
        language: str,
        focus_messages: list[LLMMessage],
        registry: Any,
        settings: Any,
        goal: "JobGoal",
        job_id: str,
        session: Any,
        emotion_enabled: bool,
    ) -> None:
        """Execute planned tools sequentially, sending a contextual update after each,
        then generate and send a final summary reply when all tools are done.
        """
        from ella.communications.telegram.sender import get_sender

        sender = get_sender()
        prior_updates: list[str] = []
        tool_result_notes: list[str] = []
        turn_index = len(goal.steps_done)
        goal_store = get_goal_store()

        try:
            for pt in planned_tasks:
                logger.info(
                    "[Planner] ── EXECUTING tool=%s args=%s", pt.tool_name, pt.args
                )
                try:
                    import time as _time
                    _t0 = _time.monotonic()
                    tool_result = await registry.execute(pt.tool_name, pt.args)
                    _elapsed = _time.monotonic() - _t0
                    tool_result_str = str(tool_result)
                    logger.info(
                        "[Planner] ── RESULT tool=%s %.2fs %d chars | %s",
                        pt.tool_name, _elapsed, len(tool_result_str),
                        tool_result_str[:150].replace("\n", " "),
                    )
                except Exception:
                    logger.exception("Planned tool %s failed", pt.tool_name)
                    tool_result_str = f"Tool {pt.tool_name} failed to execute."

                # Persist ToolFocus
                tf = ToolFocus(
                    turn_index=turn_index,
                    tool_name=pt.tool_name,
                    tool_args=pt.args,
                    tool_result=tool_result_str[:400],
                    reasoning=pt.reasoning,
                )
                try:
                    await goal_store.append_tool_focus(job_id, tf)
                except Exception:
                    logger.exception("Failed to persist ToolFocus for planned task")

                tool_result_notes.append(f"• {pt.tool_name}: {tool_result_str[:400]}")

                # Phase 4: Generate a natural per-tool update and send it
                update_text = await _generate_tool_update(
                    tool_name=pt.tool_name,
                    tool_result=tool_result_str,
                    topic=current_topic,
                    user_input=query_text,
                    first_reply_sentences=first_reply_sentences,
                    prior_updates=prior_updates,
                    language=language,
                )

                if update_text:
                    prior_updates.append(update_text)
                    try:
                        await sender.send_message(
                            chat_id=session.chat_id,
                            text=update_text,
                            parse_mode="HTML",
                        )
                        logger.info(
                            "[Planner] Sent tool update for %s to chat_id=%d",
                            pt.tool_name, session.chat_id,
                        )
                    except Exception:
                        logger.exception("Failed to send tool update message")

            # Phase 5: All tools done — generate and send a final summary reply
            if tool_result_notes:
                already_said_block = ""
                if first_reply_sentences:
                    already_said_block += "Initial response: " + " ".join(first_reply_sentences) + "\n"
                for i, upd in enumerate(prior_updates, 1):
                    already_said_block += f"Update {i}: {upd}\n"

                final_messages = list(focus_messages) + [
                    LLMMessage(
                        role="system",
                        content=(
                            "[All tool results]\n" + "\n".join(tool_result_notes) + "\n\n"
                            "[What you already said this turn]\n" + already_said_block.strip() + "\n\n"
                            "All planned tools have now completed. "
                            "Give a final cohesive reply that synthesises the results. "
                            "Do NOT repeat or rephrase anything in 'What you already said this turn'. "
                            "Add only new information, conclusions, or follow-up thoughts. "
                            "Be natural — speak like a friend wrapping up, not a summary report."
                        ),
                    ),
                ]
                tool_schemas = registry.get_schemas()
                raw_final = await _call_llm(final_messages, tool_schemas)
                final_result = _parse_brain_output(raw_final)
                f_reply, f_sentences, f_detail, f_language, f_emotion, _, f_tasks, f_emojis = final_result

                if _contains_chinese(query_text) and not _contains_chinese(f_reply):
                    f_language = "zh"

                logger.info(
                    "[Planner] ── FINAL REPLY sentences=%d lang=%s ── %s",
                    len(f_sentences), f_language, f_sentences,
                )

                final_payload = ReplyPayload(
                    text=f_reply,
                    sentences=f_sentences,
                    language=f_language,
                    detail_text=f_detail,
                    emojis=f_emojis,
                    emotion=f_emotion,
                )
                final_handoff = HandoffMessage(payload=final_payload, session=session)
                try:
                    await self._reply_agent.handle(final_handoff)
                except Exception:
                    logger.exception("ReplyAgent (final summary) failed")

                # Update goal step with the final reply text appended
                if f_reply:
                    session.focus.append(LLMMessage(role="assistant", content=f_reply))
                    step_index = len(goal.steps_done)
                    step_summary = StepSummary(
                        step_index=step_index,
                        agent="BrainAgent",
                        summary=summarise_focus(session.focus),
                        raw_user_text=query_text[:600],
                        raw_ella_text=f_reply[:600],
                    )
                    try:
                        await goal_store.append_step(job_id, step_summary)
                    except Exception:
                        logger.exception("Failed to update JobGoal after final reply")

        except Exception:
            logger.exception("Unexpected error in _run_planned_tasks")

    async def _run_planned_skill(
        self,
        planned_skill: "PlannedSkill",
        session: Any,
        registry: Any,
        goal: "JobGoal",
        job_id: str,
    ) -> None:
        """Execute a planned skill after the first reply has been sent.

        Builds a SkillContext tied to the current session, then either starts
        a new execution or resumes a paused one.
        """
        from ella.communications.telegram.sender import get_sender
        from ella.skills.execution import get_execution_registry as _get_exec_reg
        from ella.memory.goal import get_goal_store as _get_goal_store

        sender = get_sender()
        chat_id = session.chat_id

        async def _send_update(text: str) -> None:
            try:
                await sender.send_message(chat_id, text)
            except Exception as exc:
                logger.warning("[Skill] send_update failed (%s): %s", exc, text[:60])

        # Mutable container so _ask_user can read the run_id assigned by exec_reg
        # (only known after start() / resume() creates the checkpoint).
        _active_run: dict[str, str] = {"run_id": planned_skill.resume_run_id or ""}

        async def _ask_user(prompt: str) -> str | None:
            """Send a question, write a Redis reply slot, and poll for the answer.

            BrainAgent detects the reply slot on the next incoming turn and
            writes the user's text as the answer before routing back here.
            The skill coroutine sleeps-and-polls until the answer arrives or
            the slot expires (5 min timeout).
            """
            from ella.skills.checkpoint import get_checkpoint_store as _get_cp_store
            import asyncio as _asyncio

            try:
                await sender.send_message(chat_id, prompt)
            except Exception as exc:
                logger.warning("[Skill] ask_user send failed (%s): %s", exc, prompt[:60])

            try:
                cp_store = await _get_cp_store()
                run_id = _active_run["run_id"] or "unknown"
                await cp_store.set_pending_reply(chat_id, run_id, prompt, ttl=120)
                logger.info("[Skill] Waiting for user reply run_id=%s question=%r", run_id, prompt[:60])

                # Poll up to 2 minutes for the answer — long enough for the user
                # to notice and reply, short enough not to block the skill loop.
                deadline = 120  # seconds
                interval = 3
                elapsed = 0
                while elapsed < deadline:
                    await _asyncio.sleep(interval)
                    elapsed += interval
                    slot = await cp_store.get_pending_reply(chat_id)
                    if slot is None:
                        logger.info("[Skill] Reply slot gone (cleared externally) run_id=%s", run_id)
                        return None
                    if slot.get("answer") is not None:
                        answer = slot["answer"]
                        await cp_store.clear_pending_reply(chat_id)
                        logger.info("[Skill] Got user reply run_id=%s answer=%r", run_id, str(answer)[:60])
                        return answer

                # Timed out
                await cp_store.clear_pending_reply(chat_id)
                logger.warning("[Skill] ask_user timed out after %ds run_id=%s", deadline, run_id)
            except Exception as exc:
                logger.warning("[Skill] ask_user polling failed (%s)", exc)
            return None

        try:
            exec_reg = await _get_exec_reg()

            if planned_skill.resume_run_id:
                _active_run["run_id"] = planned_skill.resume_run_id
                logger.info("[Brain] Resuming skill '%s' run_id=%s", planned_skill.skill_name, planned_skill.resume_run_id)
                result = await exec_reg.resume(
                    run_id=planned_skill.resume_run_id,
                    session=session,
                    tool_executor=registry,
                    send_update=_send_update,
                    ask_user=_ask_user,
                )
            else:
                logger.info("[Brain] Starting skill '%s' goal=%r", planned_skill.skill_name, planned_skill.goal[:60])
                result = await exec_reg.start(
                    skill_name=planned_skill.skill_name,
                    goal=planned_skill.goal,
                    session=session,
                    tool_executor=registry,
                    send_update=_send_update,
                    ask_user=_ask_user,
                    on_run_id=lambda rid: _active_run.update({"run_id": rid}),
                )

            # Save skill result summary to the goal's shared notes only —
            # the skill itself already sent "✅ Learning complete!" to the user,
            # so we do NOT send another summary message here to avoid duplicates.
            if result and result.summary:
                try:
                    await _get_goal_store().update_notes(job_id, {
                        f"skill_{planned_skill.skill_name}_summary": result.summary[:500],
                    })
                except Exception:
                    pass

            # Surface unresolved questions only — the skill handles the rest
            if result and result.open_questions:
                q_text = "\n".join(f"• {q}" for q in result.open_questions[:3])
                await _send_update(f"🤔 还有些问题没有完全解决，之后可以继续深入：\n{q_text}")

        except Exception:
            logger.exception("[Brain] Skill execution failed for '%s'", planned_skill.skill_name)
            await _send_update("⚠️ I ran into an issue with that learning session. I'll try again next time.")

    async def _run_tool_loop(
        self,
        messages: list[LLMMessage],
        registry: Any,
        settings: Any,
        goal: JobGoal,
        job_id: str,
        model: Any = None,
        tokenizer: Any = None,
        skip_tools: bool = False,
    ) -> tuple[str, list[str], str | None, str, str | None, dict | None, list[dict], list[dict]]:
        """ReAct loop: LLM → tool call → result → repeat until plain reply.

        Each tool call is processed with its own isolated Focus slice:
          [shared context] + [current user message] + [this tool's input + result]

        The result is stored as a ToolFocus in The Goal so future turns can
        reference what each tool did without carrying full raw results in context.

        When `model` and `tokenizer` are provided (pre-loaded by handle()), they are
        used directly and are NOT unloaded here — the caller owns cleanup.
        When not provided, loads and unloads the model internally.

        When `skip_tools` is True, skips the ReAct loop and goes directly to a
        single LLM pass for the reply (used when _plan_tasks has already decided
        which tools to run separately).

        """
        turn_index = len(goal.steps_done)
        goal_store = get_goal_store()

        try:
            tool_schemas = registry.get_schemas()
            max_rounds = settings.max_tool_rounds
            rounds = 0

            # When planned tasks have already been scheduled separately, skip
            # the ReAct tool loop and go straight to a single-pass reply.
            if skip_tools:
                raw_output = await _call_llm(messages, [])
                result = _parse_brain_output(raw_output)
                if _is_fallback_result(result, raw_output):
                    logger.warning("[Brain] No JSON in first-pass output — retrying with forced JSON prompt")
                    retry_messages = list(messages) + [
                        LLMMessage(role="assistant", content=raw_output.strip()),
                        LLMMessage(
                            role="system",
                            content=(
                                "Your reply above did not include the required JSON output. "
                                "Now output ONLY the JSON — nothing else:\n"
                                '{"sentences":["sentence 1","sentence 2"],'
                                '"emojis":[],"detail":null,"language":"zh","emotion":"calmness","tasks":[]}'
                            ),
                        ),
                    ]
                    raw_output = await _call_llm(retry_messages, [])
                    result = _parse_brain_output(raw_output)
                return result

            # Accumulated tool results to include in the final reply context.
            # Each entry is a compact "Tool X returned: ..." note.
            tool_result_notes: list[str] = []

            while rounds < max_rounds:
                # Build this round's context: shared base + any prior tool notes
                round_messages = list(messages)
                if tool_result_notes:
                    round_messages.append(LLMMessage(
                        role="system",
                        content="[Tool results so far]\n" + "\n".join(tool_result_notes),
                    ))

                # On the first round with no tool results yet, remind the model
                # of the tool-use decision rule — only use a tool if genuinely needed.
                if rounds == 0 and not tool_result_notes:
                    round_messages.append(LLMMessage(
                        role="system",
                        content=(
                            "Think before acting: can you answer well from what you already know? "
                            "If yes, reply directly — no tool needed. "
                            "Only emit a <tool_call> if you genuinely cannot answer without it "
                            "(e.g. user explicitly asked to search, or you need live/current data). "
                            "Use ONLY the argument names in each tool's schema."
                        ),
                    ))

                raw_output = await _call_llm(round_messages, tool_schemas)

                tool_call = _extract_tool_call(raw_output)
                if not tool_call:
                    # No more tool calls — produce the final reply
                    # Give LLM one last pass with all tool notes in context
                    if tool_result_notes:
                        final_messages = list(messages) + [LLMMessage(
                            role="system",
                            content="[Tool results so far]\n" + "\n".join(tool_result_notes),
                        )]
                        raw_output = await _call_llm(final_messages, [])
                    result = _parse_brain_output(raw_output)
                    # If parse fell back to plain text (no JSON found), retry once
                    # with a minimal "give me JSON now" prompt so we don't lose
                    # the model's answer in a wall of thinking prose.
                    if _is_fallback_result(result, raw_output):
                        logger.warning("[Brain] No JSON in LLM output — retrying with forced JSON prompt")
                        retry_messages = list(messages) + [LLMMessage(
                            role="assistant",
                            content=raw_output.strip(),
                        ), LLMMessage(
                            role="system",
                            content=(
                                "Your reply above did not include the required JSON output. "
                                "Now output ONLY the JSON — nothing else:\n"
                                '{"sentences":["sentence 1","sentence 2"],'
                                '"emojis":[],"detail":null,"language":"zh","emotion":"calmness","tasks":[]}'
                            ),
                        )]
                        raw_output = await _call_llm(retry_messages, [])
                        result = _parse_brain_output(raw_output)
                    return result

                # --- Isolated per-tool Focus ---
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                logger.info(
                    "[Tool] ── CALL round=%d ── %s | args=%s",
                    rounds + 1, tool_name, tool_args,
                )

                # Execute tool
                import time as _time
                _tool_t0 = _time.monotonic()
                tool_result = await registry.execute(tool_name, tool_args)
                _tool_elapsed = _time.monotonic() - _tool_t0
                tool_result_str = str(tool_result)

                logger.info(
                    "[Tool] ── RESULT round=%d ── %s | %.2fs | %d chars | preview: %s",
                    rounds + 1, tool_name, _tool_elapsed, len(tool_result_str),
                    tool_result_str[:200].replace("\n", " "),
                )

                # Ask the LLM to reason about just this tool's result in isolation.
                # Use a minimal prompt — NOT the full messages list — so the model
                # cannot bleed internal field names (tool_focuses, steps_done, etc.)
                # back into subsequent tool call arguments.
                tool_focus_messages = [
                    LLMMessage(
                        role="system",
                        content="You are a concise analyst. Answer in one sentence only.",
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            f"Tool called: {tool_name}\n"
                            f"Tool result (truncated):\n{tool_result_str[:800]}\n\n"
                            "In one sentence, what does this result tell us relevant to the user's request?"
                        ),
                    ),
                ]
                reasoning_raw = await _call_llm(tool_focus_messages, [])
                # Strip any JSON wrapper the LLM might emit
                reasoning = re.sub(r"\{[\s\S]*\}", "", reasoning_raw).strip()[:300]
                if not reasoning:
                    reasoning = f"{tool_name} returned {len(tool_result_str)} chars of data."

                logger.info("[Tool] ── REASONING ── %s | %s", tool_name, reasoning)

                # Persist ToolFocus to The Goal
                tf = ToolFocus(
                    turn_index=turn_index,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result_str[:400],
                    reasoning=reasoning,
                )
                try:
                    await goal_store.append_tool_focus(job_id, tf)
                except Exception:
                    logger.exception("Failed to persist ToolFocus")

                # Add compact note for subsequent rounds (not the full raw result)
                tool_result_notes.append(
                    f"• {tool_name}: {reasoning}"
                )
                rounds += 1
                logger.info("[Tool] ── round %d complete, %d tool(s) used so far ──", rounds, rounds)

            # Max rounds hit — produce final reply with everything collected
            logger.warning("[Tool] max rounds (%d) reached — forcing final reply", max_rounds)
            final_messages = list(messages)
            if tool_result_notes:
                final_messages.append(LLMMessage(
                    role="system",
                    content="[Tool results so far]\n" + "\n".join(tool_result_notes),
                ))
            raw_output = await _call_llm(final_messages, [])
            return _parse_brain_output(raw_output)

        except Exception:
            logger.exception("BrainAgent LLM inference failed")
            msg = "Sorry, I encountered an error processing your request."
            return msg, [msg], None, "en", None, None, [], []


async def _call_llm(messages: list[LLMMessage], tool_schemas: list[dict]) -> str:
    from ella.memory.focus import call_llm
    
    formatted_messages = list(messages)

    # Always inject the JSON format reminder as the very last message so it is
    # the most recent instruction the model sees before generating its output.
    _json_reminder = (
        "YOUR FINAL OUTPUT MUST BE ONLY a JSON object — no prose, no markdown before or after it:\n"
        '{"sentences":["first sentence here","second sentence here"],'
        '"emojis":[],"detail":null,"language":"zh","emotion":"calmness","tasks":[]}\n'
        "sentences = 2-5 natural spoken sentences in the user's language. "
        "detail = null unless there is long content. "
        "emotion = your emotional state for this reply (one of the 27 valid emotion labels). "
        "ONLY use these exact keys: sentences, emojis, detail, language, emotion, user_emotion, tasks. "
        "Do NOT use: description, reply, text, answer, response, content, mood."
    )
    if tool_schemas:
        import json
        schema_text = json.dumps(tool_schemas, indent=2)
        formatted_messages.append(LLMMessage(
            role="system",
            content=(
                "Tools are available but optional. Only emit a <tool_call> if you "
                "genuinely cannot answer without it. To call a tool, you MUST output ONLY this exact format:\n"
                "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"val\"}}</tool_call>\n\n"
                f"Available tools:\n{schema_text}\n\n"
                "Use ONLY the argument names in each tool's schema — never pass internal fields as arguments.\n\n"
                + _json_reminder
            ),
        ))
    else:
        formatted_messages.append(LLMMessage(
            role="system",
            content=_json_reminder,
        ))

    output = await call_llm(formatted_messages)
    log_preview = re.sub(r"<think>[\s\S]*?</think>", "<think>…</think>", output, flags=re.IGNORECASE)
    logger.info("[LLM] raw output preview: %s", log_preview[:300])
    return output


def _extract_tool_call(text: str) -> dict | None:
    """Parse a Qwen2.5 / DeepSeek-R1 function-call block from LLM output."""
    # Strip any <think>...</think> reasoning block first so it doesn't
    # interfere with tool-call detection.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    # Also strip untagged reasoning prose that precedes the actual tool call.
    tag_pos = text.find("<tool_call>")
    if tag_pos > 0:
        text = text[tag_pos:]
    # Qwen2.5 emits tool calls as: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: look for a raw JSON object with "name" and "arguments"
    match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,[^{}]*"arguments"\s*:', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _normalise_for_cmp(s: str) -> str:
    """Strip punctuation/whitespace/CJK particles for fuzzy duplicate detection."""
    s = s.lower()
    # Remove CJK particles and common filler that don't carry meaning
    s = re.sub(r'[啊呀哈嘛呢吧诶哦噢嗯啦哇哎]', '', s)
    # Remove all punctuation and whitespace
    s = re.sub(r'[\s\W_]+', '', s, flags=re.UNICODE)
    return s


def _sentences_are_similar(a: str, b: str, threshold: float = 0.60) -> bool:
    """Return True if two sentences share enough content to be considered duplicates.

    Three checks in order:
      1. Exact match after normalisation
      2. Substring containment (one is wholly inside the other)
      3. Character trigram Jaccard ≥ threshold (catches near-exact rephrases)
      4. Shorter-side coverage: if ≥65% of the shorter sentence's trigrams
         appear in the longer one, they convey the same core idea — even when
         the longer sentence adds extra context that dilutes the Jaccard score.
    """
    na, nb = _normalise_for_cmp(a), _normalise_for_cmp(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Substring containment
    if na in nb or nb in na:
        if min(len(na), len(nb)) >= 6:
            return True
    # Character trigram sets
    def trigrams(s: str) -> set[str]:
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}
    tg_a, tg_b = trigrams(na), trigrams(nb)
    if not tg_a or not tg_b:
        return False
    intersection = len(tg_a & tg_b)
    union = len(tg_a | tg_b)
    # Standard Jaccard
    if intersection / union >= threshold:
        return True
    # Shorter-side coverage: how much of the shorter sentence is in the longer one.
    # This catches "A said X and Y" vs "A said Y" — same core, different length.
    shorter = min(len(tg_a), len(tg_b))
    if shorter > 0 and intersection / shorter >= 0.55:
        return True
    return False


def _dedup_sentences(sentences: list[str]) -> list[str]:
    """Remove near-duplicate sentences from a list, keeping first occurrence.

    Also logs each removal so it's visible in debug output.
    """
    kept: list[str] = []
    for s in sentences:
        if any(_sentences_are_similar(s, k) for k in kept):
            logger.debug("[Brain] Dedup removed near-duplicate sentence: %r", s[:60])
        else:
            kept.append(s)
    return kept


def _parse_brain_output(text: str) -> tuple[str, list[str], str | None, str, str | None, dict | None, list[dict], list[dict]]:
    """Extract reply text, sentences, detail, language, emotion, user_emotion, tasks, and emojis.

    Returns (reply, sentences, detail, language, emotion, user_emotion, tasks, emojis).
    - reply:        full reply as a single string (join of sentences, for storage/logging)
    - sentences:    list of individual spoken sentences
    - detail:       follow-up text content, or None
    - language:     'en' or 'zh'
    - emotion:      Ella's self-assessed emotion label (one of 27), or None
    - user_emotion: dict with label/valence/energy/dominance/intensity, or None
    - tasks:        list of task dicts
    - emojis:       list of {"after": N, "emoji": "X"} dicts
    """
    # Strip ALL <think>...</think> blocks (thinking models may emit more than one).
    # Log the first block at DEBUG level so it's inspectable without flooding logs.
    think_blocks = list(re.finditer(r"<think>([\s\S]*?)</think>", text, re.IGNORECASE))
    if think_blocks:
        first_thinking = think_blocks[0].group(1).strip()
        if first_thinking:
            logger.debug("[Think] %d chars of reasoning: %s…", len(first_thinking), first_thinking[:200])
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # Some thinking models (e.g. DeepSeek-R1-Distill) emit reasoning as plain
    # prose BEFORE the JSON object, without <think> tags.  Discard everything
    # that appears before the first "{" so only the JSON is parsed.
    first_brace = text.find("{")
    if first_brace > 0:
        prefix = text[:first_brace].strip()
        if prefix:
            logger.debug("[Think] %d chars of untagged reasoning stripped: %s…", len(prefix), prefix[:200])
        text = text[first_brace:]

    # The LLM sometimes opens the reply object with "[" instead of "{".
    # Normalise that before attempting to parse.
    normalised = re.sub(r'^\s*\[("sentences")', r'{\1', text.strip())
    if normalised != text.strip():
        text = normalised + "}" if not normalised.rstrip().endswith("}") else normalised

    # Try every JSON object in the output, largest first (greedy match so we
    # capture the full outer object, not the smallest nested one).
    for match in sorted(re.finditer(r"\{[\s\S]*\}", text), key=lambda m: -len(m.group(0))):
        try:
            data = json.loads(match.group(0))

            # Accept either new "sentences" schema or legacy "reply" string.
            # Also handles wrong-schema JSON the model sometimes emits (e.g.
            # {"description": "..."} or {"answer": "..."}) by extracting the
            # longest string value found in the object.
            raw_sentences: list[str] = []
            if "sentences" in data and isinstance(data["sentences"], list):
                raw_sentences = [str(s).strip() for s in data["sentences"] if str(s).strip()]
            elif "reply" in data:
                raw_sentences = [str(data["reply"]).strip()]
            else:
                # Hunt for any string value in the object as a last resort
                for key in ("text", "answer", "response", "message", "description", "content"):
                    if key in data and isinstance(data[key], str) and data[key].strip():
                        raw_sentences = [data[key].strip()]
                        logger.warning("[Brain] Wrong JSON schema (key=%r) — extracted as fallback sentence", key)
                        break
                if not raw_sentences:
                    # Try the longest string value in the whole object
                    all_strings = [(k, v) for k, v in data.items() if isinstance(v, str) and v.strip()]
                    if all_strings:
                        best_key, best_val = max(all_strings, key=lambda kv: len(kv[1]))
                        raw_sentences = [best_val.strip()]
                        logger.warning("[Brain] Unknown JSON schema (key=%r) — extracted as fallback sentence", best_key)

            if not raw_sentences:
                continue

            # Sanitise each sentence: strip JSON/emoji fragments and timestamps
            # the LLM may include verbatim from the focus message context.
            clean: list[str] = []
            for s in raw_sentences:
                s = re.sub(r'\s*\[[\s\S]*?\]\s*\{[\s\S]*?\}', '', s).strip()
                s = re.sub(r'\s*\{[\s\S]*?"emoji"[\s\S]*?\}\s*$', '', s).strip()
                # Strip leading ISO/UTC timestamps like [2026-02-23 10:40:03 UTC]
                s = re.sub(r'^\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\]]*\]\s*', '', s).strip()
                # Strip stray angle-bracket characters the model copies from prompt examples
                s = re.sub(r'^[<>]+', '', s).strip()
                s = re.sub(r'[<>]+$', '', s).strip()
                if s:
                    clean.append(s)
            if not clean:
                continue

            sentences = _dedup_sentences(clean)
            reply = " ".join(sentences)  # flat string for storage/logging

            detail = data.get("detail") or None
            if detail:
                detail = str(detail).strip() or None

            # Derive language from actual sentence content — more reliable than the
            # LLM's "language" field which it frequently gets wrong.
            language = "zh" if _contains_chinese(reply) else str(data.get("language", "en"))

            # Extract emotion label — validate against known 27 labels
            from ella.emotion.models import VALID_EMOTION_LABELS
            raw_emotion = str(data.get("emotion", "")).strip().lower()
            emotion = raw_emotion if raw_emotion in VALID_EMOTION_LABELS else None

            # Extract user_emotion dict — accept if it has at least a label
            raw_user_emotion = data.get("user_emotion")
            user_emotion: dict | None = None
            if isinstance(raw_user_emotion, dict) and raw_user_emotion.get("label"):
                user_emotion = {
                    "label":     str(raw_user_emotion.get("label", "")).strip().lower(),
                    "valence":   float(raw_user_emotion.get("valence", 0.0)),
                    "energy":    float(raw_user_emotion.get("energy", 0.4)),
                    "dominance": float(raw_user_emotion.get("dominance", 0.5)),
                    "intensity": float(raw_user_emotion.get("intensity", 0.0)),
                }

            tasks = data.get("tasks", [])
            if not isinstance(tasks, list):
                tasks = []

            raw_emojis = data.get("emojis", [])
            emojis: list[dict] = []
            if isinstance(raw_emojis, list):
                for e in raw_emojis:
                    if isinstance(e, dict) and "emoji" in e:
                        try:
                            emojis.append({
                                "after": int(e.get("after", 999)),
                                "emoji": str(e["emoji"]).strip(),
                            })
                        except (TypeError, ValueError):
                            pass

            logger.info(
                "Parsed brain output: %d sentence(s), lang=%s, emotion=%s, emojis=%d, tasks=%d | %s",
                len(sentences), language, emotion, len(emojis), len(tasks),
                [s[:30] for s in sentences],
            )
            return reply, sentences, detail, language, emotion, user_emotion, tasks, emojis
        except json.JSONDecodeError:
            continue

    # Total parse failure — the model output plain text instead of JSON.
    # Strip markdown formatting and JSON punctuation, then take first 2 sentences.
    language = "zh" if _contains_chinese(text) else "en"
    plain = text.strip()
    # Remove JSON structural characters that may appear if a partial JSON leaked through
    plain = re.sub(r'^\s*\{', '', plain)
    plain = re.sub(r'\}\s*$', '', plain)
    plain = re.sub(r'"[a-z_]+":\s*', '', plain)  # strip "key": prefixes
    plain = re.sub(r'[{}\[\]"]+', ' ', plain)     # strip remaining JSON punctuation
    # Remove markdown headers, bullets, bold markers, numbered lists
    plain = re.sub(r"#{1,6}\s+", "", plain)
    plain = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", plain)
    plain = re.sub(r"^\s*[-*]\s+", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*\d+\.\s+", "", plain, flags=re.MULTILINE)
    # Split on sentence boundaries and take first 2 non-empty sentences
    raw_parts = re.split(r"(?<=[。！？.!?])\s*", plain)
    sentences: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if part and len(part) > 5:
            sentences.append(part)
        if len(sentences) >= 2:
            break
    if not sentences:
        sentences = [plain[:200].strip()]
    fallback = " ".join(sentences)
    logger.warning("[Brain] JSON parse failed — falling back to plain text (%d chars → %d sentences)", len(text), len(sentences))
    return fallback, sentences, None, language, None, None, [], []


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _is_fallback_result(
    result: tuple,
    raw_output: str,
) -> bool:
    """Return True if _parse_brain_output fell back to plain-text extraction.

    We detect this by checking whether the raw output contains no valid JSON
    object with a 'sentences' or 'reply' key — i.e. the parser had to
    synthesise sentences from prose rather than reading them from JSON.
    """
    # Quick check: if the raw output (after stripping think blocks) contains a
    # proper sentences JSON, the parse succeeded and no retry is needed.
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", raw_output, flags=re.IGNORECASE).strip()
    first_brace = stripped.find("{")
    if first_brace >= 0:
        candidate = stripped[first_brace:]
        for m in sorted(re.finditer(r"\{[\s\S]*\}", candidate), key=lambda x: -len(x.group(0))):
            try:
                data = json.loads(m.group(0))
                if "sentences" in data or "reply" in data:
                    return False  # valid parse — no retry needed
            except json.JSONDecodeError:
                pass
    return True  # no valid JSON found — retry warranted


# Explicit topic-change phrases in Chinese and English.
# Any of these in the user's message is an unambiguous signal to drop old context.
_TOPIC_CHANGE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Chinese
        r"说点别的", r"换个话题", r"换[一个]?话题", r"聊点别的", r"换[个点]别的",
        r"不想[再说|聊]", r"别再说.*了", r"换换", r"聊别的", r"说别的",
        r"换个[方向|方面|主题]", r"改变.*话题", r"说点其他",
        # English
        r"\bchange (the )?topic\b", r"\btalk about something else\b",
        r"\blet'?s? (talk about|discuss) something (else|different)\b",
        r"\bswitch (to something|topics?)\b", r"\bmove on\b",
        r"\bforget (about )?that\b", r"\bdrop (it|this|that)\b",
        r"\bsomething else\b", r"\bdifferent subject\b",
    ]
]


def _detect_topic_shift(
    query_text: str,
    goal: "JobGoal",
) -> tuple[bool, str]:
    """Detect topic shift using keyword matching + embedding similarity.

    Two-stage, zero LLM cost:
      1. Keyword check — explicit "change topic" phrases → instant True
      2. Embedding cosine similarity between the new message and the current
         goal objective.  Low similarity (< 0.25) on a non-trivial message
         strongly suggests the topic has changed.

    Returns (shifted, new_objective_hint).
    new_objective_hint is a short label derived from the message itself
    (used to update the Redis goal objective when shifted=True).
    """
    if not goal.steps_done:
        return False, ""

    msg = query_text.strip()
    if not msg:
        return False, ""

    # Stage 1 — explicit phrase match (instant, highest confidence)
    for pattern in _TOPIC_CHANGE_PATTERNS:
        if pattern.search(msg):
            logger.info("[TopicShift] Explicit phrase matched: %r", msg[:60])
            return True, f"New topic after: {msg[:80]}"

    # Stage 2 — embedding cosine similarity against goal objective
    # Only run if we have a non-trivial prior objective (not the generic default)
    prior = goal.objective.strip()
    generic_prefixes = ("social conversation", "Social conversation")
    if any(prior.startswith(p) for p in generic_prefixes):
        # Social chat has no specific topic — skip similarity check
        return False, ""

    try:
        from ella.memory.embedder import embed
        import numpy as np

        v_msg = np.array(embed(msg), dtype=float)
        v_obj = np.array(embed(prior[:200]), dtype=float)
        norm_m = np.linalg.norm(v_msg)
        norm_o = np.linalg.norm(v_obj)
        if norm_m > 0 and norm_o > 0:
            similarity = float(np.dot(v_msg, v_obj) / (norm_m * norm_o))
            logger.debug("[TopicShift] Embedding similarity=%.3f (threshold=0.25)", similarity)
            if similarity < 0.25 and len(msg) > 6:
                logger.info(
                    "[TopicShift] Low similarity %.3f — topic shift inferred from: %r",
                    similarity, msg[:60],
                )
                return True, f"New topic: {msg[:80]}"
    except Exception:
        logger.debug("[TopicShift] Embedding similarity check failed — skipping")

    return False, ""


async def _is_topic_shift(
    query_text: str,
    goal: "JobGoal | None",
) -> bool:
    """Pre-recall check: returns True when the user has clearly changed topic.

    Caches the result in goal.shared_notes so _maybe_update_objective can
    reuse it without re-running the detection.
    Runs synchronously (no LLM, no I/O) so it adds < 5 ms overhead.
    """
    if goal is None or not goal.steps_done:
        return False
    try:
        shifted, new_obj = _detect_topic_shift(query_text, goal)
        goal.shared_notes["_topic_shift"] = {"shifted": shifted, "new_objective": new_obj}
        return shifted
    except Exception:
        return False


async def _maybe_update_objective(
    query_text: str,
    goal: "JobGoal",
    job_id: str,
) -> None:
    """Update the goal objective if a topic shift was detected.

    Reads the cached result written by _is_topic_shift so no second
    detection pass is needed.
    """
    cached = goal.shared_notes.pop("_topic_shift", None)
    if cached is None:
        # Cache miss (shouldn't normally happen) — run detection now
        shifted, new_obj = _detect_topic_shift(query_text, goal)
    else:
        shifted = cached.get("shifted", False)
        new_obj = cached.get("new_objective", "")

    if shifted and new_obj:
        prior = goal.objective[:80]
        goal_store = get_goal_store()
        await goal_store.update_objective(job_id, new_obj)
        # Clear step history so the old conversation doesn't bleed into new topic
        g = await goal_store.read(job_id)
        if g is not None:
            g.steps_done = []
            g.tool_focuses = []
            await goal_store.create(g)
        logger.info(
            "Goal updated (topic shift): %s → %s | history cleared",
            prior, new_obj[:80],
        )
    else:
        logger.debug("No topic shift detected for job %s", job_id)
