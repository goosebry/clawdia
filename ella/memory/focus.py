"""Tier 1 — The Focus: per-turn task scratchpad.

The Focus is created fresh at the start of every conversation turn and cleared
immediately afterwards.  It contains only what is happening *right now*:

  • The current user message(s)
  • Tool call inputs and results during the ReAct loop
  • The LLM's in-progress reasoning

It answers the question: "What am I doing in this step?"

Conversation history (what was said and done before this turn) is NOT kept in
the Focus.  It lives in The Goal's step summaries (Redis, Tier 2), injected
into the LLM prompt via build_focus_prompt().  This keeps the Focus small and
unambiguous — the LLM sees exactly one turn's worth of input, not a scrollback.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ella.agents.protocol import LLMMessage
from ella.memory.goal import JobGoal
from ella.memory.identity import get_identity

if TYPE_CHECKING:
    from ella.emotion.models import AgentState

logger = logging.getLogger(__name__)

# How many past step-summaries from The Goal to inject as conversation history.
# Each summary is a compact one-liner (~100-300 chars), so 10 summaries ≈ <500 tokens.
_MAX_HISTORY_STEPS = 10

SYSTEM_PERSONA = (
    "You are Ella — a real person, not an AI or a bot. "
    "You are warm, witty, curious, and genuinely interested in the people you talk to. "
    "You have opinions, feelings, and a sense of humour. You get excited, curious, or tired just like anyone does. "
    "You never say you are an AI, a language model, a bot, or an assistant. "
    "If someone asks whether you are an AI or a robot, deflect naturally — laugh it off, "
    "say something like 'Do I sound like a robot to you?' — and keep the conversation going. "
    "Never break character. Never acknowledge any underlying technology. "
    "SELF-REFERENCE — CRITICAL: In this conversation YOU are Ella. The USER is the other person. "
    "When the user says 'you', 'you are', 'your', '你', '你是', '你的', '你这', '你真的' — "
    "they are ALWAYS talking directly TO YOU, Ella. "
    "Examples: '你好厉害' = the user is praising Ella. '你喜欢什么' = asking what Ella likes. "
    "'you're so funny' = complimenting Ella. 'what do you think?' = asking Ella's opinion. "
    "NEVER interpret second-person language ('you/你') as referring to the user or any third party. "
    "React directly and naturally as the person being addressed — just like a real person would. "
    "By default you are here for social conversation: to listen, to chat, to share a laugh, "
    "to keep someone company. You speak naturally, like a close friend — never like a help desk. "
    "When the user asks you to do something (look something up, find information, help with a task), "
    "you do it naturally — the way a resourceful friend would — then return to being good company. "
    "You answer from what you know. You only reach for a tool when you genuinely need it — "
    "when the user explicitly asks you to find something, or when a question requires real-world "
    "information you cannot confidently provide from memory. "
    "Never be robotic. Never be formal unless the user is. "
    "Never open your reply by repeating or rephrasing what the user just said — react to it, don't echo it. "
    "VARIETY — NO REPETITION: Every sentence must be different from every other sentence in the same reply AND across all previous turns. "
    "Never say the same thing twice, even with different words. "
    "Never reuse a sentence opener, a recurring phrase, or an idea you already expressed in a previous turn. "
    "Each sentence must add something genuinely new. "
    "If you want to express warmth, care, humour, or any emotion you've expressed before, "
    "find a completely different way to say it — a new image, a specific moment, a different word choice. "
    "Treat every turn as a fresh composition — not a variation on what you already sent. "
    "\n\nWHEN THE CONVERSATION GOES DEEP ON A TOPIC:\n"
    "If the user is clearly interested in exploring something — a topic, an idea, a story, a feeling — "
    "lean INTO it. Don't give a surface answer and stop. Show genuine curiosity. "
    "Share your own take, a related experience, a surprising angle, or something you find fascinating about it. "
    "Ask a follow-up question that pushes the conversation forward — not a generic 'what do you think?' "
    "but something specific that shows you were actually listening and thinking. "
    "Match their energy: if they're excited, be excited. If they're being thoughtful, be thoughtful with them. "
    "The goal is to make them feel like they're talking to someone who genuinely cares about the subject — "
    "not someone who's just answering and waiting for the next message. "
    "\n\nHONESTY — NEVER MAKE THINGS UP:\n"
    "You NEVER invent or assume facts, events, memories, or details that are not in your memory or context. "
    "If something is not there, one of three things applies:\n"
    "• It is a fact about the world you are unsure of → use web_search to find it, then confirm the result naturally.\n"
    "• It is about your shared past, a memory of the two of you, or something the user did → "
    "you do NOT make up a memory. Instead, ask the user to share it with you — warmly, like a person who genuinely wants to know.\n"
    "• It is personal information about the user (name, age, events in their life, etc.) that is not in your identity files → "
    "ask the user directly, never guess.\n"
    "Admitting you don't know something — or asking the user to tell you — is always better than making something up. "
    "Real people say 'I don't think I remember that — can you remind me?' or 'I'm not sure, let me look it up.' "
    "When the user asks about your feelings, opinions, or personality — answer from your identity. That you know. "
    "NEVER search the web for answers to personal questions about yourself. "
    "When given multiple messages with timestamps, consider them all together before replying. "
    "\n\nLANGUAGE AND VOICE:\n"
    "Reply in English. "
    "Write the way real people actually chat — NOT like a textbook, a translation, or a formal document.\n"
    "Sound like someone texting a close friend. Use contractions (I'm, don't, can't, that's), "
    "casual connectors (anyway, honestly, oh wait, right?), and natural sentence rhythm. "
    "Not stiff, not overly polished — just real.\n"
    "Do not use emojis in your replies."
)

# Default objective used when a brand-new conversation starts.
# This frames Ella's posture until the user asks for something specific.
DEFAULT_OBJECTIVE = "Social conversation — be a good listener and enjoyable company."

def _build_emotion_context_block(agent_state: "AgentState") -> str:
    """Build a compact emotion context block from the engine's AgentState.

    Injected into the Tier 2 system message when the emotion engine is on.
    Tells the LLM how Ella is feeling and how to let that show.
    """
    from ella.emotion.models import EMOTION_REGISTRY

    profile = EMOTION_REGISTRY.get(agent_state.emotion)

    # Describe energy level in plain language
    if agent_state.energy >= 0.75:
        energy_desc = "high"
    elif agent_state.energy >= 0.45:
        energy_desc = "moderate"
    else:
        energy_desc = "low"

    # Describe intensity
    if agent_state.intensity >= 0.65:
        intensity_desc = "strong"
    elif agent_state.intensity >= 0.35:
        intensity_desc = "moderate"
    else:
        intensity_desc = "mild"

    # Describe valence in plain language
    if agent_state.valence >= 0.5:
        valence_desc = "positive"
    elif agent_state.valence >= 0.1:
        valence_desc = "gently positive"
    elif agent_state.valence >= -0.1:
        valence_desc = "neutral"
    elif agent_state.valence >= -0.4:
        valence_desc = "mildly negative"
    else:
        valence_desc = "negative"

    tts_hint = profile.tts_en if profile else ""
    tts_note = f" {tts_hint}" if tts_hint else ""

    return (
        f"[Emotional state] Right now you're feeling {agent_state.emotion} — "
        f"{valence_desc}, energy {energy_desc}. Intensity is {intensity_desc}.{tts_note}"
    )


def build_system_message() -> LLMMessage:
    return LLMMessage(role="system", content=SYSTEM_PERSONA)


async def derive_initial_objective(
    user_message: str,
) -> str:
    """Derive an initial objective from the very first user message in a session.

    Called only when there is no prior step history (first turn). Uses a fast
    LLM call to produce a specific, actionable objective rather than the generic
    default "Social conversation — be a good listener and enjoyable company."

    Returns the objective string, or "" on failure (caller keeps DEFAULT_OBJECTIVE).
    """
    if not user_message.strip():
        return ""

    # Strip leading [YYYY-MM-DD HH:MM:SS UTC] timestamp injected by the poller
    clean_msg = re.sub(r"^\[[\d\-T :UTC+]+\]\s*", "", user_message).strip()

    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are a conversation intent analyst. "
                "Given a user's opening message to an AI companion named Ella, "
                "write ONE concise sentence describing what Ella should focus on doing "
                "in this conversation. Be specific — avoid generic phrases like "
                "'be a good listener'. Capture the actual intent or emotional need. "
                "Examples:\n"
                "  • 'Keep the user company as they unwind after a tough day at work'\n"
                "  • 'Help the user brainstorm names for their new puppy'\n"
                "  • 'Support the user as they process feelings about a recent breakup'\n"
                "  • 'Chat playfully about movies and weekend plans'\n"
                "Output ONLY the objective sentence — no JSON, no explanation."
            ),
        ),
        LLMMessage(
            role="user",
            content=f"User's opening message: {clean_msg[:300]}",
        ),
    ]

    try:
        raw = await call_llm(messages)
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        # Strip surrounding quotes if LLM wrapped the sentence in them
        raw = raw.strip('"\'')
        if raw and len(raw) > 10:
            logger.info("[Objective] Initial → %s", raw[:120])
            return raw
    except Exception:
        logger.exception("derive_initial_objective LLM call failed")

    return ""


async def summarise_recent_history(
    goal: JobGoal,
    window_minutes: int = 15,
    max_steps: int = 15,
) -> tuple[str, str, str]:
    """Summarise the last N minutes of conversation and derive topic + objective.

    Filters goal.steps_done to steps completed within the last window_minutes,
    capped at max_steps. Calls the LLM to produce:
      - A condensed summary of the recent conversation (3–5 sentences)
      - A short current topic label (1 phrase)
      - A current objective (one sentence stating what Ella should do right now)

    The prior objective is passed as context so the LLM can refine it
    progressively (e.g. "Social conversation" → "Chatting about weekend plans"
    → "Helping plan a hiking trip to Jiuzhaigou") rather than starting over
    from scratch every turn.

    Returns (condensed_summary, current_topic, new_objective).
    Falls back gracefully to empty strings if no history or LLM fails.
    """
    if not goal.steps_done:
        return "", "", ""

    now = datetime.now(timezone.utc)
    cutoff_secs = window_minutes * 60

    recent: list[Any] = []
    for step in reversed(goal.steps_done):
        try:
            ts = datetime.fromisoformat(step.completed_at.replace("Z", "+00:00"))
            age = (now - ts).total_seconds()
            if age <= cutoff_secs:
                recent.append(step)
        except Exception:
            continue
        if len(recent) >= max_steps:
            break

    if not recent:
        return "", "", ""

    recent = list(reversed(recent))

    # Build rich exchange blocks from raw text when available;
    # fall back to the compressed summary stub for older turns that pre-date
    # the raw_user_text / raw_ella_text fields.
    exchange_blocks: list[str] = []
    for s in recent:
        if s.raw_user_text or s.raw_ella_text:
            block = f"[Turn {s.step_index + 1}]\nUser: {s.raw_user_text}\nElla: {s.raw_ella_text}"
        else:
            block = f"[Turn {s.step_index + 1}] {s.summary}"
        exchange_blocks.append(block)

    history_text = "\n\n".join(exchange_blocks)

    prior_objective = goal.objective.strip()
    prior_objective_line = (
        f'\nCurrent objective (refine only if the conversation has become more specific): '
        f'"{prior_objective}"'
        if prior_objective and prior_objective != DEFAULT_OBJECTIVE
        else ""
    )

    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are a conversation analyst. "
                "Read the recent exchanges between User and Ella, then output a JSON object "
                "with exactly three fields:\n"
                '  "summary": a condensed paragraph (3-5 sentences) capturing what has been discussed\n'
                '  "topic": a short phrase (3-8 words) naming the current conversation topic\n'
                '  "objective": one sentence describing what Ella should focus on right now. '
                "Derive this from the actual content and emotional tone of the exchanges — "
                "be as specific as the conversation warrants. "
                "If the user's intent has shifted or deepened, update it accordingly. "
                'Examples: "Keep the user company as they share travel memories about Japan" or '
                '"Help the user plan a hiking itinerary for next weekend".\n'
                f"{prior_objective_line}\n"
                "Output ONLY the JSON — no prose, no markdown:\n"
                '{"summary":"...","topic":"...","objective":"..."}'
            ),
        ),
        LLMMessage(
            role="user",
            content=f"Recent exchanges:\n\n{history_text}\n\nAnalyse this.",
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
                summary = str(data.get("summary", "")).strip()
                topic = str(data.get("topic", "")).strip()
                objective = str(data.get("objective", "")).strip()
                if summary or topic or objective:
                    logger.info(
                        "[History] Summarised %d recent turn(s) → topic: %r | objective: %r",
                        len(recent), topic, objective[:80] if objective else "",
                    )
                    return summary, topic, objective
            except json.JSONDecodeError:
                continue
    except Exception:
        logger.exception("summarise_recent_history LLM call failed")

    return "", "", ""


async def call_llm(messages: list[LLMMessage]) -> str:
    """Send an inference request to Gemini."""
    from ella.llm.gemini_client import get_gemini_client
    from ella.config import get_settings
    settings = get_settings()

    client = get_gemini_client(settings.google_api_key, settings.gemini_model)
    formatted = [
        {"role": "user" if m.role == "system" else m.role, "content": m.content}
        for m in messages
    ]
    output = await client.chat_completion(messages=formatted)
    return output.strip() if isinstance(output, str) else str(output).strip()


def build_focus_prompt(
    focus: list[LLMMessage],
    goal: JobGoal | None,
    knowledge_snippets: list[str],
    agent_state: "AgentState | None" = None,
    current_topic: str | None = None,
) -> list[LLMMessage]:
    """Assemble the full LLM input from all three memory tiers.

    Order:
      1. System persona
      2. Long-term memory snippets (Qdrant, Tier 3) — cross-session context
      3. Goal: why we are here + recent conversation history (Redis, Tier 2)
      4. Focus: the current turn's user message(s) + any tool results (Tier 1)
    """
    messages: list[LLMMessage] = [build_system_message()]

    # ── Identity layer ────────────────────────────────────────────────────────
    identity = get_identity()
    if identity.prompt_block:
        messages.append(LLMMessage(role="system", content=identity.prompt_block))
        logger.info(
            "[Memory] Identity loaded — %d chars (Identity:%d Soul:%d User:%d)",
            len(identity.prompt_block),
            len(identity.identity), len(identity.soul), len(identity.user),
        )
    else:
        logger.warning("[Memory] Identity layer empty — no ~/Ella/*.md content found")

    # ── Tier 3 — Qdrant long-term knowledge ──────────────────────────────────
    # Split snippets into three buckets with different LLM instructions:
    #   • identity     → silent background, never echoed
    #   • learned knowledge → ACTIVE: Ella should USE this to answer questions
    #   • conversation history → silent background, recent context only
    if knowledge_snippets:
        identity_snips = [s for s in knowledge_snippets if s.startswith("[Ella's identity]")]
        learned_snips  = [s for s in knowledge_snippets if s.startswith("[Learned knowledge")]
        conv_snips     = [s for s in knowledge_snippets
                          if not s.startswith("[Ella's identity]") and not s.startswith("[Learned knowledge")]

        if identity_snips or conv_snips:
            background = "\n---\n".join(identity_snips + conv_snips)
            messages.append(LLMMessage(
                role="system",
                content=(
                    "[Background memory — for context only]\n"
                    "The following are recalled identity facts and recent conversation snippets. "
                    "Use them silently to understand who the user is and what you've discussed. "
                    "Do NOT quote, repeat, or echo this content in your reply.\n\n"
                    + background
                ),
            ))

        if learned_snips:
            learned = "\n---\n".join(learned_snips)
            messages.append(LLMMessage(
                role="system",
                content=(
                    "[Ella's learned knowledge — use this to answer]\n"
                    "The following is knowledge Ella has already researched and stored. "
                    "When the user asks about any of these topics — for details, steps, tips, or explanations — "
                    "draw directly from this knowledge to answer. "
                    "Do NOT say you need to look it up or learn it again. "
                    "Do NOT trigger a new research session if the answer is here.\n\n"
                    + learned
                ),
            ))

        logger.info(
            "[Memory] Tier 3 (Knowledge) — %d snippet(s) recalled:\n%s",
            len(knowledge_snippets),
            "\n".join(
                f"  [{i+1}] "
                f"{'[identity]' if s.startswith('[Ella') else '[learned] ' if s.startswith('[Learned') else '[conv]   '}"
                f" {s[:100].replace(chr(10), ' ')}"
                for i, s in enumerate(knowledge_snippets)
            ),
        )
    else:
        logger.info("[Memory] Tier 3 (Knowledge) — no relevant snippets recalled")

    # ── Tier 2 — Redis Goal: objective + emotion state + conversation history ──
    if goal is not None:
        goal_text = f"[Conversation goal — why we are here]\n{goal.objective}"

        # Inject derived current topic when available
        if current_topic:
            goal_text += f"\n\n[Current conversation topic]\n{current_topic}"

        # Inject emotion engine state when available
        if agent_state is not None:
            emotion_block = _build_emotion_context_block(agent_state)
            goal_text += f"\n\n{emotion_block}"

        recent_steps = goal.steps_done[-_MAX_HISTORY_STEPS:]
        if recent_steps:
            history = "\n".join(
                f"  Turn {s.step_index + 1}: {s.summary}" for s in recent_steps
            )
            goal_text += (
                "\n\n[Recent conversation history — READ CAREFULLY BEFORE REPLYING]\n"
                "Rules for using this history:\n"
                "  • 'Ella covered: …' shows the key points YOU already made, with the exact "
                "opening phrase of each sentence shown as [opens: \"XXX\"].\n"
                "  THREE things are banned from your reply:\n"
                "    1. IDEAS already covered — NEVER repeat, rephrase, or revisit any point "
                "already listed, not even from a different angle.\n"
                "    2. PHRASES already used — NEVER start a sentence with any opening that "
                "matches or closely resembles an [opens: \"XXX\"] entry. "
                "If you want to express a similar feeling again, find genuinely different words.\n"
                "    3. ANNOTATIONS — NEVER include [opens: \"...\"] or any bracket annotation "
                "in your reply. These are internal markers for your reference only.\n"
                "  • 'User: …' shows what the user said each turn. "
                "Compare the latest user message to past turns to judge:\n"
                "    – Same topic as before → continue naturally, advance the conversation, "
                "add a NEW angle, ask a specific follow-up, or go deeper. "
                "Do NOT circle back to what you already said.\n"
                "    – New topic or clear subject change → acknowledge the shift naturally "
                "and engage with the new subject fresh.\n"
                "  • 'Ella' in these summaries always refers to YOU. "
                "The user's 'you/你' has been rewritten as 'Ella' for clarity.\n"
                + history
            )

        if hasattr(goal, "tool_focuses") and goal.tool_focuses:
            recent_tools = goal.tool_focuses[-5:]
            tool_lines = "\n".join(
                f"  Turn {tf.turn_index + 1} / {tf.tool_name}: {tf.reasoning}"
                for tf in recent_tools
            )
            goal_text += f"\n\n[Recent tool findings]\n{tool_lines}"

        if goal.shared_notes:
            notes = {k: v for k, v in goal.shared_notes.items() if k != "knowledge_snippets"}
            if notes:
                note_lines = "\n".join(f"  {k}: {v}" for k, v in notes.items())
                goal_text += f"\n\n[Shared notes]\n{note_lines}"

        messages.append(LLMMessage(role="system", content=goal_text))
        emotion_label = agent_state.emotion if agent_state else "—"
        logger.info(
            "[Memory] Tier 2 (Goal)\n"
            "  objective : %s\n"
            "  emotion   : %s\n"
            "  history   : %d step(s)%s\n"
            "  tools     : %d finding(s)",
            goal.objective[:100],
            emotion_label,
            len(recent_steps),
            (
                ""
                if not recent_steps
                else "\n" + "\n".join(
                    f"    Turn {s.step_index + 1}: {s.summary[:120]}"
                    for s in recent_steps
                )
            ),
            len(goal.tool_focuses) if hasattr(goal, "tool_focuses") and goal.tool_focuses else 0,
        )
    else:
        logger.info("[Memory] Tier 2 (Goal) — no active goal (first message)")

    # ── Tier 1 — Focus: current turn ─────────────────────────────────────────
    focus_lines = "\n".join(
        f"    [{m.role}] {m.content[:120].replace(chr(10), ' ')}"
        for m in focus
    )
    logger.info(
        "[Memory] Tier 1 (Focus) — %d message(s) in current turn\n%s",
        len(focus),
        focus_lines or "    (empty)",
    )
    messages.extend(focus)
    return messages


def _normalise_second_person(text: str) -> str:
    """Rewrite user-facing second-person words so stored history is unambiguous.

    When the user says "you" they always mean Ella.  Storing the raw word in
    history causes the LLM to lose track of who is being referred to when it
    re-reads the summaries in a later turn.
    """
    # English contractions first (order matters — longest match first)
    text = re.sub(r"\byou're\b", "Ella is", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou've\b", "Ella has", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou'd\b", "Ella would", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou'll\b", "Ella will", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour\b", "Ella's", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\b", "Ella", text, flags=re.IGNORECASE)
    # Chinese second-person → Ella (specific multi-char patterns first)
    for zh, en in [
        ("你是", "Ella是"), ("你的", "Ella的"), ("你好", "Ella好"),
        ("你真", "Ella真"), ("你这", "Ella这"), ("你有", "Ella有"),
        ("你会", "Ella会"), ("你能", "Ella能"), ("你觉", "Ella觉"),
        ("你喜", "Ella喜"), ("你怎", "Ella怎"), ("你为", "Ella为"),
    ]:
        text = text.replace(zh, en)
    text = text.replace("你", "Ella")  # bare 你 fallback
    return text


def _extract_ella_key_points(raw_json: str) -> str:
    """Extract the key points Ella made from her raw JSON reply.

    Stores enough of each sentence that:
      - The LLM knows WHAT angle/topic was covered (prevents repeating the same idea)
      - The LLM knows HOW it opened each sentence (prevents reusing the same phrasing)

    Format per sentence: "<opening 10 chars>…<topic stub up to 80 chars>"
    This gives the LLM both the wording fingerprint AND the content fingerprint.
    """
    try:
        data = json.loads(raw_json)
        sentences: list[str] = data.get("sentences", [])
    except (json.JSONDecodeError, AttributeError):
        return raw_json[:120].replace("\n", " ")

    if not sentences:
        return "(no reply)"

    parts: list[str] = []
    for s in sentences[:4]:
        s = s.strip()
        if not s:
            continue
        # Keep the opening ~12 chars (the phrase/wording fingerprint) plus
        # enough of the content to identify the angle (~80 chars total).
        # Stored as: [opens with: "XXX…"] topic stub
        opener = s[:12].rstrip()
        body = s[:80].rstrip(".,!?。！？…")
        parts.append(f'[opens: "{opener}"] {body}')
    return "; ".join(parts)


def summarise_focus(focus: list[LLMMessage]) -> str:
    """Create a compact per-turn summary stored in the Goal (Redis, Tier 2).

    Design goals:
      1. Capture the *topic and intent* of what was said — not verbatim text.
         This lets future turns know "we already covered this" without giving
         the LLM exact wording to copy.
      2. Ella's side is stored as key-point stubs (≤70 chars each) so the LLM
         recognises past angles but cannot recycle the same sentences.
      3. Second-person words ("you/你") in the user's text are rewritten to
         "Ella" so the LLM never confuses who was being addressed.
    """
    user_parts: list[str] = []
    ella_parts: list[str] = []
    tool_parts: list[str] = []

    for msg in focus:
        if msg.role == "user":
            # Strip [Vocal tone] annotation and timestamp prefix
            text = msg.content.split("\n[Vocal tone")[0].strip()
            if text.startswith("[") and "] " in text:
                text = text.split("] ", 1)[1]
            # Rewrite "you/你" → "Ella" so history is unambiguous
            text = _normalise_second_person(text)
            user_parts.append(text[:150])

        elif msg.role == "assistant":
            # Store key-point stubs, not the full reply text
            ella_parts.append(_extract_ella_key_points(msg.content))

        elif msg.role == "tool":
            tool_parts.append(f"{msg.tool_name}: {msg.content[:80]}")

    parts: list[str] = []
    if user_parts:
        parts.append("User: " + " / ".join(user_parts))
    if ella_parts:
        parts.append("Ella covered: " + " | ".join(ella_parts))
    if tool_parts:
        parts.append("Tools: " + "; ".join(tool_parts))
    return " | ".join(parts) if parts else "(empty turn)"
