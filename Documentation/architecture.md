# Ella AI Agent — Architecture

## Overview

Ella is a Python-based multi-agent AI assistant that communicates through Telegram. It processes text, voice, and video messages, responds with AI-generated audio replies, and schedules background tasks routed to specialised executors. All LLM inference runs in-process on Apple Silicon via the MLX framework — no model server required.

---

## Agent Topology (AutoGen Handoff Pattern)

Agents are self-directing. Each receives a typed message, does its work, and hands off to the next agent by calling its `handle()` method directly. There is no central orchestrator.

```
Telegram (getUpdates long-poll)
        │
        ▼
  TelegramPoller
        │ UserTask(raw_updates, session)
        ▼
  IngestionAgent ──────── text/voice/video handlers
        │ HandoffMessage(list[MessageUnit], session)
        ▼
  BrainAgent ◄──────────► ToolRegistry (get_schemas / execute)
        │                 ▲ watchfiles hot-reload
        │                 └─ ella/tools/builtin/*.py
        │                    ella/tools/custom/*.py
        ├─ HandoffMessage(ReplyPayload, session)
        │         ▼
        │    ReplyAgent ──► Qwen3-TTS ──► sendVoice ──► Telegram
        │                └──► Qdrant (store exchange)
        │
        └─ HandoffMessage(list[Task], session)
                  ▼
             TaskAgent ──► Celery (Redis broker) ──► Worker
                          monitor progress ──► sendMessage ──► Telegram
```

### Agent Descriptions

| Agent | File | Role |
|---|---|---|
| `TelegramPoller` | `ella/communications/telegram/poller.py` | Long-polls Telegram, groups updates by chat, creates `SessionContext` |
| `IngestionAgent` | `ella/agents/ingestion_agent.py` | Converts raw updates → ordered `list[MessageUnit]` |
| `BrainAgent` | `ella/agents/brain_agent.py` | Memory recall, tool-call loop, generates reply + tasks |
| `ReplyAgent` | `ella/agents/reply_agent.py` | TTS → sendVoice (OGG/Opus), sends detail text in split chunks, stores exchange to Qdrant |
| `TaskAgent` | `ella/agents/task_agent.py` | Enqueues tasks to Celery, monitors and reports progress |
| `CeleryWorker` | `ella/tasks/worker.py` | LLM routing → Cursor CLI or OpenAI Codex |

---

## Handoff Message Protocol

All data flows between agents via typed dataclasses defined in `ella/agents/protocol.py`.

```python
@dataclass
class SessionContext:
    chat_id: int
    focus: list[LLMMessage]    # Tier 1 — active step context (in-memory)
    goal: JobGoal | None       # Tier 2 — job-scoped shared state (Redis)
    knowledge: KnowledgeStore  # Tier 3 — permanent semantic memory (Qdrant)

@dataclass
class UserTask:
    raw_updates: list[dict]    # raw Telegram update JSON
    session: SessionContext

@dataclass
class HandoffMessage:
    payload: Any               # list[MessageUnit] | ReplyPayload | list[Task]
    session: SessionContext    # forwarded unchanged through the pipeline
```

The `session` object is passed through every agent. Agents **read** from it, **append** to `focus`, and **write** to `goal` / `knowledge`. The next agent always receives the accumulated context.

---

## Three-Tier Memory Architecture

Memory is split across three tiers based on scope and access frequency.

```
Identity Layer              The Focus (Tier 1)          The Goal (Tier 2)           The Knowledge (Tier 3)
──────────────────────────  ─────────────────────       ──────────────────────────  ─────────────────────────
Scope: permanent config     Scope: per tool per turn    Scope: one conversation     Scope: permanent
Storage: ~/Ella/*.md files  Storage: in-memory          Storage: Redis hash         Storage: Qdrant vectors
         + ella_identity                                                             (4 collections)
         collection (Qdrant)
Access: every prompt        Access: every LLM token     Access: once per turn       Access: once per turn
Contents:                   Contents:                   Contents:                   Contents:
  • Identity.md               • current user message(s)   • objective (the "why")     • compact exchange summaries
    (name, age, timezone,     • this tool's input         • mood (current emotion)    • task patterns
     relationship w/ user)    • this tool's result        • step summaries            • user preferences
  • Soul.md                   • LLM's isolated reasoning  • tool focuses (findings)   • identity sections
    (personality, tone)                                   • shared notes                (ella_identity)
  • User.md
    (who the user is)
Question answered:          Question answered:          Question answered:          Question answered:
  "Who is Ella?"              "What is this tool doing?"  "Why are we doing this?"    "What did I know before?"
Lifecycle:                  Lifecycle:                  Lifecycle:                  Lifecycle:
  Loaded at startup           Created per tool per turn   Created on first message    Never deleted
  Hot-reloaded on file save   Discarded after reasoning   Objective updated only on   (TTL/explicit delete only)
  Identity re-embedded on     (isolated scratchpad)       topic shift (LLM-detected)
  every reload (Qdrant)                                   TTL = 24h idle (Redis)
```

### Identity Files (~/Ella/)

Three markdown files in `~/Ella/` define who Ella is, who the user is, and how they relate. They are loaded at startup, cached, and injected into every prompt as a fixed system layer — sitting between the base persona and the conversation goal.

| File | Purpose | Examples |
|---|---|---|
| `Identity.md` | Who Ella is — plus her **relationship with the user** | Name, age, timezone, country, background, relationship dynamic, how she addresses the user |
| `Soul.md` | Her inner character | Personality, tone, emotional style, values, quirks, **emoji use** |
| `User.md` | Who the user is | Date of birth, gender, ethnicity, country of residence, profession |

The **relationship with the user** lives inside `Identity.md` under `## Relationship with User`. This means Ella's tone — warmth, directness, how she addresses the user, how much she pushes back — adapts naturally based on a described dynamic rather than a fixed role label.

These files serve two roles simultaneously:
1. **Prompt context** — compiled into a single block by `_compile_prompt()` in `identity.py` and injected into every LLM call as a fixed system layer.
2. **Long-term memory** — each section of each file is embedded as a separate vector in the `ella_identity` Qdrant collection. This lets Ella semantically recall specific facets of her identity (e.g. her personality, her relationship with the user) as part of the normal knowledge-recall pipeline.

**Tone and emoji style** are driven jointly by `Soul.md` and the relationship section in `Identity.md`. The `_compile_prompt()` function also appends a `[Tone & emoji style]` block that tells the LLM to calibrate warmth and emoji frequency to match the described relationship dynamic.

**Default files** are kept in `identity/` in the repo and copied to `~/Ella/` by `deploy.sh` — but only if the file doesn't already exist, so your customisations are never overwritten on re-deploy. Running `uninstall.sh --rm` removes `~/Ella/` along with model files.

**On file change — full state reset + knowledge refresh:**
When the watcher detects a change in `~/Ella/`, it:
1. Reloads and recompiles the identity cache (takes effect on the next prompt)
2. Calls `refresh_identity_knowledge()` — wipes and re-embeds `ella_identity` in Qdrant
3. Calls all registered reset callbacks — currently `TelegramPoller._on_identity_changed`
4. That callback deletes each active session's Redis goal and clears `_sessions`

The next message from any chat starts a completely fresh conversation — new goal, clean focus, updated long-term identity — shaped by the new identity files.

**Prompt injection order:**
```
1. SYSTEM_PERSONA          (base character — "Ella is a real person, warm…")
2. Identity layer          (~/Ella/Identity.md + Soul.md + User.md)
3. Long-term memory        (Qdrant recall — what she remembers)
4. Conversation goal       (Redis — why we're here + recent history)
5. Focus                   (current user message + tool scratchpad)
```

**Anti-hallucination rules (enforced via both SYSTEM_PERSONA and _BRAIN_SYSTEM):**

Ella never invents facts, names, events, or memories. When information is absent from context, exactly one of three responses applies:

| Situation | Required behaviour |
|---|---|
| Fact about the world Ella is unsure of | Use `web_search` to verify first, then reply from the result |
| Shared past / memory of the user | Do NOT fabricate — ask the user to share it: *"I don't think I remember that — can you remind me?"* |
| Personal detail about the user not in identity files | Ask the user directly, never guess |

Ella's own identity, feelings, opinions, and personality are always answered from the identity layer — never searched.

### Memory Flow Per Conversation

```
First message from a chat:
   → TelegramPoller creates SessionContext(focus=[], goal=None) for chat_id
   → IngestionAgent clears focus, populates it with current user message(s) only
   → BrainAgent recalls from Knowledge (top-K) → seeds Goal's shared_notes
   → BrainAgent creates new JobGoal(objective="Social conversation — be a good
     listener and enjoyable company.") → persists to Redis
   → build_focus_prompt(): system → knowledge → goal (no history yet) → focus
   → LLM ReAct loop: tool calls stay inside focus as scratchpad
   → Reply + tasks produced
   → Ella's reply appended to focus (so summarise_focus captures both sides)
   → StepSummary written to Goal (goal stays "running")
   → ReplyAgent sends voice + detail text, stores exchange summary to Knowledge

After process restart (session wiped, goal=None):
   → TelegramPoller creates fresh SessionContext(goal=None)
   → BrainAgent checks ella:chat:goal:{chat_id} index in Redis
   → If found: loads the previous JobGoal, then checks session-gap:
       - If the last step's completed_at is within KNOWLEDGE_CONV_RECALL_MINUTES (15 min):
           resume as if nothing happened (same session)
       - If it is older than 15 min: discard old goal, create a fresh one
           (prevents stale objective + old conversation steps bleeding into a new chat session)
   → If not found (e.g. 7-day TTL expired): creates a new goal as normal

Each subsequent message from the same chat:
   → IngestionAgent clears focus → populates with current user message(s) only
   → BrainAgent loads existing Goal from Redis (or restores via chat index)
   → GoalStore.bind_chat(chat_id, job_id) refreshes the index on every turn
   → Topic-shift check (keyword + embedding, zero LLM cost) runs before recall:
       - Explicit phrase ("change topic", "let's talk about", etc.) → instant shift=True
       - Embedding similarity < 0.25 between message and goal objective → shift=True
       - On shift: conversation memory skipped, step history cleared, objective updated
   → BrainAgent recalls from Knowledge (identity always; conversations only if no shift)
   → build_focus_prompt(): system → identity → knowledge → goal+mood+history → focus
   → LLM sees compact conversation history from Goal summaries, fresh current input
   → Reply produced → mood persisted to Redis → StepSummary appended → cycle continues

Conversation ends (idle > GOAL_TTL_SECONDS = 24h):
   → Redis TTL expires → Goal removed automatically
   → Next message creates a fresh Goal; in-memory SessionContext resets on restart

Long-term memory:
   → ReplyAgent stores each exchange to Knowledge (Qdrant) after every reply
   → Knowledge recall seeds the next Goal's shared_notes for cross-session context
```

### Focus Lifetime

`session.focus` is cleared at the start of every turn by `IngestionAgent`, then populated with only the current user message(s). During the BrainAgent ReAct loop, tool call inputs and results are appended to focus as a scratchpad. The assistant's reply is appended last so `summarise_focus()` can capture both sides before the step summary is written to The Goal.

The Focus never accumulates across turns. Conversation history is derived entirely from The Goal's compact step summaries (last `_MAX_HISTORY_STEPS = 10` turns injected by `build_focus_prompt()`), keeping the LLM prompt well within the context window regardless of conversation length.

### Why Redis for The Goal (not Qdrant)?

The Goal is structured key-value data accessed on every message in a conversation — potentially every few seconds. Redis gives sub-millisecond read/write. Qdrant would add unnecessary embedding overhead for data that does not need semantic search.

---

## Tool System — Dynamic Hot-Reload

### Authoring a Custom Tool

Drop a `.py` file into `ella/tools/custom/`. It will be picked up on the next message batch with **no restart required**.

```python
# ella/tools/custom/my_tool.py
from ella.tools.registry import ella_tool

@ella_tool(
    name="fetch_weather",
    description="Get current weather for a city. Returns temperature and conditions.",
)
def fetch_weather(city: str, unit: str = "celsius") -> str:
    """city: city name. unit: celsius or fahrenheit."""
    import urllib.request, json
    # ... implementation ...
    return f"{city}: 22°C, partly cloudy"
```

**Rules for tools:**
- Decorate with `@ella_tool(name=..., description=...)` — this auto-registers on import
- Function signature parameters become the JSON schema for LLM function-calling
- Docstring lines `param: description` are used as parameter descriptions
- The function can be sync or async
- Return a string (or anything serialisable to string)

### Built-in Tools

| Tool | File | Description |
|---|---|---|
| `web_search` | `ella/tools/builtin/web_search.py` | DuckDuckGo search (no API key) |
| `run_shell` | `ella/tools/builtin/run_shell.py` | Safe shell execution (blocklist enforced) |
| `read_file` | `ella/tools/builtin/read_file.py` | Read local files (50 KB cap) |
| `write_file` | `ella/tools/builtin/write_file.py` | Write/append to local files |

### ToolRegistry Lifecycle

1. **Startup**: scan `ella/tools/builtin/` and `ella/tools/custom/`, import every `.py` file
2. **Watcher**: `watchfiles.awatch()` runs as a background asyncio task monitoring both directories
3. **File added/modified**: `importlib.reload(module)` → re-registers `@ella_tool` functions atomically
4. **File deleted**: removes all tools from that module from the registry
5. **Thread-safety**: `asyncio.Lock` guards all registry mutations

### BrainAgent Pipeline

The BrainAgent runs a five-phase pipeline per turn. The LLM is loaded once at the start of `handle()` and shared across all phases to avoid repeated model load/unload overhead.

```
Phase 0 — Setup
  Load Goal from Redis (must precede topic-shift and recall)
  Topic-shift check (zero LLM cost — used only to skip stale conversation recall):
    a. Keyword match: "change topic", "move on", "let's talk about", etc. → instant True
    b. Embedding cosine similarity(message, goal.objective) < 0.25 → True
    On shift: skip conversation recall, clear steps_done + tool_focuses
  Recall from Knowledge:
    - ella_identity: always top-2 relevant chunks
    - ella_conversations: top-K compact summaries within the last N minutes
        (default: 15 minutes, controlled by KNOWLEDGE_CONV_RECALL_MINUTES env var)
        (skipped entirely on topic shift)
        Tight window prevents old conversation threads from bleeding into new topics.
    - ella_task_patterns: skipped on topic shift
    - ella_topic_knowledge: always queried (not gated on topic shift); injected into
        planner to prevent redundant learn skill triggers; presented to main LLM as
        active knowledge Ella can draw on (not suppressed like background context)

Phase 1 — History Summarisation + Objective Update (summarise_recent_history)
  Filter goal.steps_done to steps completed within last 15 minutes (max 15 steps)
  Call LLM (fast, small prompt) → produces JSON with three fields:
    - condensed_history: 3-5 sentence paragraph of what was discussed
    - current_topic: short phrase label (e.g. "planning a hiking trip")
    - objective: one sentence describing what Ella should be doing right now
        (e.g. "Keep the user company as they share travel memories")
  LLM-generated objective is persisted to Redis every turn (goal.objective),
  replacing the stale embedding-similarity heuristic update from before.
  current_topic is injected into Tier 2 block in build_focus_prompt()
  Falls back gracefully to empty strings if no recent history

Phase 2 — Upfront Task Planning (_plan_tasks)
  Build compact prompt: current_topic + condensed_history + user message + tool list
  Call LLM: "Are there tools I should run before replying to give a better answer?"
  Returns list[PlannedTask(tool_name, args, reasoning, priority)]
  Empty list = no tools needed (social chat, factual Q&A, etc.)

Phase 3a — First Reply (immediate)
  Build focus prompt: system → identity → knowledge → goal+topic+mood+history → focus
  If planned_tasks: append hint: "You are about to run N tools — craft a brief opening
    response. Do not include results yet."
  Call LLM (skip_tools=True — single pass, no ReAct loop)
  → JSON reply: sentences[], language, emotion, user_emotion, tasks
  Send first reply via ReplyAgent (user hears Ella immediately)
  Fire any Celery long-running tasks from tasks[] — supported types: coding, shell.
    'document' and 'write' types route to codex (OpenAI) and are silently skipped
    if OPENAI_API_KEY is not configured. The brain system prompt no longer generates
    these types; research and knowledge tasks are handled via skills instead.

Phase 3b — Planned Tool Execution (_run_planned_tasks)
  For each PlannedTask (in priority order):
    a. Execute tool via registry.execute(tool_name, args)
    b. Persist ToolFocus to Goal (Redis)
    c. Call _generate_tool_update():
       LLM receives: topic + user message + first_reply_sentences + prior_updates + tool_result
       Instruction: "Share what you found in 1-2 natural sentences as a conversation
         continuation. Do NOT repeat anything already said."
    d. Send update text via sender.send_message() (text, no TTS — fast delivery)
    prior_updates accumulates so each successive update doesn't echo earlier ones

Phase 4 — Final Summary (after all tools complete)
  Build final_messages = focus_messages + [All tool results] + [What you already said]
  Instruction: "Give a final cohesive reply synthesising results. Do NOT repeat
    anything already said."
  Call LLM → JSON reply → send via ReplyAgent (full voice + detail)
  Append final StepSummary to Goal

Turn ends:
  Summarise turn (user + all replies) → StepSummary written to Goal
  Emotion engine updates (contagion + self-update from self-assessed emotion)
  LLM unloaded, Metal cache cleared
```

**When no tools are planned** (Phase 2 returns empty list), Phase 3a runs a normal ReAct loop (`_run_tool_loop` with `skip_tools=False`) and Phases 3b/4 are skipped entirely. The pipeline degrades gracefully to the original single-pass flow.

**The "what has already been said" thread:**

All LLM calls after the first reply receive a growing record of what was said earlier in the same turn:
- Per-tool update: receives `first_reply_sentences` + all `prior_updates`
- Final summary: receives `first_reply_sentences` + all `prior_updates` + all raw tool results

This ensures every piece of text Ella sends in a turn is a genuine continuation — no repetition, no echoing across the separate LLM calls.

### Reply Structure (ReplyPayload)

`BrainAgent` always produces a JSON object with these fields:

| Field | Purpose | Format constraints |
|---|---|---|
| `sentences` | Ordered list of spoken sentences (2–5 entries) | One complete thought per entry; no markdown, no URLs, **no emoji** — each synthesised as a separate voice message |
| `emojis` | Ordered list of emoji insertions between sentences | `[{"after": N, "emoji": "😊"}]` — N is 0-based index into `sentences`; -1 = before first; 999 = after last |
| `detail` | Full content for follow-up text message | Markdown, URLs, lists allowed; `null` if nothing extra |
| `language` | Reply language | `"en"` or `"zh"` |
| `mood` | Ella's current emotional state | One of: `playful`, `warm`, `thoughtful`, `excited`, `calm`, `caring`, `curious`, `teasing` |
| `tasks` | Background tasks to schedule | `[]` if none |

The LLM is the authoritative splitter — it decides where one spoken unit ends and the next begins, preventing unnatural mid-sentence breaks that regex heuristics can introduce. Duplicate or near-duplicate sentences are removed post-generation by a trigram + shorter-side coverage check before delivery.

`ReplyPayload` carries the corresponding Python fields:
- `sentences: list[str]` — LLM-authored sentence list; if empty, `ReplyAgent` falls back to `split_into_sentences()` regex splitting
- `text: str` — flat join of sentences (used for storage and logging only)
- `mood: str` — carried to TTS to select the per-mood delivery instruct string

`ReplyAgent` sends them in order:
1. **Voice messages + emojis** — Each entry in `sentences` is synthesised by Qwen3-TTS (with mood-specific instruct) → OGG/Opus → `sendVoice`. After each voice message, any LLM-requested emojis at that position (`{"after": N}`) are sent as plain `sendMessage`. Tokens that contain no speakable characters after emoji-stripping are silently skipped.
2. **Text message(s)** — `detail` is split into ≤ 4 096-char chunks and sent as `sendMessage` calls, labelled `(1/N)` when multiple.

### Per-sentence Delivery — sentences and emojis in sequence

```
LLM output (sentences[]):
  [0]: "That sounds really interesting."
  [1]: "Tell me more about it."
  [2]: "I'd love to hear the whole story."
  emojis: [{"after": 0, "emoji": "😊"}, {"after": 2, "emoji": "✨"}]

Delivery order:
  ├── sentences[0]: "That sounds really interesting."    ──► sendVoice
  ├── after[0]:     "😊"                                 ──► sendMessage
  ├── sentences[1]: "Tell me more about it."             ──► sendVoice
  ├── sentences[2]: "I'd love to hear the whole story."  ──► sendVoice
  └── after[2]:     "✨"                                 ──► sendMessage
```

Each `sendVoice` is preceded by a `record_voice` chat action. If TTS fails for every sentence, a single text fallback is sent.

---

## Mood System

Ella maintains a current emotional mood that evolves naturally with the conversation. The mood affects both the LLM's writing style (via system prompt injection) and the TTS delivery (via dynamic instruct strings).

### Mood Lifecycle

```
Turn N:
  LLM picks mood="excited" in its JSON output
  → mood persisted to Redis Goal (goal.mood)

Turn N+1:
  build_focus_prompt() reads goal.mood="excited"
  → injects "[Ella's current mood: excited]\nYou're feeling excited..."
     into the Tier 2 goal system message
  → LLM writes in excited register
  → ReplyPayload.mood = "excited"
  → tts_to_wav(mood="excited") uses instruct="excited, energetic tone, slightly faster pace, full of enthusiasm"
```

### Available Moods

| Mood | TTS instruct (Chinese) | When natural |
|---|---|---|
| `warm` | gentle and friendly, like talking with a close friend | Default, casual conversation |
| `playful` | relaxed and lively, slightly cheeky | Joking, light banter |
| `thoughtful` | slower pace, measured and calm, as if reflecting | Serious topics, reflection |
| `excited` | energetic, slightly faster, full of enthusiasm | Good news, shared enthusiasm |
| `calm` | peaceful and soothing, soft and quiet | Late night, comfort |
| `caring` | gentle and attentive, with warmth and concern | User is stressed or down |
| `curious` | inquisitive, slight upward lilt, exploratory | Interesting topic, questions |
| `teasing` | slightly mischievous, playfully teasing | Playful back-and-forth |

The mood starts at `"warm"` on every new conversation. The LLM shifts it naturally — it's not forced or scripted. The `SPEECH_INSTRUCT` in `.env` acts as a fallback only when the LLM picks a mood not in the table above.

### Conversation Memory — Compact Summaries

Exchanges stored in Qdrant are compressed to a one-line digest at write time:

```
User: Any good book recommendations? | Ella: Yes, I have a few in mind — what genre do you prefer?
```

The embedding vector is computed from the full raw text (for accurate semantic retrieval), but only the summary is stored in the payload — preventing the LLM from copying verbatim text from recalled memories. Legacy points without a `summary` field fall back to 60-char truncation.

---

## Model Selection and RAM Budget (M1 Pro 32GB)

Models are **never all loaded simultaneously**. STT, VL, and Chat models are loaded on-demand, used, then unloaded with `del model; mx.clear_cache()`. TTS and embeddings are permanent residents.

| Stage | Model | RAM | Loaded |
|---|---|---|---|
| TTS | Qwen3-TTS-1.7B-8bit | ~2 GB | **Permanent resident** |
| STT | whisper-large-v3-turbo | ~1.5 GB | On-demand |
| Photo/Video | Qwen2.5-VL-7B-4bit | ~5–6 GB | On-demand |
| Chat/routing LLM | Qwen3-14B-4bit-AWQ | ~9–10 GB | On-demand |
| Embeddings | MiniLM-L12-v2 | ~0.4 GB | Permanent resident |

**Peak RAM at any single stage (M1 Pro 32GB):**

| Stage | Total |
|---|---|
| STT stage | ~5 GB (whisper 1.5 + TTS 2 + OS 3.5) |
| Video stage | ~13.5 GB (VL 6 + TTS 2 + OS 3.5) ← safe |
| Chat LLM stage | ~15 GB (14B 10 + TTS 2 + OS 3.5) ← safe |
| TTS stage | ~5.5 GB (TTS 2 + OS 3.5) |

---

## Infrastructure

```yaml
# docker-compose.yml
services:
  redis:    redis:7-alpine     # Celery broker + backend + Tier 2 Goal store
  qdrant:   qdrant/qdrant      # Tier 3 Knowledge store
```

Start with:
```bash
docker compose up -d
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | *(required)* | Bot token from @BotFather |
| `MLX_CHAT_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Chat LLM |
| `MLX_VL_MODEL` | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | Video VL model |
| `MLX_WHISPER_MODEL` | `mlx-community/whisper-small-mlx` | STT model |
| `TTS_MODEL` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | Qwen3-TTS MLX model |
| `SPEAKER_WAV_PATH` | `assets/speaker.wav` | Reference voice WAV for Qwen3-TTS cloning (≥3s) |
| `TTS_VOICE` | `Chelsie` | Built-in voice when no reference WAV is present |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection |
| `OPENAI_API_KEY` | *(optional)* | Required for Codex task routing |
| `TOOLS_CUSTOM_DIR` | `ella/tools/custom` | Hot-reload watched directory |
| `MAX_TOOL_ROUNDS` | `5` | Max ReAct iterations per message batch |
| `GOAL_TTL_SECONDS` | `86400` | Redis TTL for JobGoal (24h) |
| `KNOWLEDGE_RECALL_TOP_K` | `5` | Qdrant results recalled at job start |
| `KNOWLEDGE_FRESHNESS_DAYS` | `30` | Days before topic knowledge is flagged as potentially stale |
| `FETCH_CACHE_HOURS` | `24` | Hours before a cached webpage MD file is re-fetched |

---

## Dependency Compatibility Notes

These compatibility issues have been resolved and their fixes are baked into `deploy.sh` and `requirements.txt`. They are documented here for reference.

| Issue | Root Cause | Fix |
|---|---|---|
| `AttributeError: AsyncQdrantClient has no 'search'` | `qdrant-client` ≥ 1.9 replaced `.search()` with `.query_points()` | `knowledge.py` uses `.query_points()` |
| `FileNotFoundError: ffmpeg` | `mlx-whisper` requires `ffmpeg` system binary to decode audio | `deploy.sh` installs `ffmpeg` via Homebrew |
| `TypeError: NoneType not iterable` in `video_processing_auto.py` | `transformers` `VIDEO_PROCESSOR_MAPPING_NAMES` contains `None` entries | Patched installed file with `if extractors is None: continue` guard |
| `RuntimeError: Event loop is closed` in Celery worker | `httpx.AsyncClient` and `redis.asyncio` client are cached as module-level singletons. Celery forks workers from the main process; the forked child inherits these objects bound to the parent's event loop, which no longer exists | `worker_process_init` signal in `celery_app.py` resets all async singletons to `None` after each fork. `httpx.AsyncClient` is also created lazily on first use in `TelegramSender` so it binds to the fresh loop created by `execute_task` |

### System Requirements

- **macOS** on Apple Silicon (M1/M2/M3/M4) — MLX requires Metal GPU
- **Python 3.11+** — mlx-audio and mlx-lm support 3.11, 3.12, 3.13
- **Docker** — for Redis and Qdrant
- **ffmpeg** — for audio conversion by mlx-whisper and OGG encoding (`brew install ffmpeg`)

---

## Running Ella

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN at minimum
```

### 3. Start infrastructure
```bash
docker compose up -d
```

### 4. Initialise Qdrant collections
```bash
python scripts/init_qdrant.py
```

### 5. Start Celery worker (separate terminal)
```bash
celery -A ella.tasks.celery_app worker --loglevel=info --concurrency=1
```

### 6. Start Ella
```bash
python -m ella.main
```

---

## Project Structure

```
ai.Ella/
├── ella/
│   ├── config.py                      # pydantic-settings configuration
│   ├── main.py                        # startup + poller loop + emotion decay task
│   ├── communications/                # All inbound/outbound channel adapters
│   │   └── telegram/                  # Telegram-specific implementation
│   │       ├── poller.py              # getUpdates long-poll loop
│   │       ├── sender.py              # sendMessage / sendVoice helpers
│   │       └── models.py              # Pydantic models for Telegram payloads
│   ├── agents/
│   │   ├── protocol.py                # SessionContext, UserTask, HandoffMessage
│   │   ├── base_agent.py              # BaseAgent ABC
│   │   ├── ingestion_agent.py         # Media → ordered list[MessageUnit]
│   │   ├── brain_agent.py             # Memory recall → tool loop → reply + tasks
│   │   ├── reply_agent.py             # TTS → sendVoice → store to Qdrant
│   │   └── task_agent.py              # Enqueue → Celery → monitor → notify
│   ├── ingestion/
│   │   ├── text_handler.py            # Strip Telegram HTML formatting
│   │   ├── voice_handler.py           # mlx-whisper STT + SER (returns text + VocalEmotion)
│   │   ├── voice_handler.py           # Whisper STT — transcript only, emotion by LLM
│   │   ├── video_handler.py           # OpenCV frames + Qwen2.5-VL summary
│   │   ├── photo_handler.py           # Qwen2.5-VL single-image description
│   │   └── sequencer.py               # Sort by message_id
│   ├── tools/
│   │   ├── registry.py                # ToolRegistry + @ella_tool decorator
│   │   ├── social_base.py             # Shared SocialPost dataclass for all social tools
│   │   ├── builtin/
│   │   │   ├── web_search.py
│   │   │   ├── run_shell.py
│   │   │   ├── read_file.py
│   │   │   ├── write_file.py
│   │   │   ├── download_file.py       # Download remote file → ~/Ella/downloads/
│   │   │   ├── read_pdf.py            # PDF → Markdown file (cached)
│   │   │   ├── fetch_webpage.py       # Webpage → Markdown file (cached by FETCH_CACHE_HOURS)
│   │   │   └── social_rednote.py      # Rednote search → top-K posts + all comments
│   │   └── custom/                    # Drop user tools here (hot-reloaded)
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseSkill, SkillContext, SkillCheckpoint, SkillResult
│   │   ├── registry.py                # SkillRegistry + @ella_skill decorator (hot-reload)
│   │   ├── execution.py               # SkillExecutionRegistry: start/resume/cancel lifecycle
│   │   ├── checkpoint.py              # SkillCheckpointStore: Redis + MySQL dual-write
│   │   ├── builtin/
│   │   │   ├── learn.py               # LearnSkill: Research→Analyse→Accumulate loop
│   │   │   └── research.py            # ResearchSkill: web + social + file sourcing
│   │   └── custom/                    # Drop user skills here (hot-reloaded)
│   ├── tts/
│   │   └── qwen3.py                   # Qwen3-TTS singleton + 27-emotion bilingual delivery map
│   ├── emotion/
│   │   ├── __init__.py
│   │   ├── models.py                  # EmotionProfile, AgentState, UserState, PersonalityTraits
│   │   ├── store.py                   # async MySQL EmotionStore (upsert, read, history)
│   │   └── engine.py                  # contagion, self-update, decay algorithms
│   ├── memory/
│   │   ├── embedder.py                # sentence-transformers (multilingual)
│   │   ├── focus.py                   # Tier 1: in-memory LLMMessage helpers
│   │   ├── goal.py                    # Tier 2: JobGoal + GoalStore (Redis)
│   │   ├── identity.py                # Identity loader + Personality.json/md loading
│   │   └── knowledge.py               # Tier 3: KnowledgeStore (Qdrant)
│   └── tasks/
│       ├── celery_app.py              # Celery app init
│       └── worker.py                  # Task routing + execution
├── assets/
│   ├── .gitkeep                       # Keeps folder in git
│   ├── .gitignore                     # Excludes audio files from git
│   └── speaker.m4a                    # (local only) Reference voice for TTS cloning
├── identity/
│   ├── Identity.md                    # Default identity template
│   ├── Soul.md                        # Default soul/personality template
│   ├── User.md                        # Default user profile template
│   ├── Personality.json               # Default emotion engine trait values
│   └── Personality.md                 # Default personality narrative
├── scripts/
│   ├── init_qdrant.py                 # Create Qdrant collections (idempotent)
│   ├── migrate_emotion.sql            # Create emotion MySQL tables (idempotent)
│   └── migrate_skills.sql             # Create skill run tables (idempotent)
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── Documentation/
    └── architecture.md                # This file
```

---

## Skill System

### Overview

Skills extend Ella's capabilities beyond single tool calls. A **skill** is a named, stateful, multi-step workflow that orchestrates tools and other skills to produce a durable outcome — typically stored in long-term memory.

**Tools vs. Skills:**

| | Tool | Skill |
|---|---|---|
| Scope | Single, atomic operation | Multi-step workflow |
| State | Stateless — fire and forget | Stateful — checkpointed |
| Duration | Seconds | Minutes to hours |
| Resumable | No | Yes — survives reboots |
| Output | Inline result string | Knowledge stored in Qdrant |

### Architecture

```
BrainAgent
    │
    ├─── SkillRegistry          (catalogue: what skills exist)
    │        @ella_skill decorator
    │        hot-reload watcher
    │        get_skills_schema() → injected into task planner prompt
    │
    └─── SkillExecutionRegistry (runtime: what executions are running)
             start() / resume() / cancel() / list_active()
             │
             └─── SkillCheckpointStore
                      Redis  ella:skill:{run_id}  (no TTL — fast working copy)
                      MySQL  ella_skill_runs      (permanent backup)
```

### Key Components

**`SkillRegistry`** (`ella/skills/registry.py`)
- Stateless catalogue of available skill definitions
- Decorator-based registration: `@ella_skill(name="learn", description="...")`
- Hot-reload watcher on `ella/skills/builtin/` and `ella/skills/custom/`
- `get_skills_schema()` returns `{name: description}` injected into BrainAgent task planner

**`SkillExecutionRegistry`** (`ella/skills/execution.py`)
- Runtime lifecycle manager — knows what executions are currently running or paused
- `start(skill_name, goal, ...)` — creates new execution, runs skill
- `resume(run_id, ...)` — loads paused **or failed** checkpoint, re-enters skill at saved phase
- `cancel(run_id)` — marks cancelled, clears Redis
- On Ella startup: scans MySQL for paused executions, notifies each user

**`SkillCheckpointStore`** (`ella/skills/checkpoint.py`)
- Dual-write: Redis (fast, no TTL) + MySQL (permanent, survives volume loss)
- On Redis miss: reconstructs from MySQL row (including `notes_snapshot`), re-saves to Redis before resuming
- `save(checkpoint)` → Redis + MySQL upsert
- `load(run_id)` → Redis first, MySQL fallback
- `list_paused(chat_id)` → used at startup for notifications
- `list_resumable(chat_id, max_age_hours=24)` → returns recent `failed` or `paused` runs eligible for resume

**Failure recovery:** When a `learn` skill run fails mid-cycle (e.g. crash, OOM, sub-skill unregistered), all collected notes are preserved in `ella_skill_runs.notes_snapshot`. On the next trigger of the same topic, BrainAgent's planner detects the failed run via `list_resumable()` and passes `resume_run_id` to `execution.resume()`. `LearnSkill.run()` detects non-empty `context.notes` from the checkpoint and skips the research phase entirely, jumping directly to Analyse → Accumulate with the previously collected data.

**`SkillCheckpoint`** (`ella/skills/base.py`)
```python
@dataclass
class SkillCheckpoint:
    run_id: str          # UUID, also MySQL ella_skill_runs.run_id
    skill_name: str
    chat_id: int
    goal: str
    phase: str           # "research" | "read" | "analyse" | "accumulate" | "done"
    cycle: int           # current research cycle (1-indexed)
    notes: list[str]     # accumulated knowledge passages
    questions: list[str] # open questions from last Analyse phase
    artifacts: list[str] # local file paths downloaded so far
    sources_done: list[str] # URLs already processed (dedup on resume)
    status: str          # "running" | "paused" | "completed" | "failed" | "cancelled"
    updated_at: str      # ISO 8601 UTC timestamp
```

**`SkillContext`** — shared mutable state passed through the skill call chain. When a skill invokes a sub-skill via `invoke_skill()`, the same context is passed so all accumulated notes, artifacts, and questions flow into a unified result. Includes a circular invocation guard (`_active_skills` set).

### Skill Execution Lifecycle

```
User message arrives
        │
        ▼
BrainAgent.handle() — Guard 1: active-skill reply slot
  Check Redis ella:skill:reply:{chat_id}
  → Slot exists (answer=None): deliver reply to waiting skill, return
  → No slot: continue
        │
        ▼
BrainAgent Phase 2 (_plan_tasks)
    Sees [SKILLS] and tool list in task planner prompt.
    Returns either:
      tasks=[...] (tools): web_search or social_rednote for normal chat when the user asks about something (do you know X?, 你能讲一下X?, 哪里X最好?). Results are used inline; no learn skill.
      skill={name:"learn", goal:"X"} only when the user explicitly commands Ella to learn (e.g. 深入学习X, 帮我研究X, learn about X, research X).
    Learn skill is never triggered for questions or curiosity — only for explicit learn/research commands. For questions, the planner uses web_search or social_rednote as tasks instead.
        │
        ▼
BrainAgent Phase 3a — sends first reply (answers from memory if possible)
        │
        ▼
If planned_skill is not None (explicit learn command):
BrainAgent._run_planned_skill()
  Checks SkillExecutionRegistry.list_active(chat_id)
  Also checks SkillCheckpointStore.list_resumable(chat_id) for recent failed runs
  → No resumable run: SkillExecutionRegistry.start("learn", "X", on_run_id=…)
  → Paused or failed run with matching goal: SkillExecutionRegistry.resume(run_id)
        │
        ▼
LearnSkill.run(goal, context)
  Detects context.notes non-empty → is_resume=True (skip research, go to Analyse)
  For each cycle (max 3):
    1. RESEARCH  → invoke sub-skill "research" (skipped on resume for first cycle)
    2. ANALYSE   → LLM: chunked notes (12k chars/chunk, all notes covered)
    3. GAP FILL  → auto-research questions, then ask_user if still unresolved (2-min timeout, clears questions on any outcome)
  ACCUMULATE → chunk notes, store in ella_topic_knowledge (Qdrant)
  SENSITIVITY → ask user: public / internal / private / secret (default: internal, 2-min timeout)
  _unload_llm() → clear GPU cache before returning
        │
        ▼
SkillResult(summary, stored_points, artifacts, open_questions)
  → summary appended to JobGoal.shared_notes
  → summary sent to user via Telegram
```

#### Redis Slot Summary

| Key | TTL | Purpose |
|---|---|---|
| `ella:skill:reply:{chat_id}` | 2 min | Skill waiting for user answer (`ask_user`) |

#### ask_user / Reply Slot Handshake

When a skill needs user input (`context.ask_user(prompt)`):

1. Skill sends the question via Telegram.
2. `SkillCheckpointStore.set_pending_reply(chat_id, run_id, prompt)` writes a Redis slot (`ella:skill:reply:{chat_id}`, TTL 2 min).
3. The skill polls the slot every 3 seconds for up to 2 minutes.
4. On the user's next Telegram message, `BrainAgent.handle()` detects the slot (answer=None), calls `deliver_reply(chat_id, user_text)`, and returns without any further processing.
5. The polling skill sees the answer, clears the slot, and continues execution.

This prevents user replies from triggering a new skill run while one is already active.

### Built-in Skills

**LearnSkill** (`ella/skills/builtin/learn.py`)
- Triggered when user asks Ella to learn, research, or study a topic
- Gap-driven loop: Research → Analyse → fill gaps → repeat (max 3 cycles)
- LLM used only in Analyse phase — all sourcing is deterministic
- **Chunked analysis:** notes are split into 12k-char windows; each is analysed independently and questions merged — no notes are ever dropped
- LLM singleton loaded once, unloaded (with `mx.metal.clear_cache()`) before returning so BrainAgent can load its reply model without GPU OOM
- Stores chunked knowledge in `ella_topic_knowledge` Qdrant collection
- Asks user for sensitivity label before storing

**ResearchSkill** (`ella/skills/builtin/research.py`)
- Sub-skill invoked by LearnSkill; pure sourcing, no LLM
- Source chain per URL/result:

| Source | Tool chain | Output |
|---|---|---|
| Web (snippets) | `web_search` | Inline text |
| Web (full page) | `fetch_webpage` → `read_file` | `~/Ella/downloads/{hash}.md` |
| PDF | `download_file` → `read_pdf` → `read_file` | `~/Ella/downloads/{name}.md` |
| Rednote (小红书) | `social_rednote` | Inline `SocialPost` JSON |

### New Tools

| Tool | File | Description |
|---|---|---|
| `download_file` | `ella/tools/builtin/download_file.py` | Download a file from a URL to `~/Ella/downloads/`. Returns local path. |
| `read_pdf` | `ella/tools/builtin/read_pdf.py` | Convert a local PDF to Markdown (cached permanently as `.md` file). Returns `.md` path. |
| `fetch_webpage` | `ella/tools/builtin/fetch_webpage.py` | Fetch a webpage, extract main content, save as Markdown (cached by `FETCH_CACHE_HOURS`). Returns `.md` path. |
| `social_rednote` | `ella/tools/builtin/social_rednote.py` | Search Rednote, filter to top-K posts by engagement score, return full post text + all comments. |

**Deterministic tool chain:** All tools that produce file output save `.md` files to `~/Ella/downloads/`. The skill then uses the existing `read_file` tool to read the content. No LLM is involved until the Analyse phase.

#### LearnSkill — Analysis step (detailed)

The **Analyse** phase is the only place the learning skill uses the LLM. It turns raw research notes into a summary and a list of knowledge gaps (questions to research next or ask the user).

**Inputs**

| Input | Source | Description |
|-------|--------|-------------|
| `goal` | `LearnSkill.run(goal, context)` | The learning topic (e.g. "糖醋排骨的历史"). |
| `context.notes` | Research sub-skill + optional user input | List of strings: each string is one "note" (e.g. content of a webpage, a PDF section, or a user reply). Notes are accumulated across the Research step and any prior cycles. |

**Chunking (before LLM)**

- Notes are not sent to the LLM in one block (that would exceed context and truncate).
- `_chunk_notes(context.notes)` splits the notes into **non-overlapping chunks** of at most **12,000 characters** (~3000 tokens) each.
- Splitting is done **at note boundaries**: the separator is `\n\n---\n\n`. A note is never cut in the middle; when adding the next note would exceed 12k chars, the current buffer is emitted as one chunk and a new chunk starts with that note.
- So: **input** = `context.notes` (list of strings) → **output** = list of strings (chunks), each chunk = one or more full notes joined by `\n\n---\n\n`.

**Per-chunk LLM call**

- For **each chunk**, the skill calls the LLM once with:
  - **System prompt** (`_ANALYSE_SYSTEM`): instructs the model to act as an expert research analyst, (1) synthesise the notes into a concise summary, (2) identify specific knowledge gaps or questions for further research. Response must be **only** a single-line JSON: `{"summary": "...", "questions": ["question1", "question2"]}`. Summary under 300 words, at most 3 questions; if comprehensive, `questions` can be `[]`.
  - **User message**: `Topic: {goal}\n\nResearch notes (chunk N/M):\n{chunk}\n\nSynthesise and identify gaps.`
- The raw LLM output is then:
  - Stripped of any `<think>...</think>` blocks (Qwen3 thinking).
  - Parsed: the first JSON object in the response is extracted with a regex `\{.*\}`; that object must have `summary` and `questions` (array of strings).

**Merging results across chunks**

- **Summary**: Only the **last** chunk’s `summary` is kept (`last_summary`). So with multiple chunks, the final summary is the synthesis of the **last** 12k‑char window (typically the tail of the notes).
- **Questions**: All chunks’ `questions` arrays are **merged**: each question is added to `all_questions` if it is non-empty and not already in the list. Duplicates are skipped. The final list is **capped at 3** (`all_questions[:3]`).

**Output of `_analyse`**

- Return value: `dict[str, Any]` with two keys:
  - `"summary"`: string — the last chunk’s summary (or `""` if LLM failed).
  - `"questions"`: list of up to 3 strings — merged, de-duplicated gap questions from all chunks.
- If the LLM fails to load or a chunk’s JSON parse fails, that chunk is skipped (warning logged); previous chunks’ questions and the last successful summary are still used.

**How the caller uses it**

- `LearnSkill.run()` calls `analysis = await _analyse(goal, context)` after each Research step (and again after gap-filling research).
- It does:
  - `context.questions.extend(analysis["questions"])` to add new gap questions.
  - `summary_so_far = analysis["summary"]` for the cycle’s summary (and later `final_summary` if the loop exits).
- **Gap resolution**: If there are questions and cycles remain, the skill (1) invokes Research for up to 3 of those questions, (2) calls `_analyse` again on the updated `context.notes`, (3) then may ask the user via `ask_user` for remaining gaps. So the analysis output drives both automatic follow-up research and the user-facing “Do you know anything about these?” step.
- After the loop, `final_summary` is used in `SkillResult.summary` and (if empty) replaced by a separate `_synthesise(goal, context)` call that runs one more LLM pass over the full notes to produce a 2–4 paragraph summary for the user.

**Summary table**

| Stage | Input | Output |
|-------|--------|--------|
| Chunking | `context.notes` (list of strings) | List of strings, each ≤12k chars, split at note boundaries |
| Per-chunk LLM | `goal` + one chunk + system prompt | One JSON `{summary, questions}` per chunk |
| Merge | All chunk JSONs | `summary` = last chunk’s summary; `questions` = merged, de-duplicated, max 3 |
| Return | — | `{"summary": str, "questions": list[str]}` |

**What chunking does NOT do**

- **Reformatting / sanitising:** Chunking only splits by size at note boundaries. It does **not** strip URLs, HTML, headers/footers, or other boilerplate. Some cleaning happens earlier at the **tool** level (e.g. `fetch_webpage` uses trafilatura to extract main content and convert to Markdown), but notes still include source labels like `[Web page: {url}]` and the saved page content often includes a `# {url}\n\n_Source: {url}_` header. Research appends raw snippets (web search, PDF, Rednote) with those prefixes intact; no step in LearnSkill removes them before analysis or storage.
- **Quality / relevance:** There is no scoring or filtering of notes by relevance. All notes are passed to the LLM and into storage. The Analyse step asks the model to synthesise and identify gaps, but **not** to drop or down-rank low-relevance content.
- **Single final MD document:** Accumulated knowledge is **not** written to one consolidated Markdown file. It is stored as many small chunks (≈2000 chars each) in Qdrant (`ella_topic_knowledge`). `context.artifacts` holds the list of source file paths (e.g. `~/Ella/downloads/web_xxx.md`) from Research; no merged “final knowledge.md” is produced for the run.

### Knowledge Storage — `ella_topic_knowledge`

A new Qdrant collection stores all learned knowledge permanently.

**Payload fields:**
- `topic` — normalised topic label
- `source_url` — origin URL or file path
- `source_type` — `"web"` | `"pdf"` | `"rednote"` | `"user_input"`
- `chunk_text` — the knowledge passage (~512 tokens)
- `sensitivity` — `"public"` | `"internal"` | `"private"` | `"secret"` (default: `"secret"`)
- `learned_at` — ISO 8601 UTC timestamp of when the chunk was stored
- `learned_by_chat_id` — chat that triggered the learning run

**Staleness detection:** BrainAgent and LearnSkill compare `learned_at` against `KNOWLEDGE_FRESHNESS_DAYS`. Stale results are surfaced with: *"This was learned on [date] — it may be out of date."*

### Configuration

| Variable | Default | Description |
|---|---|---|
| `KNOWLEDGE_FRESHNESS_DAYS` | `30` | Days before learned knowledge is considered stale. Used in Tier 3 recall annotations and LearnSkill pre-checks. |
| `FETCH_CACHE_HOURS` | `24` | Hours before a cached webpage (in `~/Ella/downloads/`) is re-fetched. PDFs are cached permanently (no time check). |

### Adding Custom Skills

Drop a `.py` file into `ella/skills/custom/`. It will be picked up on the next message — no restart required.

```python
# ella/skills/custom/my_skill.py
from ella.skills.base import BaseSkill, SkillContext, SkillResult
from ella.skills.registry import ella_skill

@ella_skill(
    name="summarise_channel",
    description="Fetch and summarise all recent posts from a Telegram channel.",
)
class SummariseChannelSkill(BaseSkill):
    name = "summarise_channel"
    description = "Fetch and summarise all recent posts from a Telegram channel."

    async def run(self, goal: str, context: SkillContext) -> SkillResult:
        await context.send_update("Fetching channel posts…")
        await context.checkpoint("fetch")
        # ... implementation ...
        return SkillResult(summary="Done", stored_points=0, artifacts=[], open_questions=[])
```

### MySQL Tables

Created by `scripts/migrate_skills.sql` (rerunnable):

- **`ella_skill_runs`** — permanent audit log and checkpoint backup. Columns: `run_id`, `chat_id`, `skill_name`, `goal`, `status`, `phase`, `cycle`, `stored_points`, `sources_done` (JSON), `notes_snapshot` (LONGTEXT), `summary`, timestamps.
- **`ella_skill_open_questions`** — unresolved knowledge gaps after max cycles. Foreign key → `ella_skill_runs`.

---

## Emotion Engine

The emotion engine gives Ella a persistent, quantitative emotional state that evolves across all conversations. It is grounded in three established models from affective computing research.

### Theoretical Foundation

**PAD Model (Mehrabian & Russell, 1974)**

Every emotion is represented as a point in three continuous dimensions:
- **Pleasure** (`valence`, -1 → +1): positive vs negative affect
- **Arousal** (`energy`, 0 → 1): activation level
- **Dominance** (`dominance`, 0 → 1): sense of control and agency

Dominance is the critical third dimension — it separates `anger` (high-arousal negative, high-dominance) from `fear` (high-arousal negative, low-dominance), enabling a tough personality to respond to challenge with `excitement` while a timid one responds with `anxiety`.

**Bosse / Pereira-Paiva Generic Contagion Model (ACII 2011)**

Applied across all three PAD dimensions:

```
E_new = E_old + α × S_ecs × (E_source − E_old)
```

Where `α` = Ella's expressiveness (sender expressiveness) and `S_ecs` = per-emotion-category susceptibility from Doherty's Emotional Contagion Scale.

**Doherty's Emotional Contagion Scale (ECS, 1997)**

Defines five independent susceptibility categories: happiness, love, fear, anger, sadness. Each has its own weight in `Personality.json`, allowing Ella to catch joy easily but resist anger — or vice versa depending on personality configuration.

### Emotion Vocabulary

27 emotions from Cowen & Keltner (2017), each with empirical PAD coordinates from the NRC VAD Lexicon, a baseline intensity, momentum, and bilingual TTS delivery instructions:

`admiration` `adoration` `aesthetic_appreciation` `amusement` `anger` `anxiety` `awe` `awkwardness` `boredom` `calmness` `confusion` `craving` `disgust` `empathic_pain` `entrancement` `excitement` `fear` `horror` `interest` `joy` `nostalgia` `relief` `romance` `sadness` `satisfaction` `surprise` `sexual_desire`

After any PAD update, the emotion label is resolved by **nearest-neighbour lookup in PAD space** — no hardcoded label mappings.

### Personality Configuration

Two files in `~/Ella/` control the engine's behaviour:

**`Personality.json`** — machine-readable trait numbers consumed by the algorithms:

| Trait | Effect |
|---|---|
| `resilience` | Attenuates negative valence pulls; protects dominance from collapsing under pressure |
| `volatility` | Speed of state change — scales the ECS susceptibility |
| `expressiveness` | Maps to `α` in the Bosse equation; how strongly state is pulled toward the source |
| `optimismBias` | Constant offset added to valence after every contagion or decay pass |
| `dominanceBase` | Ella's natural baseline dominance — key differentiator for tough vs. timid responses |
| `ecs.happiness` | ECS susceptibility: joy, excitement, amusement, awe, relief, admiration, … |
| `ecs.love` | ECS susceptibility: romance, sexual_desire, adoration, craving |
| `ecs.fear` | ECS susceptibility: fear, horror, anxiety, awkwardness |
| `ecs.anger` | ECS susceptibility: anger, disgust |
| `ecs.sadness` | ECS susceptibility: sadness, empathic_pain, nostalgia, boredom, confusion |

**`Personality.md`** — narrative description of the same traits in plain language, injected into the LLM prompt alongside `Soul.md`. Tells the LLM *why* Ella's emotional responses look the way they do. Hot-reloaded — changes take effect on the next turn.

### Storage

One MySQL row per user (`chat_id`) in `ella_emotion_state` — cross-session, never expires. A rolling history of the last 10 state changes per user in `ella_emotion_history`. Schema created by `scripts/migrate_emotion.sql`.

### Voice Input — Emotion Detection

For voice messages, Whisper transcribes the audio to text. Emotion is then detected from the **transcript text** by the brain LLM — the same model that generates Ella's reply.

**Why not acoustic SER?** Acoustic-only Speech Emotion Recognition models (wav2vec2, HuBERT, emotion2vec+) are trained on acted/exaggerated speech datasets. They cannot reliably detect emotion in natural conversational Chinese — a speaker saying "滚，别理我" (get lost, don't talk to me) in a controlled tone is classified as "neutral" at 100% confidence by acoustic models. The LLM, reading the words, correctly identifies anger.

**How it works:**
1. Whisper transcribes the voice message to text.
2. The transcript is passed to the brain LLM as a user message, with a `[source=VOICE]` marker in the log.
3. The LLM infers `user_emotion` from the transcript content as part of its normal JSON output.

This approach is accurate for both Chinese and English without any additional model or dependency.

### Per-Turn Flow

```
VOICE MESSAGE:
[Download OGG → Whisper STT → transcript text]
        ↓
[MessageUnit carries transcript text (source="voice")]
        ↓
TEXT / VOICE:
[Load AgentState from MySQL + PersonalityTraits from Personality.json]
        ↓
[Inject emotion context block into focus prompt]
        ↓
[LLM JSON output includes:]
  "emotion": "<one of 27 labels>"        ← Ella's self-assessment for this turn
  "user_emotion": {                       ← user emotion inferred from transcript text
    "label": "...", "valence": 0.0,
    "energy": 0.0, "dominance": 0.0, "intensity": 0.0
  }
        ↓
[use LLM user_emotion if intensity > 0.25]
        ↓
[apply_contagion()]  [apply_self_update(chat_id, emotion_label)]
        ↓
[emotion_label → TTS delivery instruction selection (bilingual)]
```

### Decay

A background asyncio task (`_decay_loop` in `main.py`) runs every 4 hours and calls `apply_decay()` for all users with an emotion state row. State drifts toward baseline (`calmness`, valence 0.2, energy 0.4) at a rate scaled by `(1 - momentum)` — high-momentum emotions (calmness, nostalgia) resist decay while low-momentum ones (surprise, horror) fade quickly.

### Feature Flag

The entire engine is controlled by `EMOTION_ENABLED` in `.env` (default `true`). When `false`:
- No MySQL connection is opened
- `emotion` / `user_emotion` fields are omitted from the LLM schema
- The emotion context block is omitted from the focus prompt
- `tts_to_wav` falls back to `SPEECH_INSTRUCT`
- The decay loop is not started
- `Personality.json` and `Personality.md` are not loaded
