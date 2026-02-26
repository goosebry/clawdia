# Learn pipeline quality improvement — handover

## Purpose

This document summarises the **current behaviour** of Ella’s learning pipeline and the **goals for improvement**, and provides a **handover prompt** to use in a new chat to plan and implement the work.

---

## Current behaviour (summary)

### End-to-end flow

1. **LearnSkill** runs a loop: **Research → Analyse → gap resolution → repeat** (max 3 cycles), then **Accumulate** (store) and optional **synthesise** (final summary).
2. **Research** (sub-skill) collects raw material:
   - Web search (DDG) snippets → appended to `context.notes` with a `[Web search snippets for 'goal']` prefix.
   - Top search result URLs → `fetch_webpage` (HTML → Markdown via trafilatura), content read and appended as `[Web page: {url}]\n{content}` (first 4000 chars).
   - PDFs → download, `read_pdf` to MD, appended as `[PDF: {url}]\n{content}` (first 8000 chars).
   - Rednote posts → formatted with title, author, URL, engagement, comments, appended as separate notes.
3. **Analyse** (LLM):
   - Notes are **chunked by size only** (≤12k chars per chunk, split at note boundaries). No reformatting, no stripping of URLs/headers.
   - One LLM call per chunk: system prompt asks for a JSON `{summary, questions}` (synthesise + identify gaps). No instruction to filter by relevance or quality.
   - Results merged: last chunk’s summary kept; questions from all chunks merged, de-duplicated, capped at 3.
4. **Accumulate** (`_store_knowledge`):
   - Each **note** is split again into **~2000 char chunks** (by newlines), and each chunk is stored as a **separate point** in Qdrant `ella_topic_knowledge` with `topic`, `source_url`, `chunk_text`, `sensitivity`, `learned_at`, etc.
   - No single consolidated “final knowledge” document (e.g. one MD file) is produced. Artifacts are the list of source file paths from Research.

### What is **not** done today

- **Reformatting / sanitising:** Chunking does not strip URLs, HTML remnants, or source headers. Tools do some extraction (e.g. trafilatura for web pages), but notes still contain labels like `[Web page: {url}]` and often a `# {url}\n\n_Source: {url}_` header in the content. No step removes or normalises these before analysis or storage.
- **Quality / relevance:** No scoring or filtering. All notes are sent to the LLM and all chunks are stored. The Analyse prompt does not ask to drop low-relevance or off-topic content.
- **Focus / depth:** The loop is gap-driven (questions → more research), but there is no explicit “relevance to goal” or “depth vs breadth” control. Stored chunks are raw note segments, not rewritten for clarity or focus.
- **Single final knowledge artefact:** Knowledge exists only as many small Qdrant points. There is no merged “final knowledge.md” (or similar) that is high-level, readable, and easy to audit.
- **Queryability / retrieval accuracy:** Retrieval uses the same embedding model as the rest of the app over `chunk_text`. Chunks are often noisy (URLs, headers, mixed sources), which can hurt semantic match quality and accuracy when users query later.

---

## Goals for improvement

We want the **final stored knowledge** to be:

1. **High quality** — Clean, readable content; no URLs/HTML/boilerplate in stored text; consistent structure where helpful.
2. **Relevant** — Focused on the learning goal; low-relevance or off-topic material filtered out or down-weighted before storage.
3. **Focused and in-depth** — Prioritise depth on the original problem/topic; avoid storing shallow or tangential snippets unless they add value.
4. **Highly queryable** — Chunks (and any summary artefact) structured so that semantic search returns the right pieces for user questions (good recall and precision).
5. **High accuracy on query matching** — Retrieval over stored knowledge should surface the most accurate, on-topic results (e.g. correct use of `min_score` and chunk design so that queries like “糖醋排骨怎么做” don’t match loosely related cooking content).

Concrete outcomes to aim for (to be refined in planning):

- A **sanitise / normalise** step (before or as part of analysis) that strips or normalises URLs, source headers, and obvious boilerplate from note text.
- A **relevance / quality** step (e.g. per-note or per-chunk scoring/filtering) so that only sufficiently relevant content is passed to synthesis and storage; option to ask the LLM to tag or filter by relevance to goal.
- **Focus on goal and depth:** Analyse (and any new steps) explicitly consider “relevance to goal” and “depth”; optionally merge or rewrite chunks into goal-focused, self-contained passages before storage.
- **Final knowledge format:** Consider producing a **single consolidated artefact** (e.g. one Markdown document per learning run) that is human-readable and machine-friendly, in addition to or in coordination with Qdrant chunks (so retrieval can remain semantic over high-quality text).
- **Storage and retrieval:** Store only cleaned, goal-focused chunks (and optionally a summary doc); ensure chunk boundaries and content are optimised for later semantic query matching and that retrieval parameters (e.g. `min_score`, top_k) are set for high accuracy.

---

## Handover prompt (for new chat)

Copy the block below into a **new chat** to plan this work. It is self-contained so the new session can start without re-reading this file in full.

---

```
We need to improve Ella's learning pipeline so that final stored knowledge is high quality, relevant, focused on the original goal with depth, highly queryable, and provides high accuracy on query matching.

**Current behaviour (short):**
- LearnSkill: Research (web search, fetch_webpage, PDF, Rednote) → Analyse (LLM per 12k-char chunk: synthesise + gap questions) → gap resolution → Accumulate (store in Qdrant as ~2k char chunks). No single final MD file.
- Chunking is size-only (no reformatting or sanitising). Notes include source labels and URLs (e.g. "[Web page: {url}]", "# {url}\n\n_Source: {url}_"). fetch_webpage uses trafilatura for HTML→MD but the saved file still has that header; Research appends content as-is.
- No relevance/quality filtering: all notes go to the LLM and all chunks are stored. Analyse prompt does not ask to filter low-relevance content.
- Storage: many small chunks in Qdrant (ella_topic_knowledge). Retrieval uses same embedder; min_score (e.g. 0.75) filters loose matches. No consolidated "final knowledge" document per run.

**Goals to achieve:**
1. **High quality** — Stored text clean: no URLs/HTML/boilerplate in the content we store; consistent, readable structure.
2. **Relevant** — Only content relevant to the learning goal stored; filter or down-weight off-topic / low-relevance material.
3. **Focused and in-depth** — Prioritise depth on the original problem; avoid storing shallow or tangential snippets unless they add value.
4. **Highly queryable** — Chunks (and any summary artefact) structured so semantic search returns the right pieces for user questions (good recall and precision).
5. **High accuracy on query matching** — Retrieval over stored knowledge should surface the most accurate, on-topic results (chunk design and retrieval params tuned for accuracy).

**Relevant code/docs:**
- LearnSkill: `ella/skills/builtin/learn.py` (run loop, _analyse, _chunk_notes, _store_knowledge, _chunk_text).
- ResearchSkill: `ella/skills/builtin/research.py` (how notes are appended: web search, fetch_webpage+read_file, PDF, Rednote).
- fetch_webpage: `ella/tools/builtin/fetch_webpage.py` (trafilatura + header "# {url}\n\n_Source: {url}_").
- Knowledge storage / recall: `ella/memory/knowledge.py` (store_topic_knowledge, recall_topic_knowledge, min_score).
- Architecture: `Documentation/architecture.md` — "LearnSkill — Analysis step (detailed)" and "What chunking does NOT do"; "Knowledge Storage — ella_topic_knowledge".
- This handover: `Documentation/learn-pipeline-quality-handover.md`.

**Ask:** Please plan the work to achieve the goals above. In the plan:
- Propose concrete steps (e.g. sanitise/normalise step, relevance filtering, goal-focused chunking or rewriting, optional single final MD artefact, retrieval tuning).
- Call out where in the pipeline each change fits (Research vs LearnSkill vs tools vs storage/recall).
- Keep DB/APIs unchanged where possible (e.g. Qdrant schema, MySQL skill tables); prefer additive or in-place content improvement.
- Do not implement yet — output a clear plan (and optionally tasks) we can implement in a follow-up.
```

---

## References

- **LearnSkill:** `ella/skills/builtin/learn.py`
- **ResearchSkill:** `ella/skills/builtin/research.py`
- **fetch_webpage:** `ella/tools/builtin/fetch_webpage.py`
- **Knowledge store/recall:** `ella/memory/knowledge.py`
- **Architecture:** `Documentation/architecture.md` (LearnSkill, Analysis step, Knowledge Storage)
