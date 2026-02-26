# Learn pipeline quality improvement

## Overview

Improve Ella's learning pipeline so stored knowledge is high quality, relevant, goal-focused, and highly queryable. Add sanitisation, relevance + extraction (at the start of Analyse, before chunking), window-based streaming for long docs with accumulate-as-we-go, optional consolidated artefact, and retrieval tuning — without changing DB schemas or APIs.

**Production environment:** 32B main model; Mac Studio M4 Max (40-core GPU, 128GB RAM). Design for large-context segments and batched/single-model pipeline; optional parallel segment processing where supported.

---

## Current pipeline (reminder)

- **Research** appends raw text with prefixes (`[Web page: url]`, `[PDF: url]`, etc.). `fetch_webpage` adds a header; Research prepends source labels and caps lengths.
- **Analyse:** `_chunk_notes(context.notes)` runs first (≤12k chars per chunk at note boundaries); one LLM call per chunk for `{summary, questions}`; no relevance filtering.
- **Accumulate:** Each note is split with `_chunk_text` (~2k chars); each piece stored in Qdrant. No single final MD file.

---

## 1. Sanitise / normalise step

**Goal:** Stored text and text sent to the LLM should be free of URLs, source headers, and obvious boilerplate in the content we analyse and store (source URL stays in metadata).

**Where:** LearnSkill, applied to each note **before** both the relevance+extraction step and Accumulate.

**Proposal:** Add `_sanitise_note(note: str)` (or return body + source_url + source_type). Strip or normalise: first-line source label (keep for metadata, remove from body); for web/PDF, patterns like `# {url}`, `_Source: {url}_`, `---` in the header region. Keep `fetch_webpage` behaviour as-is for cached files; sanitisation happens when LearnSkill prepares notes. Reuse sanitised list for relevance+extraction and for storage.

---

## 2. Relevance and extraction (start of Analyse, before chunking)

**Goal:** Only content relevant to the learning goal is passed to synthesis and storage; when only part of a source is relevant (e.g. 10% of a PDF), extract those passages instead of keeping or dropping the whole note.

**Placement:** At the **very beginning of `_analyse`**, before `_chunk_notes` is called. Chunking currently happens inside `_analyse`; the relevance+extraction pass must run over the **unchunked** note list first.

**Flow inside Analyse:**

1. **Relevance + extraction pass** over the full list of notes (unchunked). One LLM pass that for each note returns: **keep** (use entire note), **drop**, or **extract** (list of goal-relevant passages). Preserve `source_url` / `source_type` for all kept or extracted content.
2. **Filter/expand** the note list: kept notes stay; dropped notes removed; extracted passages become separate items (or one concatenated item per source) with same source metadata.
3. **Then** call `_chunk_notes(filtered_notes)` and run the existing per-chunk synthesis + gap-questions loop unchanged.

So: **relevance+extraction (LLM over notes) → filter/expand → chunk filtered list → per-chunk synthesis+gaps**. No chunking until after relevance and extraction.

**Context window:** The LLM can only process inputs that fit its context. Long notes (e.g. 100+ pages) cannot be sent in one call. So we use **window-based processing** (see §3).

---

## 3. Long documents: stream per window, extract and accumulate

**Goal:** Handle extreme cases (e.g. 500-page PDF) in prod and dev without loading the full doc into memory or into one context. Same design for short and long notes.

**Single design:** Process each note in **windows**. For each window: run relevance + extraction; **accumulate** kept/extracted content. Short note = one window; long note = many windows. One loop, no separate “short path” vs “long path”.

**Streaming and accumulation:**

- For each note, iterate over it in **windows** (see splitting below). Never hold more than one window in context at a time.
- For each window: one LLM call (relevance + extraction). Append any kept or extracted passages to an **accumulator** for that note, with `source_url` / `source_type` preserved.
- After the last window, the accumulator is the **effective content** for that note. That is what goes into the rest of the pipeline (chunking for synthesis, then storage).

**Splitting long notes into windows:**

- **Prefer structure when present:** If the converted MD has headings (`##`, `###`), split on them first. Depends on tool output quality; many documents will have no headers.
- **Fallback:** When there are no headings or a section is still longer than the context window: split by **paragraph** (`\n\n`) or by **fixed size** (e.g. 24k–32k tokens per window). Use the largest window that fits the model’s context in production.
- **V1:** Use simple splitting (fixed-size or paragraph). Defer per-window “topic boundary” LLM and heavy semantic splitting.

**Short notes (under threshold, e.g. 24k–32k tokens):** Process in one window. Optionally **batch** several short notes in one LLM call (“For each of these N notes: keep / drop / extract passages”) to reduce calls in the common case.

**Config for dev vs prod:**

- `LEARN_MAX_TOKENS_PER_WINDOW`: production uses model’s safe limit (e.g. 32k); dev can use smaller (e.g. 8k) for resource-constrained machines.
- `LEARN_MAX_WINDOWS_PER_NOTE`: unset (or high) in production; in dev set (e.g. 5) to cap iterations per note and test the full streaming+accumulation path without stressing the machine.

---

## 4. Focus on goal and depth

**Goal:** Prefer in-depth, goal-focused content; avoid storing shallow or tangential snippets unless they add value.

**Proposal:** Update the Analyse system prompt (the synthesis+gaps part) with: (1) “Consider relevance to the learning goal”; (2) “Prefer in-depth, goal-focused content over shallow or tangential snippets.” Optional later: an LLM rewrite step that turns filtered/extracted content into goal-focused passages before storage.

---

## 5. Final knowledge artefact (optional single MD)

**Goal:** One human-readable consolidated document per learning run in addition to Qdrant chunks.

**Where:** LearnSkill, after `_store_knowledge`.

**Proposal:** Optionally write one Markdown file (e.g. `~/Ella/knowledge/{topic_slug}_{date}.md`) with goal, date, and final summary (and optionally key points). Add path to `context.artifacts`. Do not embed the full MD into Qdrant to avoid duplication; artefact is for humans and optional future “recall by file” features.

---

## 6. Storage and retrieval tuning

**What we store:** Only sanitised, filtered/extracted content; keep `source_url` and `source_type` for provenance. Chunk at ~2k chars (newline boundaries) for storage; no Qdrant schema change.

**Retrieval:** Make topic `min_score` configurable (e.g. `knowledge_topic_min_score` in config). Optionally apply a small `min_score` when building topic_snippets in general recall so clearly off-topic chunks are not injected.

---

## Implementation order (suggested)

| Step | Location        | Description |
|------|-----------------|-------------|
| 1    | LearnSkill      | Add `_sanitise_note()`; run sanitisation before relevance+extraction and before Accumulate. |
| 2    | LearnSkill      | At start of `_analyse`, add relevance+extraction pass over notes (unchunked); then filter/expand; then existing `_chunk_notes` → synthesis+gaps. |
| 3    | LearnSkill      | For notes over token threshold: window-based iteration (split by heading/paragraph/size), one LLM call per window, accumulate kept/extracted content; support batching for short notes. |
| 4    | Config          | Add `LEARN_MAX_TOKENS_PER_WINDOW`, `LEARN_MAX_WINDOWS_PER_NOTE` for dev/prod. |
| 5    | LearnSkill      | (Optional) Single consolidated MD artefact per run. |
| 6    | Config + BrainAgent | Make topic `min_score` configurable; optional min_score for topic snippets in general recall. |

---

## What stays unchanged

- Qdrant schema for `ella_topic_knowledge` (same payload fields).
- MySQL skill tables.
- ResearchSkill and fetch_webpage behaviour (optional later improvements independent).
- SkillContext and checkpoint format; notes can remain as-is in checkpoint; we derive a sanitised/filtered view for Analyse and storage.

---

## References

- LearnSkill: `ella/skills/builtin/learn.py`
- ResearchSkill: `ella/skills/builtin/research.py`
- fetch_webpage: `ella/tools/builtin/fetch_webpage.py`
- Knowledge: `ella/memory/knowledge.py`
- Architecture: `Documentation/architecture.md` (LearnSkill, Analysis step, Knowledge Storage)
- Handover: `Documentation/learn-pipeline-quality-handover.md`
