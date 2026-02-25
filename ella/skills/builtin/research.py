"""ResearchSkill — sub-skill that sources raw material from the web and social media.

This is a building block invoked by LearnSkill (and potentially other skills).
It is purely a sourcing skill: it finds URLs, downloads/fetches content, converts
everything to Markdown files, and appends the file paths to context.artifacts
and raw text snippets to context.notes.

All sourcing operations are deterministic (no LLM). The LLM is only used by the
calling skill (LearnSkill) during the Analyse phase.

Source priority order:
  1. web_search (DDG) — fast snippets
  2. fetch_webpage — full page content for top search results
  3. download_file + read_pdf — for .pdf links
  4. social_rednote — community knowledge, lived experience (Chinese-language)
"""
from __future__ import annotations

import json
import logging

from ella.skills.base import BaseSkill, SkillContext, SkillResult
from ella.skills.registry import ella_skill

logger = logging.getLogger(__name__)

_MAX_PAGES_PER_SEARCH = 5   # how many full pages to fetch from search results
_SOCIAL_ENABLED = True       # toggle for social media sourcing


@ella_skill(
    name="research",
    description=(
        "Source raw material about a topic from web search, full-page fetches, PDFs, "
        "and Rednote social media posts. Does NOT analyse or synthesise — it only collects. "
        "Use as a sub-step inside a learning workflow. Returns collected notes and file paths."
    ),
)
class ResearchSkill(BaseSkill):
    name = "research"
    description = (
        "Source raw material about a topic from web search, full-page fetches, PDFs, "
        "and Rednote social media posts. Does NOT analyse or synthesise — it only collects."
    )

    async def run(self, goal: str, context: SkillContext) -> SkillResult:
        run_id = context.run_id
        logger.info("[ResearchSkill] run_id=%s goal=%r", run_id, goal[:80])
        await context.send_update(f"🔍 Researching: {goal[:80]}…")
        await context.checkpoint("research")

        # ── Step 1: Web search (DDG snippets) ────────────────────────────────
        search_result = await context.tool_executor.execute("web_search", {"query": goal, "max_results": 10})
        search_str = str(search_result)
        if search_result and not search_str.startswith("Error"):
            context.notes.append(f"[Web search snippets for '{goal}']\n{search_result}")
            logger.info("[ResearchSkill] run_id=%s web_search OK: %d chars", run_id, len(search_str))
        else:
            logger.warning("[ResearchSkill] run_id=%s web_search failed: %.200s", run_id, search_str)

        # Extract URLs from search results to fetch full pages
        urls = _extract_urls(search_str)
        logger.info("[ResearchSkill] run_id=%s extracted %d URLs from search results", run_id, len(urls))

        # ── Step 2: Fetch full pages / download PDFs ──────────────────────────
        pages_fetched = 0
        for url in urls:
            if url in context.sources_done:
                continue
            if pages_fetched >= _MAX_PAGES_PER_SEARCH:
                break

            context.sources_done.append(url)

            if url.lower().endswith(".pdf"):
                await context.send_update(f"📄 Downloading PDF: {url[:60]}…")
                local_path = await context.tool_executor.execute("download_file", {"url": url})
                if local_path and not str(local_path).startswith("Error"):
                    md_path = await context.tool_executor.execute("read_pdf", {"path": str(local_path)})
                    if md_path and not str(md_path).startswith("Error"):
                        content = await context.tool_executor.execute("read_file", {"path": str(md_path)})
                        if content and not str(content).startswith("Error"):
                            context.artifacts.append(str(md_path))
                            context.notes.append(f"[PDF: {url}]\n{str(content)[:8000]}")
                            pages_fetched += 1
                            logger.info("[ResearchSkill] run_id=%s PDF ok: %s (%d chars)", run_id, url[:60], len(str(content)))
                        else:
                            logger.warning("[ResearchSkill] run_id=%s read_file failed for PDF md: %.100s", run_id, str(content))
                    else:
                        logger.warning("[ResearchSkill] run_id=%s read_pdf failed: %.100s", run_id, str(md_path))
                else:
                    logger.warning("[ResearchSkill] run_id=%s download_file failed for %s: %.100s", run_id, url[:60], str(local_path))
            else:
                await context.send_update(f"🌐 Reading page: {url[:60]}…")
                md_path = await context.tool_executor.execute("fetch_webpage", {"url": url})
                if md_path and not str(md_path).startswith("Error"):
                    content = await context.tool_executor.execute("read_file", {"path": str(md_path)})
                    if content and not str(content).startswith("Error"):
                        context.artifacts.append(str(md_path))
                        context.notes.append(f"[Web page: {url}]\n{str(content)[:4000]}")
                        pages_fetched += 1
                        logger.info("[ResearchSkill] run_id=%s page ok: %s (%d chars)", run_id, url[:60], len(str(content)))
                    else:
                        logger.warning("[ResearchSkill] run_id=%s read_file failed for page md: %.100s", run_id, str(content))
                else:
                    logger.warning("[ResearchSkill] run_id=%s fetch_webpage failed for %s: %.100s", run_id, url[:60], str(md_path))

        await context.checkpoint("research_pages_done")
        logger.info("[ResearchSkill] run_id=%s pages phase done: %d pages fetched", run_id, pages_fetched)

        # ── Step 3: Rednote social posts ─────────────────────────────────────
        if _SOCIAL_ENABLED:
            await context.send_update(
                f"📱 Searching Rednote for: {goal[:60]}…\n"
                "(If a browser window opens asking you to scan a QR code, "
                "please do so — I'll wait up to 3 minutes for you to log in.)"
            )
            # The tool itself keeps the browser open and polls for login.
            # No retry needed here — one call handles the full login → search flow.
            rednote_result = await context.tool_executor.execute(
                "social_rednote", {"query": goal, "max_results": 20, "top_k": 5}
            )
            rednote_str = str(rednote_result)

            if rednote_str and '"error"' not in rednote_str and not rednote_str.startswith("Error"):
                try:
                    posts = json.loads(rednote_str)
                    if isinstance(posts, list):
                        for post in posts:
                            text = _format_social_post(post)
                            context.notes.append(text)
                        logger.info("[ResearchSkill] run_id=%s Rednote: %d posts added", run_id, len(posts))
                    else:
                        logger.warning("[ResearchSkill] run_id=%s Rednote: unexpected response type %s", run_id, type(posts).__name__)
                except json.JSONDecodeError as exc:
                    logger.warning("[ResearchSkill] run_id=%s Rednote: JSON parse error: %s | raw=%.200s", run_id, exc, rednote_str)
            elif '"error"' in rednote_str or rednote_str.startswith("Error"):
                logger.warning("[ResearchSkill] run_id=%s Rednote tool error: %.200s", run_id, rednote_str)

        await context.checkpoint("research_social_done")

        return SkillResult(
            summary=f"Research complete: {pages_fetched} pages + social posts collected.",
            stored_points=0,
            artifacts=list(context.artifacts),
            open_questions=[],
        )


def _extract_urls(search_text: str) -> list[str]:
    """Extract URLs from web_search result text."""
    import re
    urls = re.findall(r"https?://[^\s\)\"'<>]+", search_text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        u = u.rstrip(".,;")
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _format_social_post(post: dict) -> str:
    """Format a SocialPost dict as a readable text block for notes."""
    title = post.get("title", "")
    body = post.get("body", "")
    author = post.get("author", "")
    url = post.get("url", "")
    likes = post.get("likes", 0)
    collects = post.get("collects", 0)
    comments_count = post.get("comments_count", 0)
    comments = post.get("comments", [])

    lines = [
        f"[Rednote post — engagement: {likes}❤️ {collects}⭐ {comments_count}💬]",
        f"Title: {title}",
        f"Author: {author}  URL: {url}",
        f"Content: {body}",
    ]
    if comments:
        lines.append(f"Comments ({len(comments)}):")
        for c in comments[:50]:  # cap to prevent token overflow
            lines.append(f"  • {c[:200]}")
    return "\n".join(lines)
