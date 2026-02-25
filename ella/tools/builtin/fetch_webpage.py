"""Tool: fetch_webpage — fetch a web page and cache it as a Markdown file."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import httpx

from ella.config import get_settings
from ella.tools.registry import ella_tool

logger = logging.getLogger(__name__)

_DOWNLOADS_DIR = Path.home() / "Ella" / "downloads"


@ella_tool(
    name="fetch_webpage",
    description=(
        "Fetch a web page URL, extract its main readable content, convert it to a "
        "Markdown file saved to ~/Ella/downloads/{url-hash}.md, and return the file path. "
        "Use for articles, documentation, and blog posts. "
        "Then use read_file on the returned path to read the content. "
        "Do NOT use for PDFs — use download_file + read_pdf for those. "
        "Do NOT use for social media — use social_rednote or similar tools for those. "
        "The result is cached by URL: if the cached file is fresh it is returned immediately."
    ),
)
async def fetch_webpage(url: str) -> str:
    """Fetch a webpage, strip it to clean Markdown, cache to disk, return path.

    url: the full URL of the webpage to fetch
    """
    _DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    md_path = _DOWNLOADS_DIR / f"web_{url_hash}.md"

    # Return cached file if fresh enough
    settings = get_settings()
    cache_hours = getattr(settings, "fetch_cache_hours", 24)
    if md_path.exists():
        age_hours = (time.time() - md_path.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            logger.info("[fetch_webpage] Cache hit (%.1fh old): %s", age_hours, url)
            return str(md_path)

    logger.info("[fetch_webpage] Fetching %s", url)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Ella/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # Extract main content with trafilatura (best-in-class readability)
    md_content: str | None = None
    try:
        import trafilatura  # type: ignore  # pip install trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
        )
        if extracted and extracted.strip():
            md_content = extracted
    except ImportError:
        logger.warning("[fetch_webpage] trafilatura not installed — falling back to markdownify")
    except Exception as e:
        logger.warning("[fetch_webpage] trafilatura failed: %s — falling back", e)

    # Fallback: markdownify HTML→MD
    if not md_content:
        try:
            import markdownify  # type: ignore  # pip install markdownify
            md_content = markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])
        except ImportError:
            # Last resort: strip all tags
            import re
            md_content = re.sub(r"<[^>]+>", " ", html)
            md_content = re.sub(r"\s{2,}", " ", md_content).strip()

    if not md_content or not md_content.strip():
        return f"Error: could not extract readable content from {url}"

    # Prepend source metadata
    header = f"# {url}\n\n_Source: {url}_\n\n---\n\n"
    full_content = header + md_content.strip()

    md_path.write_text(full_content, encoding="utf-8")
    logger.info("[fetch_webpage] Cached %d chars to %s", len(full_content), md_path.name)
    return str(md_path)
