"""Built-in tool: web search using DuckDuckGo HTML (no API key required).

Uses the DuckDuckGo HTML endpoint which returns real web results, unlike the
Instant Answer JSON API which only answers well-known factual queries.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

from ella.tools.registry import ella_tool

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}


@ella_tool(
    name="web_search",
    description=(
        "Search the web and return real results with titles, URLs, and snippets. "
        "Use when: the user explicitly asks to search or look something up; "
        "the user asks for current news, prices, or live data you cannot know; "
        "the user asks for specific links, images, or recommendations you cannot provide from memory. "
        "Do NOT use for: casual conversation topics you can discuss from general knowledge; "
        "personal questions about yourself or your relationship with the user; "
        "facts you already know confidently."
    ),
)
def web_search(query: str, max_results: int = 5) -> str:
    """query: the search query string. max_results: number of results to return (1-10)."""
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    max_results = min(max(1, max_results), 10)

    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return f"Search failed: {exc}"

    results = _parse_ddg_html(html, max_results)

    if not results:
        # Fallback: try the lite endpoint
        try:
            url_lite = f"https://lite.duckduckgo.com/lite/?q={encoded}"
            req2 = urllib.request.Request(url_lite, headers=_HEADERS)
            with urllib.request.urlopen(req2, timeout=12) as resp2:
                html2 = resp2.read().decode("utf-8", errors="ignore")
            results = _parse_ddg_lite(html2, max_results)
        except Exception:
            pass

    if not results:
        return "No search results found for this query."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n\n".join(lines)


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML search results page."""
    results: list[dict[str, str]] = []

    # Result blocks are <div class="result ..."> containing title, url, snippet
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    for url, title_html, snippet_html in blocks:
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        # DDG redirects — extract actual URL from uddg param
        url_match = re.search(r"uddg=([^&]+)", url)
        if url_match:
            url = urllib.parse.unquote(url_match.group(1))
        if title and snippet:
            results.append({"title": title[:120], "url": url, "snippet": snippet[:300]})
        if len(results) >= max_results:
            break

    return results


def _parse_ddg_lite(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo Lite results page (simpler HTML fallback)."""
    results: list[dict[str, str]] = []

    # Lite has <a class="result-link"> for titles and <td class="result-snippet"> for snippets
    title_urls = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    snippets = re.findall(r'class="result-snippet"[^>]*>(.*?)</(?:td|span)>', html, re.DOTALL)

    for i, (url, title_html) in enumerate(title_urls):
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
        url_match = re.search(r"uddg=([^&]+)", url)
        if url_match:
            url = urllib.parse.unquote(url_match.group(1))
        if title:
            results.append({"title": title[:120], "url": url, "snippet": snippet[:300]})
        if len(results) >= max_results:
            break

    return results
