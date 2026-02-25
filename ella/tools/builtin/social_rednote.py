"""Tool: social_rednote — search Rednote (小红书) and return top posts with comments.

Uses DrissionPage + Chromium (not Playwright) — required for Rednote's
anti-bot fingerprinting. Login state is persisted as a browser profile so
the QR-code scan is a one-time setup.

Browser profile location: ~/Ella/browser_profiles/rednote/

Login flow
----------
First call (not logged in):
  - Opens the browser and navigates to the Rednote login page.
  - Waits up to LOGIN_TIMEOUT_S for the user to scan the QR code and log in.
  - Once login is detected, proceeds immediately to search.
  - The session cookie is saved in the browser profile — subsequent calls
    skip the login step entirely.

If LOGIN_TIMEOUT_S is exceeded without a successful login, the browser is
closed and an error is returned.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from ella.tools.registry import ella_tool
from ella.tools.social_base import SocialPost

logger = logging.getLogger(__name__)

_PROFILE_DIR = Path.home() / "Ella" / "browser_profiles" / "rednote"
_LOGIN_URL = "https://www.xiaohongshu.com/login"
_SEARCH_URL = "https://www.xiaohongshu.com/search_result?keyword={query}&source=web_explore_feed"

# How long to wait for the user to complete QR-code login before giving up.
LOGIN_TIMEOUT_S = 180  # 3 minutes
# How often to poll the page to check if login completed.
LOGIN_POLL_S = 3


@ella_tool(
    name="social_rednote",
    description=(
        "Search Rednote (小红书 / XiaoHongShu) for posts about a topic, filter to the "
        "most popular and credible ones by engagement score (likes + collects + comments + shares), "
        "and return the full text of each post plus all its comments. "
        "Use when you need community knowledge, lived experience, product reviews, or "
        "Chinese-language discussion about a topic. "
        "Do NOT use for general web content — use fetch_webpage for that. "
        "Login state is persisted in ~/Ella/browser_profiles/rednote/ — first use requires "
        "a one-time QR-code scan in the browser window that opens automatically. "
        "The tool will wait up to 3 minutes for login before proceeding to search."
    ),
)
async def social_rednote(
    query: str,
    max_results: int = 20,
    top_k: int = 5,
) -> str:
    """Search Rednote, return top-K posts sorted by engagement, with all comments.

    query: search keywords (Chinese or English)
    max_results: number of posts to fetch before filtering (default 20)
    top_k: number of top posts to return after filtering by engagement score (default 5)

    Returns a JSON array of posts, or a JSON error object.
    If login is needed the browser opens automatically — this tool waits for
    the user to scan the QR code (up to 3 minutes) before searching.
    """
    try:
        from DrissionPage import ChromiumPage, ChromiumOptions  # type: ignore
    except ImportError:
        return json.dumps({
            "error": "DrissionPage is not installed. Run: pip install DrissionPage",
        })

    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    options = ChromiumOptions()
    options.set_argument("--no-sandbox")
    options.set_argument("--disable-dev-shm-usage")
    options.set_user_data_path(str(_PROFILE_DIR))

    try:
        page = ChromiumPage(addr_or_opts=options)
    except Exception as e:
        return json.dumps({"error": f"Failed to launch browser: {e}"})

    try:
        # Navigate to home to load any saved session cookie
        page.get("https://www.xiaohongshu.com")
        await asyncio.sleep(3)

        if _needs_login(page):
            logger.info("[social_rednote] Not logged in — opening login page and waiting for QR scan")
            page.get(_LOGIN_URL)
            await asyncio.sleep(2)

            # Wait for the user to complete login, polling every LOGIN_POLL_S
            elapsed = 0
            logged_in = False
            while elapsed < LOGIN_TIMEOUT_S:
                await asyncio.sleep(LOGIN_POLL_S)
                elapsed += LOGIN_POLL_S
                # After a successful QR scan, Rednote redirects away from /login
                current_url = ""
                try:
                    current_url = page.url  # type: ignore[attr-defined]
                except Exception:
                    pass
                if "/login" not in current_url and not _needs_login(page):
                    logged_in = True
                    logger.info(
                        "[social_rednote] Login detected after %ds — proceeding to search", elapsed
                    )
                    break
                if elapsed % 30 == 0:
                    logger.info(
                        "[social_rednote] Still waiting for login (%ds / %ds)…",
                        elapsed, LOGIN_TIMEOUT_S,
                    )

            if not logged_in:
                page.quit()
                return json.dumps({
                    "error": (
                        f"Rednote login not completed within {LOGIN_TIMEOUT_S}s. "
                        "Please scan the QR code in the browser window and try again."
                    ),
                })

        # ── Logged in — proceed to search ────────────────────────────────────
        search_url = _SEARCH_URL.format(query=query)
        logger.info("[social_rednote] Searching: %s", query)
        page.get(search_url)
        await asyncio.sleep(3)

        posts_raw = await _collect_search_results(page, max_results)
        if not posts_raw:
            page.quit()
            return json.dumps({"error": "No posts found for query: " + query})

        # Sort by engagement, keep top_k
        posts_raw.sort(key=lambda p: p.engagement_score, reverse=True)
        top_posts = posts_raw[:top_k]
        logger.info(
            "[social_rednote] Fetched %d posts, keeping top %d by engagement",
            len(posts_raw), len(top_posts),
        )

        # Open each top post and collect all comments
        for post in top_posts:
            if post.url:
                post.comments = await _collect_comments(page, post.url)
                logger.info(
                    "[social_rednote] Post '%s': %d comments",
                    post.title[:40], len(post.comments),
                )

        page.quit()
        return json.dumps([asdict(p) for p in top_posts], ensure_ascii=False)

    except Exception as e:
        logger.exception("[social_rednote] Error during search")
        try:
            page.quit()
        except Exception:
            pass
        return json.dumps({"error": f"Search failed: {e}"})


def _needs_login(page: object) -> bool:
    """Return True if the current page is actively blocking with a login wall.

    Only fires when:
      - The URL is explicitly /login, OR
      - The page contains login-modal-specific markers (not generic QR codes
        that appear in nav/footer on every Rednote page).
    """
    try:
        url: str = page.url  # type: ignore[attr-defined]
        if "/login" in url:
            return True
        html: str = page.html  # type: ignore[attr-defined]
        # These strings only appear inside the actual login modal/container —
        # not on explore/search pages where the user is already authenticated.
        # Deliberately exclude "qrcode" and "qr-code" which appear site-wide.
        login_modal_indicators = [
            "login-container",
            "loginContainer",
            "sign-container",
            "请登录后",          # "Please log in to..."
            "登录后查看",         # "Log in to view"
            "登录/注册",          # Login/Register button text in modal
        ]
        return any(ind in html for ind in login_modal_indicators)
    except Exception:
        return False


async def _collect_search_results(page: object, max_results: int) -> list[SocialPost]:
    """Scroll the search results page and extract post metadata."""
    posts: list[SocialPost] = []
    seen_ids: set[str] = set()

    for _ in range(max(1, max_results // 10)):
        try:
            items = page.eles("css:.note-item")  # type: ignore[attr-defined]
            if not items:
                items = page.eles("css:[class*='note']")  # type: ignore[attr-defined]
        except Exception:
            break

        for item in items:
            if len(posts) >= max_results:
                break
            try:
                post = _parse_search_item(item)
                if post and post.post_id not in seen_ids:
                    seen_ids.add(post.post_id)
                    posts.append(post)
            except Exception:
                continue

        if len(posts) >= max_results:
            break

        try:
            page.scroll.down(800)  # type: ignore[attr-defined]
            await asyncio.sleep(1.5)
        except Exception:
            break

    return posts


def _parse_search_item(item: object) -> SocialPost | None:
    """Extract metadata from a single search result card."""
    try:
        link = item.ele("css:a", timeout=1)  # type: ignore[attr-defined]
        href = link.attr("href") if link else ""
        post_id = href.split("/")[-1].split("?")[0] if href else ""
        url = f"https://www.xiaohongshu.com{href}" if href and href.startswith("/") else href

        title_el = (
            item.ele("css:.title", timeout=1)
            or item.ele("css:[class*='title']", timeout=1)  # type: ignore[attr-defined]
        )
        title = title_el.text.strip() if title_el else ""

        author_el = (
            item.ele("css:.author", timeout=1)
            or item.ele("css:[class*='author']", timeout=1)  # type: ignore[attr-defined]
        )
        author = author_el.text.strip() if author_el else ""

        likes = _parse_count(item, ["css:.like-count", "css:[class*='like']"])
        collects = _parse_count(item, ["css:.collect-count", "css:[class*='collect']"])
        comments_count = _parse_count(item, ["css:.comment-count", "css:[class*='comment']"])

        if not post_id or not url:
            return None

        return SocialPost(
            platform="rednote",
            post_id=post_id,
            url=url,
            title=title,
            body="",
            author=author,
            published_at="",
            likes=likes,
            collects=collects,
            comments_count=comments_count,
            shares=0,
        )
    except Exception:
        return None


def _parse_count(item: object, selectors: list[str]) -> int:
    """Try multiple CSS selectors to extract a numeric engagement count."""
    import re
    for selector in selectors:
        try:
            el = item.ele(selector, timeout=0.5)  # type: ignore[attr-defined]
            if el:
                text = el.text.strip().replace(",", "").replace("k", "000").replace("K", "000")
                m = re.search(r"\d+", text)
                if m:
                    return int(m.group())
        except Exception:
            continue
    return 0


async def _collect_comments(page: object, post_url: str) -> list[str]:
    """Open a post page and collect all comments by scrolling."""
    comments: list[str] = []

    try:
        page.get(post_url)  # type: ignore[attr-defined]
        await asyncio.sleep(3)

        for _ in range(20):  # max 20 scroll rounds per post
            try:
                comment_els = page.eles("css:.comment-item")  # type: ignore[attr-defined]
                if not comment_els:
                    comment_els = page.eles("css:[class*='comment-item']")  # type: ignore[attr-defined]
            except Exception:
                break

            for el in comment_els:
                try:
                    text = el.text.strip()
                    if text and text not in comments:
                        comments.append(text)
                except Exception:
                    continue

            try:
                prev_count = len(comments)
                page.scroll.down(600)  # type: ignore[attr-defined]
                await asyncio.sleep(1.5)
                if len(comments) == prev_count:
                    break  # no new comments loaded
            except Exception:
                break

    except Exception as e:
        logger.warning("[social_rednote] Failed to collect comments from %s: %s", post_url, e)

    return comments
