"""Tool: download_file — fetch a remote file to ~/Ella/downloads/."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

from ella.tools.registry import ella_tool

logger = logging.getLogger(__name__)

_DOWNLOADS_DIR = Path.home() / "Ella" / "downloads"
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB safety cap


@ella_tool(
    name="download_file",
    description=(
        "Download a file from a URL to ~/Ella/downloads/ and return the local file path. "
        "Use when you have a direct link to a PDF, Markdown, image, or other document "
        "you want to read. Do NOT use for web pages — use fetch_webpage for that. "
        "Do NOT use for social media — use social_rednote or similar tools for that."
    ),
)
async def download_file(url: str) -> str:
    """Download a remote file and return its local path.

    url: the full URL of the file to download
    """
    _DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Derive a stable filename: use the URL's last path segment, or a hash
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    raw_name = unquote(parsed.path.rstrip("/").split("/")[-1]) or hashlib.md5(url.encode()).hexdigest()
    # Sanitise — keep only safe characters
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
    dest = _DOWNLOADS_DIR / safe_name

    if dest.exists():
        logger.info("[download_file] Cache hit: %s", dest)
        return str(dest)

    logger.info("[download_file] Downloading %s → %s", url, dest)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total = 0
                with dest.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > _MAX_BYTES:
                            dest.unlink(missing_ok=True)
                            return f"Error: file at {url} exceeds 50 MB size limit."
                        f.write(chunk)
        logger.info("[download_file] Saved %d bytes to %s", total, dest)
        return str(dest)
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} when downloading {url}"
    except Exception as e:
        dest.unlink(missing_ok=True)
        return f"Error downloading {url}: {e}"
