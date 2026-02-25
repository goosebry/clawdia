"""Ingestion handler for plain text messages."""
from __future__ import annotations

import re


_TELEGRAM_FORMATTING = re.compile(r"<[^>]+>")


def process_text(text: str) -> str:
    """Strip Telegram HTML formatting tags and normalise whitespace."""
    cleaned = _TELEGRAM_FORMATTING.sub("", text)
    return cleaned.strip()
