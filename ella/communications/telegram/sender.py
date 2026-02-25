"""Telegram Bot API sender helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from ella.config import get_settings

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org/bot{token}/{method}"
_FILE_BASE = "https://api.telegram.org/file/bot{token}/{file_path}"


class TelegramSender:
    def __init__(self, token: str) -> None:
        self._token = token
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared async client, creating it lazily on first use.

        Lazy creation ensures the client is bound to whatever event loop is
        running at call time — important in Celery forked workers where the
        parent's loop is gone and execute_task creates a fresh one.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    def _url(self, method: str) -> str:
        return _BASE.format(token=self._token, method=method)

    async def _post(self, method: str, **kwargs: Any) -> dict[str, Any]:
        resp = await self._get_client().post(self._url(method), **kwargs)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data.get('description')}")
        return data["result"]

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "HTML",
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self._post("sendMessage", json=payload)

    async def send_voice(
        self,
        chat_id: int,
        voice_path: str | Path,
        caption: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        path = Path(voice_path)
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)
        # Detect MIME type from extension; Telegram expects audio/ogg for voice messages
        suffix = path.suffix.lower()
        mime = "audio/ogg" if suffix == ".ogg" else "audio/wav"
        with path.open("rb") as f:
            files = {"voice": (path.name, f, mime)}
            return await self._post("sendVoice", data=data, files=files)

    async def send_photo(
        self,
        chat_id: int,
        photo_path: str | Path,
        caption: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        path = Path(photo_path)
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)
        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        with path.open("rb") as f:
            files = {"photo": (path.name, f, mime)}
            return await self._post("sendPhoto", data=data, files=files)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self._post("sendChatAction", json={"chat_id": chat_id, "action": action})

    async def get_file(self, file_id: str) -> dict[str, Any]:
        return await self._post("getFile", json={"file_id": file_id})

    async def download_file(self, file_path: str, dest: Path) -> None:
        url = _FILE_BASE.format(token=self._token, file_path=file_path)
        async with self._get_client().stream("GET", url) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

    async def download_file_id(self, file_id: str, dest: Path) -> Path:
        file_info = await self.get_file(file_id)
        file_path = file_info["file_path"]
        await self.download_file(file_path, dest)
        return dest

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_sender: TelegramSender | None = None


def get_sender() -> TelegramSender:
    global _sender
    if _sender is None:
        _sender = TelegramSender(get_settings().telegram_bot_token)
    return _sender
