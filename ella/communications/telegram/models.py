"""Pydantic models for Telegram Bot API payloads."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class TelegramUser(BaseModel):
    id: int
    is_bot: bool = False
    first_name: str = ""
    username: str | None = None
    language_code: str | None = None


class TelegramChat(BaseModel):
    id: int
    type: str
    title: str | None = None
    username: str | None = None
    first_name: str | None = None


class TelegramAudio(BaseModel):
    file_id: str
    file_unique_id: str
    duration: int = 0
    mime_type: str | None = None
    file_size: int | None = None


class TelegramVoice(BaseModel):
    file_id: str
    file_unique_id: str
    duration: int = 0
    mime_type: str | None = None
    file_size: int | None = None


class TelegramVideo(BaseModel):
    file_id: str
    file_unique_id: str
    width: int = 0
    height: int = 0
    duration: int = 0
    mime_type: str | None = None
    file_size: int | None = None


class TelegramDocument(BaseModel):
    file_id: str
    file_unique_id: str
    file_name: str | None = None
    mime_type: str | None = None
    file_size: int | None = None


class TelegramPhotoSize(BaseModel):
    """One resolution variant of a Telegram photo."""
    file_id: str
    file_unique_id: str
    width: int = 0
    height: int = 0
    file_size: int | None = None


class TelegramMessage(BaseModel):
    message_id: int
    date: int
    chat: TelegramChat
    from_: TelegramUser | None = None
    text: str | None = None
    voice: TelegramVoice | None = None
    video: TelegramVideo | None = None
    audio: TelegramAudio | None = None
    document: TelegramDocument | None = None
    # Telegram sends photos as a list of sizes (thumbnail → full). We pick the last (largest).
    photo: list[TelegramPhotoSize] | None = None
    caption: str | None = None

    model_config = {"populate_by_name": True}

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "TelegramMessage":
        if "from" in data:
            data = dict(data)
            data["from_"] = data.pop("from")
        return cls.model_validate(data)

    @property
    def best_photo(self) -> TelegramPhotoSize | None:
        """Return the highest-resolution photo size, or None."""
        if not self.photo:
            return None
        return max(self.photo, key=lambda p: p.width * p.height)

    @property
    def media_type(self) -> str | None:
        if self.voice:
            return "voice"
        if self.video:
            return "video"
        if self.audio:
            return "audio"
        if self.photo:
            return "photo"
        if self.text:
            return "text"
        return None


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage | None = None

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "TelegramUpdate":
        update = cls.model_validate(data)
        if update.message is None and "message" in data:
            update.message = TelegramMessage.from_raw(data["message"])
        return update
