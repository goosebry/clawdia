"""Tests for ella/communications/telegram/models.py — no network needed."""
import pytest
from ella.communications.telegram.models import TelegramMessage, TelegramUpdate, TelegramChat, TelegramUser


def _make_text_update(update_id=1, chat_id=100, text="Hello"):
    return {
        "update_id": update_id,
        "message": {
            "message_id": 42,
            "date": 1700000000,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 99, "is_bot": False, "first_name": "Test"},
            "text": text,
        }
    }


def _make_voice_update(update_id=2, chat_id=100):
    return {
        "update_id": update_id,
        "message": {
            "message_id": 43,
            "date": 1700000001,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 99, "is_bot": False, "first_name": "Test"},
            "voice": {
                "file_id": "voice_file_123",
                "file_unique_id": "uq_voice",
                "duration": 5,
            },
        }
    }


def _make_video_update(update_id=3, chat_id=100):
    return {
        "update_id": update_id,
        "message": {
            "message_id": 44,
            "date": 1700000002,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 99, "is_bot": False, "first_name": "Test"},
            "video": {
                "file_id": "video_file_456",
                "file_unique_id": "uq_video",
                "width": 1280,
                "height": 720,
                "duration": 10,
            },
        }
    }


def test_parse_text_message():
    raw = _make_text_update(text="Hello, 世界!")
    msg = TelegramMessage.from_raw(raw["message"])
    assert msg.text == "Hello, 世界!"
    assert msg.message_id == 42
    assert msg.chat.id == 100
    assert msg.media_type == "text"


def test_parse_voice_message():
    raw = _make_voice_update()
    msg = TelegramMessage.from_raw(raw["message"])
    assert msg.voice is not None
    assert msg.voice.file_id == "voice_file_123"
    assert msg.media_type == "voice"


def test_parse_video_message():
    raw = _make_video_update()
    msg = TelegramMessage.from_raw(raw["message"])
    assert msg.video is not None
    assert msg.video.file_id == "video_file_456"
    assert msg.media_type == "video"


def test_from_field_remapped():
    """'from' → 'from_' remapping works correctly."""
    raw = _make_text_update()
    msg = TelegramMessage.from_raw(raw["message"])
    assert msg.from_ is not None
    assert msg.from_.first_name == "Test"
    assert msg.from_.id == 99


def test_no_media_type():
    raw = {
        "message_id": 1,
        "date": 1700000000,
        "chat": {"id": 1, "type": "private"},
    }
    msg = TelegramMessage.from_raw(raw)
    assert msg.media_type is None


def test_update_parse():
    raw = _make_text_update(update_id=55)
    update = TelegramUpdate.from_raw(raw)
    assert update.update_id == 55
    assert update.message is not None
    assert update.message.text == "Hello"


def test_update_no_message():
    raw = {"update_id": 1}
    update = TelegramUpdate.from_raw(raw)
    assert update.update_id == 1
    assert update.message is None
