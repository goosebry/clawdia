"""Tests for ReplyAgent — TTS and Telegram sender are stubbed."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ella.agents.protocol import HandoffMessage, LLMMessage, ReplyPayload, SessionContext


def _make_session(chat_id: int = 100) -> SessionContext:
    session = SessionContext(chat_id=chat_id)
    session.focus = [
        LLMMessage(role="user", content="Hello, how are you?"),
        LLMMessage(role="assistant", content="I'm doing great!"),
    ]
    return session


def test_reply_agent_sends_voice_when_tts_succeeds(tmp_path):
    from ella.agents.reply_agent import ReplyAgent

    # Create a real temp WAV file so the path-exists check passes
    wav_path = str(tmp_path / "reply.wav")
    open(wav_path, "wb").close()

    session = _make_session()
    payload = ReplyPayload(text="I'm doing great!", language="en")
    handoff = HandoffMessage(payload=payload, session=session)

    with (
        patch("ella.agents.reply_agent.tts_to_wav", return_value=wav_path),
        patch("ella.agents.reply_agent.get_sender") as mock_sender_fn,
    ):
        sender = MagicMock()
        sender.send_voice = AsyncMock()
        sender.send_message = AsyncMock()
        mock_sender_fn.return_value = sender
        session.knowledge = None

        agent = ReplyAgent()
        asyncio.run(agent.handle(handoff))

    sender.send_voice.assert_called_once()
    call_kwargs = sender.send_voice.call_args
    assert call_kwargs.kwargs["chat_id"] == 100 or call_kwargs.args[0] == 100


def test_reply_agent_falls_back_to_text_when_tts_fails():
    from ella.agents.reply_agent import ReplyAgent

    session = _make_session()
    payload = ReplyPayload(text="Hello!", language="en")
    handoff = HandoffMessage(payload=payload, session=session)

    with (
        patch("ella.agents.reply_agent.tts_to_wav", return_value=None),
        patch("ella.agents.reply_agent.get_sender") as mock_sender_fn,
    ):
        sender = MagicMock()
        sender.send_voice = AsyncMock()
        sender.send_message = AsyncMock()
        mock_sender_fn.return_value = sender
        session.knowledge = None

        agent = ReplyAgent()
        asyncio.run(agent.handle(handoff))

    sender.send_voice.assert_not_called()
    sender.send_message.assert_called_once()


def test_reply_agent_stores_to_knowledge():
    from ella.agents.reply_agent import ReplyAgent

    session = _make_session()
    payload = ReplyPayload(text="Reply text", language="zh")
    handoff = HandoffMessage(payload=payload, session=session)

    knowledge_mock = MagicMock()
    knowledge_mock.store_exchange = AsyncMock()
    session.knowledge = knowledge_mock

    with (
        patch("ella.agents.reply_agent.tts_to_wav", return_value=None),
        patch("ella.agents.reply_agent.get_sender") as mock_sender_fn,
    ):
        sender = MagicMock()
        sender.send_message = AsyncMock()
        mock_sender_fn.return_value = sender

        agent = ReplyAgent()
        asyncio.run(agent.handle(handoff))

    knowledge_mock.store_exchange.assert_called_once_with(
        chat_id=100,
        user_text="Hello, how are you?",
        assistant_text="Reply text",
    )


def test_reply_agent_ignores_non_handoff():
    from ella.agents.reply_agent import ReplyAgent
    from ella.agents.protocol import UserTask

    agent = ReplyAgent()
    session = SessionContext(chat_id=1)
    task = UserTask(raw_updates=[], session=session)
    result = asyncio.run(agent.handle(task))
    assert result == []


def test_reply_agent_ignores_wrong_payload():
    from ella.agents.reply_agent import ReplyAgent

    agent = ReplyAgent()
    session = SessionContext(chat_id=1)
    handoff = HandoffMessage(payload="not a ReplyPayload", session=session)
    result = asyncio.run(agent.handle(handoff))
    assert result == []
