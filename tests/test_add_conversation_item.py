from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
)
from livekit.agents.llm.chat_context import ChatMessage

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_session import FakeActions
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class SimpleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def _make_session() -> AgentSession:
    """Create a minimal session without TranscriptSynchronizer wrapping."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "placeholder")
    actions.add_llm("ok", ttft=0.1, duration=0.1)
    actions.add_tts(0.5, ttfb=0.1, duration=0.1)

    user_speeches = actions.get_user_speeches(speed_factor=1.0)
    llm_responses = actions.get_llm_responses(speed_factor=1.0)
    tts_responses = actions.get_tts_responses(speed_factor=1.0)

    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.5,
            min_speech_duration=0.05,
        ),
        stt=FakeSTT(fake_user_speeches=user_speeches),
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        turn_handling={"turn_detection": None},
        aec_warmup_duration=None,
    )

    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()

    return session


async def _close(session: AgentSession) -> None:
    """Drain and close session, letting background tasks finish."""
    await asyncio.sleep(5)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()


async def test_add_conversation_item_appears_in_history() -> None:
    """A message added via add_conversation_item is retrievable from session.history."""
    session = _make_session()
    await session.start(SimpleAgent())

    msg = ChatMessage(role="user", content=["injected text"])
    result = session.add_conversation_item(msg)

    assert result is True
    found = session.history.get_by_id(msg.id)
    assert found is not None
    assert found.text_content == "injected text"
    assert found.role == "user"

    await _close(session)


async def test_add_conversation_item_emits_event() -> None:
    """add_conversation_item fires the conversation_item_added event."""
    session = _make_session()

    received: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", received.append)

    await session.start(SimpleAgent())

    msg = ChatMessage(role="user", content=["event test"])
    session.add_conversation_item(msg)

    matching = [e for e in received if e.item.id == msg.id]
    assert len(matching) == 1
    assert matching[0].item.role == "user"
    assert matching[0].item.text_content == "event test"

    await _close(session)


async def test_add_conversation_item_visible_to_agent() -> None:
    """A message added via add_conversation_item appears in the active agent's chat context."""
    session = _make_session()
    agent = SimpleAgent()
    await session.start(agent)

    msg = ChatMessage(role="user", content=["agent-visible"])
    session.add_conversation_item(msg)

    found = agent.chat_ctx.get_by_id(msg.id)
    assert found is not None
    assert found.text_content == "agent-visible"

    await _close(session)


async def test_add_conversation_item_dedup_by_id() -> None:
    """Adding the same message ID twice is idempotent — only one copy in history."""
    session = _make_session()
    await session.start(SimpleAgent())

    msg = ChatMessage(role="user", content=["dedup test"])
    first = session.add_conversation_item(msg)
    second = session.add_conversation_item(msg)

    assert first is True
    assert second is False

    matches = [item for item in session.history.items if item.id == msg.id]
    assert len(matches) == 1

    await _close(session)


async def test_add_conversation_item_before_start() -> None:
    """add_conversation_item works on a session that hasn't been started yet."""
    session = _make_session()

    received: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", received.append)

    msg = ChatMessage(role="user", content=["pre-start"])
    result = session.add_conversation_item(msg)

    assert result is True
    assert session.history.get_by_id(msg.id) is not None

    matching = [e for e in received if e.item.id == msg.id]
    assert len(matching) == 1
