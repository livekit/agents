"""The agent transcript must survive a TTS that advertises alignment it never sends (#6493).

``capabilities.aligned_transcript`` says word timings were *requested*, not that the
model/language pair returns any, and the timed channel exists as soon as the node
streams — before a single timed word has arrived. Committing the turn's transcript to
that channel therefore left the turn with no transcript at all whenever the provider
sent none, which is what surfaced as "no agent transcript was returned from tts".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable

import pytest

from livekit.agents.types import TimedString
from livekit.agents.voice.agent_activity import _aligned_transcript_or_text
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_session import FakeActions, create_session, run_session
from .fake_tts import FakeTTS
from .test_agent_session import MyAgent

pytestmark = pytest.mark.unit


async def _stream(*items: str) -> AsyncIterable[str]:
    for item in items:
        yield item


async def _collect(source: AsyncIterable[str]) -> list[str]:
    return [chunk async for chunk in source]


async def test_aligned_transcript_is_forwarded_when_it_arrives() -> None:
    timed = _stream(
        TimedString("hello ", start_time=0.0, end_time=0.5),
        TimedString("there", start_time=0.5, end_time=1.0),
    )
    text = _stream("hello there")

    forwarded = await _collect(_aligned_transcript_or_text(timed, text))

    assert forwarded == ["hello ", "there"]
    # the timings are what the aligned path exists for, so they must survive
    assert all(isinstance(chunk, TimedString) for chunk in forwarded)
    # the spoken text was left untouched
    assert await _collect(text) == ["hello there"]


async def test_falls_back_to_the_spoken_text_when_no_timings_arrive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    timed = _stream()  # the provider advertised alignment and sent none
    text = _stream("hello ", "there")

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        forwarded = await _collect(_aligned_transcript_or_text(timed, text))

    assert forwarded == ["hello ", "there"], "the turn would otherwise have no transcript"
    assert any("no aligned transcript" in record.message for record in caplog.records)


async def test_partial_alignment_is_not_duplicated(caplog: pytest.LogCaptureFixture) -> None:
    # one timed word is proof the provider aligns; the rest is its business, and
    # replaying the spoken text on top would repeat what was already forwarded
    timed = _stream(TimedString("hello", start_time=0.0, end_time=0.5))
    text = _stream("hello there")

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        forwarded = await _collect(_aligned_transcript_or_text(timed, text))

    assert forwarded == ["hello"]
    assert not [r for r in caplog.records if "no aligned transcript" in r.message]


async def test_turn_keeps_its_transcript_when_the_tts_sends_no_timings() -> None:
    # end to end: the session's TTS claims alignment (as cartesia/elevenlabs do by
    # default) while emitting audio only, which is the shape reported in #6493 for a
    # model/language pair that cannot align
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "Hello, how are you?", stt_delay=0.1)
    actions.add_llm("I'm doing well, thank you!", ttft=0.1, duration=0.2)
    actions.add_tts(2.0, ttfb=0.2, duration=0.2)

    session = create_session(actions, extra_kwargs={"use_tts_aligned_transcript": True})
    assert isinstance(session.tts, FakeTTS)
    session.tts._capabilities.aligned_transcript = True

    conversation: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", conversation.append)

    await asyncio.wait_for(run_session(session, MyAgent()), timeout=30)

    agent_messages = [
        ev.item for ev in conversation if ev.item.type == "message" and ev.item.role == "assistant"
    ]
    assert agent_messages, "the agent turn was never recorded"
    assert agent_messages[0].text_content, (
        "the agent turn has no transcript: the aligned channel was empty and nothing fell back"
    )
