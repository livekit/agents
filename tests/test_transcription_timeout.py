from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, UserTranscriptionTimeoutEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0
TIMEOUT = 2.0


def _agent() -> Agent:
    return Agent(instructions="You are a helpful assistant.")


async def test_fires_when_vad_speech_not_transcribed() -> None:
    # empty transcript => VAD SOS/EOS fire but STT emits nothing
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    t_origin = await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)  # 2.5 - 0.5
    assert events[0].vad_speech_started_at - t_origin == pytest.approx(0.5, abs=0.5)


async def test_no_event_when_transcribed() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?")
    actions.add_llm("I'm doing well, thank you!")
    actions.add_tts(2.0)

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert events == []


async def test_interim_only_still_fires() -> None:
    # an interim transcript with no final => STT may have dropped it, so the timeout
    # must still fire (interim transcripts don't count as confirmation)
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", final=False)

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)


async def test_disabled() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    session = create_session(actions, extra_kwargs={"transcription_timeout": None})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert events == []


async def test_accumulates_across_bursts() -> None:
    # two bursts within the timeout window => a single event, durations summed
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "")
    actions.add_user_speech(2.0, 3.0, "")

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)  # 1.0 + 1.0


async def test_refires_on_next_attempt() -> None:
    # two bursts separated by more than the timeout => one event each; the uncommitted
    # turn is not reset, so speech_duration accumulates and the start stays pinned
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "")
    actions.add_user_speech(5.0, 6.0, "")

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 2
    assert events[0].speech_duration == pytest.approx(1.0, abs=0.5)
    assert events[1].speech_duration == pytest.approx(2.0, abs=0.5)
    # same still-open turn => same start anchor
    assert events[1].vad_speech_started_at == pytest.approx(events[0].vad_speech_started_at)
