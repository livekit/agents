from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, UserTranscriptionTimeoutEvent

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

TIMEOUT = 2.0
ON = {"transcription_timeout": TIMEOUT}


async def _timeouts(
    actions: FakeActions, *, drain_delay: float = 5.0, **kwargs
) -> tuple[list[UserTranscriptionTimeoutEvent], float]:
    session = create_session(actions, **kwargs)
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)
    agent = Agent(instructions="You are a helpful assistant.")
    t_origin = await asyncio.wait_for(run_session(session, agent, drain_delay=drain_delay), 60)
    return events, t_origin


async def test_fires_when_vad_speech_not_transcribed() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    events, t_origin = await _timeouts(actions, extra_kwargs=ON)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)
    assert events[0].vad_speech_started_at - t_origin == pytest.approx(0.5, abs=0.5)
    assert events[0].created_at - t_origin == pytest.approx(2.5 + TIMEOUT, abs=0.1)


async def test_no_event_when_transcribed() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?")
    actions.add_llm("I'm doing well, thank you!")
    actions.add_tts(2.0)

    events, _ = await _timeouts(actions, extra_kwargs=ON)

    assert events == []


async def test_interim_only_still_fires() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", final=False)

    events, _ = await _timeouts(actions, extra_kwargs=ON)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)


async def test_disabled_by_default() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    events, _ = await _timeouts(actions)

    assert events == []


async def test_fires_while_agent_is_speaking() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story.")
    actions.add_tts(10.0)
    actions.add_user_speech(5.0, 6.0, "")

    events, _ = await _timeouts(
        actions, extra_kwargs=ON, turn_handling={"interruption": {"enabled": False}}, drain_delay=10
    )

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(1.0, abs=0.5)


async def test_no_event_without_stt() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    events, _ = await _timeouts(
        actions,
        extra_kwargs=ON,
        with_stt=False,
        turn_handling={"turn_detection": "manual"},
        drain_delay=10,
    )

    assert events == []


async def test_accumulates_across_bursts() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "")
    actions.add_user_speech(2.0, 3.0, "")

    events, _ = await _timeouts(actions, extra_kwargs=ON)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)


async def test_refires_on_next_attempt() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "")
    actions.add_user_speech(5.0, 6.0, "")

    events, _ = await _timeouts(actions, extra_kwargs=ON)

    assert len(events) == 2
    assert events[0].speech_duration == pytest.approx(1.0, abs=0.5)
    assert events[1].speech_duration == pytest.approx(2.0, abs=0.5)
    # both bursts belong to the same uncommitted turn
    assert events[1].vad_speech_started_at == pytest.approx(events[0].vad_speech_started_at)
