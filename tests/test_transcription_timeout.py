from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import Agent, UserTranscriptionTimeoutEvent
from livekit.agents.voice.audio_recognition import AudioRecognition

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0
TIMEOUT = 2.0


def _agent() -> Agent:
    return Agent(instructions="You are a helpful assistant.")


async def test_fires_when_vad_speech_not_transcribed() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    t_origin = await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)
    assert events[0].vad_speech_started_at - t_origin == pytest.approx(0.5, abs=0.5)
    assert events[0].created_at - t_origin == pytest.approx(2.5 + TIMEOUT, abs=0.1)


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
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Hello, how are you?", final=False)

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)


async def test_disabled_by_default() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    session = create_session(actions)
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert events == []


async def test_fires_while_agent_is_speaking() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Tell me a story.")
    actions.add_llm("Here is a long story.")
    actions.add_tts(10.0)
    actions.add_user_speech(5.0, 6.0, "")

    session = create_session(
        actions,
        extra_kwargs={"transcription_timeout": TIMEOUT},
        turn_handling={"interruption": {"enabled": False}},
    )
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent(), drain_delay=10), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(1.0, abs=0.5)


async def test_no_event_without_stt() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    session = create_session(
        actions,
        extra_kwargs={"transcription_timeout": TIMEOUT},
        with_stt=False,
        turn_handling={"turn_detection": "manual"},
    )
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent(), drain_delay=10), timeout=SESSION_TIMEOUT)

    assert events == []


async def test_accumulates_across_bursts() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "")
    actions.add_user_speech(2.0, 3.0, "")

    session = create_session(actions, extra_kwargs={"transcription_timeout": TIMEOUT})
    events: list[UserTranscriptionTimeoutEvent] = []
    session.on("user_transcription_timeout", events.append)

    await asyncio.wait_for(run_session(session, _agent()), timeout=SESSION_TIMEOUT)

    assert len(events) == 1
    assert events[0].speech_duration == pytest.approx(2.0, abs=0.5)


async def test_refires_on_next_attempt() -> None:
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
    # Both bursts belong to the same uncommitted turn.
    assert events[1].vad_speech_started_at == pytest.approx(events[0].vad_speech_started_at)


NO_TRANSCRIPT_MESSAGE = "no transcript for user speech"


def _no_transcript_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == NO_TRANSCRIPT_MESSAGE]


async def test_no_transcript_warning_fires_without_option(
    caplog: pytest.LogCaptureFixture,
) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "")

    session = create_session(actions)
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    await asyncio.wait_for(run_session(session, _agent(), drain_delay=10), timeout=SESSION_TIMEOUT)

    records = _no_transcript_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].speech_duration == pytest.approx(2.0, abs=0.5)
    assert hasattr(records[0], "stt")


@pytest.mark.parametrize(
    ("start", "end", "transcript", "final"),
    [
        (0.5, 2.5, "Hello, how are you?", True),  # transcribed
        (0.5, 2.5, "Hello, how are you?", False),  # interim only still counts as a transcript
        (0.5, 1.0, "", True),  # short blip below the minimum speech duration
    ],
)
async def test_no_transcript_warning_skipped(
    caplog: pytest.LogCaptureFixture, start: float, end: float, transcript: str, final: bool
) -> None:
    actions = FakeActions()
    actions.add_user_speech(start, end, transcript, final=final)
    if transcript and final:
        actions.add_llm("I'm doing well, thank you!")
        actions.add_tts(2.0)

    session = create_session(actions)
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    await asyncio.wait_for(run_session(session, _agent(), drain_delay=10), timeout=SESSION_TIMEOUT)

    assert _no_transcript_records(caplog) == []


def test_no_transcript_warning_skipped_when_audio_withheld() -> None:
    # uninterruptible agent speech and AEC warmup send silence to STT, so no transcript is
    # expected; the fake audio input pushes too few frames to drive this end to end
    recognition = AudioRecognition.__new__(AudioRecognition)
    recognition._no_transcript_warning_handle = None
    recognition._stt_audio_withheld = True
    recognition._segment_transcribed = False

    recognition._arm_no_transcript_warning(2.0)

    assert recognition._no_transcript_warning_handle is None
