from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, AgentSession, LanguageCode
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_session import FakeActions, create_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_TRANSCRIPT = "what's the weather tomorrow"


@pytest.fixture
async def running_session(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[AgentSession, Agent]]:
    session = create_session(
        FakeActions(),
        turn_handling={
            "turn_detection": getattr(request, "param", "manual"),
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
        },
        can_pause_audio=True,
    )
    agent = Agent(instructions="You are a helpful assistant.")
    audio_output = session.output.audio
    assert isinstance(audio_output, _SyncedAudioOutput)
    try:
        await session.start(agent, record=False)
        assert session._activity is not None
        assert session._activity._on_enter_task is not None
        await session._activity._on_enter_task
        yield session, agent
    finally:
        await session.aclose()
        await audio_output._synchronizer.aclose()


def _final_transcript(text: str) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(text=text, language=LanguageCode("en"), confidence=0.9)],
    )


@pytest.mark.parametrize("phase", ["scheduled", "waiting_for_stt", "pending_delivery"])
async def test_clear_preserves_manual_commit(
    running_session: tuple[AgentSession, Agent], phase: str
) -> None:
    session, agent = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    if phase == "waiting_for_stt":
        recognition._last_final_transcript_time = time.time() - 1.0

    commit = session.commit_user_turn(skip_reply=True)
    if phase == "waiting_for_stt":
        await asyncio.sleep(0)
        assert recognition._commit_user_turn_atask is not None
        assert not recognition._commit_user_turn_atask.done()
        assert not recognition._final_transcript_received.is_set()
        assert not recognition._user_turn_committed
    elif phase == "pending_delivery":
        assert await commit == _TRANSCRIPT
        assert recognition._end_of_turn_task is not None
        assert not recognition._end_of_turn_task.done()

    stt_pipeline = recognition._stt_pipeline
    session.clear_user_turn()

    assert recognition._current_transcript == _TRANSCRIPT
    assert recognition._stt_pipeline is stt_pipeline
    if phase == "waiting_for_stt":
        recognition._final_transcript_received.set()

    assert await commit == _TRANSCRIPT
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task
    assert activity._user_turn_completed_atask is not None
    await activity._user_turn_completed_atask

    messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert len(messages) == 1
    assert messages[0].text_content == _TRANSCRIPT
    assert messages[0].transcript_confidence == pytest.approx(0.9)


async def test_clear_discards_next_turn_after_manual_commit(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task

    await recognition._on_stt_event(_final_transcript("the next turn"))
    stt_pipeline = recognition._stt_pipeline
    session.clear_user_turn()

    assert recognition._current_transcript == ""
    assert recognition._stt_pipeline is not stt_pipeline
    assert recognition._last_final_transcript_time is None


async def test_clear_recovers_from_stale_manual_commit_flag(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    task = recognition._end_of_turn_task
    assert task is not None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    recognition._user_turn_committed = True

    session.clear_user_turn()

    assert recognition._current_transcript == ""
    assert not recognition._user_turn_committed


@pytest.mark.parametrize("running_session", ["vad", "stt"], indirect=True)
async def test_clear_still_discards_non_manual_commit(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    assert recognition._end_of_turn_task is not None
    assert not recognition._end_of_turn_task.done()
    stt_pipeline = recognition._stt_pipeline

    session.clear_user_turn()

    assert recognition._current_transcript == ""
    assert recognition._stt_pipeline is not stt_pipeline


async def test_ignored_clear_preserves_realtime_buffer(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    realtime_session = MagicMock()
    activity._rt_session = realtime_session
    try:
        session.clear_user_turn()

        realtime_session.clear_audio.assert_not_called()
        assert recognition._current_transcript == _TRANSCRIPT
    finally:
        activity._rt_session = None
