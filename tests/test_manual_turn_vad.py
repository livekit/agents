from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from livekit.agents import Agent, AgentSession, LanguageCode, vad
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


def _start_of_speech() -> vad.VADEvent:
    return vad.VADEvent(
        type=vad.VADEventType.START_OF_SPEECH,
        samples_index=0,
        timestamp=time.time(),
        speech_duration=0.1,
        silence_duration=0.0,
    )


@pytest.mark.parametrize("phase", ["queued", "pending_delivery"])
@pytest.mark.parametrize("clear_before_vad", [False, True])
async def test_manual_snapshot_survives_vad_and_live_clears(
    running_session: tuple[AgentSession, Agent], phase: str, clear_before_vad: bool
) -> None:
    session, agent = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    commit = session.commit_user_turn(skip_reply=True)
    if phase == "queued":
        await asyncio.sleep(0)
        assert commit.done()
    assert await commit == _TRANSCRIPT
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()
    if clear_before_vad:
        session.clear_user_turn()

    await recognition._on_vad_event(_start_of_speech())
    await recognition._on_stt_event(_final_transcript("never mind"))
    session.clear_user_turn()
    assert recognition._current_transcript == ""
    await task
    assert activity._user_turn_completed_atask is not None
    await activity._user_turn_completed_atask

    messages = [
        item for item in agent.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert len(messages) == 1
    assert messages[0].text_content == _TRANSCRIPT
    assert messages[0].transcript_confidence == pytest.approx(0.9)
    assert not recognition._user_turn_committed

    await recognition._on_stt_event(_final_transcript("the next turn"))
    assert recognition._current_transcript == "the next turn"


async def test_vad_preserves_all_pending_manual_snapshots(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, agent = running_session
    activity = session._activity
    assert activity is not None
    recognition = activity._audio_recognition
    assert recognition is not None
    for text in (_TRANSCRIPT, "next turn"):
        await recognition._on_stt_event(_final_transcript(text))
        assert await session.commit_user_turn(skip_reply=True) == text
        session.clear_user_turn()
    tasks = list(recognition._manual_turn_tasks)
    assert len(tasks) == 2

    await recognition._on_vad_event(_start_of_speech())
    await asyncio.gather(*tasks)
    assert activity._user_turn_completed_atask is not None
    await activity._user_turn_completed_atask

    assert [
        item.text_content
        for item in agent.chat_ctx.items
        if item.type == "message" and item.role == "user"
    ] == [_TRANSCRIPT, "next turn"]
    assert not recognition._user_turn_committed


@pytest.mark.parametrize("running_session", ["vad", "stt"], indirect=True)
async def test_vad_still_cancels_non_manual_delivery(
    running_session: tuple[AgentSession, Agent],
) -> None:
    session, _ = running_session
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    await recognition._on_stt_event(_final_transcript(_TRANSCRIPT))
    await session.commit_user_turn(skip_reply=True)
    task = recognition._end_of_turn_task
    assert task is not None and not task.done()

    await recognition._on_vad_event(_start_of_speech())

    with pytest.raises(asyncio.CancelledError):
        await task
