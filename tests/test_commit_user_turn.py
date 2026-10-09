from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class _AudioRecognitionStub:
    def __init__(self, transcript_fut: asyncio.Future[str]) -> None:
        self._transcript_fut = transcript_fut
        self._end_of_turn_task: asyncio.Task[None] | None = None

    def _commit_user_turn(self, **_: Any) -> asyncio.Future[str]:
        return self._transcript_fut


def _create_activity() -> AgentActivity:
    session = AgentSession()
    return AgentActivity(Agent(instructions="test"), session)


@pytest.mark.asyncio
async def test_commit_user_turn_waits_for_current_turn_processing() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    eou_gate = asyncio.Event()
    turn_gate = asyncio.Event()
    message_committed_fut = loop.create_future()
    speech_handle = SpeechHandle.create()
    speech_handle._user_message_committed_fut = message_committed_fut

    async def finish_turn() -> SpeechHandle:
        await turn_gate.wait()
        return speech_handle

    recognition._end_of_turn_task = asyncio.create_task(eou_gate.wait())
    activity._user_turn_completed_atask = asyncio.create_task(finish_turn())
    transcript_fut.set_result("hello")

    await asyncio.sleep(0)
    assert not commit_fut.done()

    eou_gate.set()
    await asyncio.sleep(0)
    assert not commit_fut.done()

    turn_gate.set()
    await asyncio.sleep(0)
    assert not commit_fut.done()

    message_committed_fut.set_result(None)
    assert await commit_fut == "hello"


@pytest.mark.asyncio
async def test_commit_user_turn_ignores_previous_turn_tasks() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    transcript_fut.set_result("")
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()

    previous_eou_task = asyncio.create_task(asyncio.Event().wait())
    previous_turn_task = asyncio.create_task(asyncio.Event().wait())
    recognition._end_of_turn_task = previous_eou_task
    activity._user_turn_completed_atask = previous_turn_task
    activity._audio_recognition = cast(Any, recognition)

    try:
        commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)
        assert await asyncio.wait_for(commit_fut, timeout=1.0) == ""
    finally:
        previous_eou_task.cancel()
        previous_turn_task.cancel()
        await asyncio.gather(previous_eou_task, previous_turn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_commit_wait_does_not_cancel_turn_processing() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)
    eou_task = asyncio.create_task(asyncio.Event().wait())
    recognition._end_of_turn_task = eou_task
    transcript_fut.set_result("hello")
    await asyncio.sleep(0)

    commit_fut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await commit_fut

    assert not eou_task.cancelled()
    eou_task.cancel()
    await asyncio.gather(eou_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_commit_user_turn_waits_for_replacement_eou_task() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    first_eou_task = asyncio.create_task(asyncio.Event().wait())
    recognition._end_of_turn_task = first_eou_task
    transcript_fut.set_result("hello")
    await asyncio.sleep(0)

    replacement_gate = asyncio.Event()
    replacement_eou_task = asyncio.create_task(replacement_gate.wait())
    first_eou_task.cancel()
    recognition._end_of_turn_task = replacement_eou_task
    await asyncio.sleep(0)
    assert not commit_fut.done()

    replacement_gate.set()
    assert await commit_fut == "hello"


@pytest.mark.asyncio
async def test_commit_user_turn_propagates_eou_failure() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    async def fail_eou() -> None:
        raise RuntimeError("end-of-turn processing failed")

    recognition._end_of_turn_task = asyncio.create_task(fail_eou())
    transcript_fut.set_result("hello")

    with pytest.raises(RuntimeError, match="end-of-turn processing failed"):
        await commit_fut


@pytest.mark.asyncio
async def test_commit_user_turn_propagates_turn_processing_failure() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    async def fail_turn_processing() -> None:
        raise RuntimeError("user-turn processing failed")

    activity._user_turn_completed_atask = asyncio.create_task(fail_turn_processing())
    transcript_fut.set_result("hello")

    with pytest.raises(RuntimeError, match="user-turn processing failed"):
        await commit_fut


@pytest.mark.asyncio
async def test_turn_processing_exposes_pipeline_commit_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession()
    activity = AgentActivity(Agent(instructions="test", llm=FakeLLM()), session)
    activity._scheduling_paused = False
    activity._turn_detection = "manual"

    speech_handle = SpeechHandle.create()
    monkeypatch.setattr(activity, "_generate_reply", lambda **_: speech_handle)
    turn_info = _EndOfTurnInfo(
        skip_reply=False,
        new_transcript="hello",
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
    )

    turn_task = asyncio.create_task(activity._user_turn_completed_impl(None, turn_info))
    activity._user_turn_completed_atask = turn_task
    await asyncio.sleep(0)

    assert await turn_task is speech_handle
    assert speech_handle._user_message_committed_fut is not None
    assert not speech_handle._user_message_committed_fut.done()

    activity._mark_user_message_committed(speech_handle)
    await speech_handle._user_message_committed_fut
