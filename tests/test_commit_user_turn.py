from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
)
from livekit.agents.voice.endpointing import BaseEndpointing
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class _AudioRecognitionStub:
    def __init__(self, transcript_fut: asyncio.Future[str]) -> None:
        self._transcript_fut = transcript_fut
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._user_silence_ev = asyncio.Event()
        self._user_silence_ev.set()
        self.turn_completion_fut: asyncio.Future[SpeechHandle | None] | None = None

    @property
    def _speaking(self) -> bool:
        return not self._user_silence_ev.is_set()

    async def _wait_for_user_silence(self) -> None:
        await self._user_silence_ev.wait()

    def _commit_user_turn(
        self,
        *,
        turn_completion_fut: asyncio.Future[SpeechHandle | None],
        **_: Any,
    ) -> asyncio.Future[str]:
        self.turn_completion_fut = turn_completion_fut
        return self._transcript_fut


class _TurnHooks:
    def __init__(self, *, complete_turn: bool, accept_turn: bool = True) -> None:
        self.complete_turn = complete_turn
        self.accept_turn = accept_turn
        self.turns: list[_EndOfTurnInfo] = []

    def retrieve_chat_ctx(self) -> llm.ChatContext:
        return llm.ChatContext.empty()

    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        self.turns.append(info)
        if (
            self.complete_turn
            and info.manual_turn_completion_fut is not None
            and not info.manual_turn_completion_fut.done()
        ):
            info.manual_turn_completion_fut.set_result(None)
        return self.accept_turn


def _create_recognition(
    hooks: _TurnHooks, *, endpointing_delay: float
) -> tuple[AudioRecognition, BaseEndpointing]:
    endpointing = BaseEndpointing(
        min_delay=endpointing_delay,
        max_delay=endpointing_delay,
    )
    recognition = AudioRecognition(
        AgentSession(),
        hooks=cast(Any, hooks),
        endpointing=endpointing,
        stt=cast(Any, object()),
        vad=None,
        interruption_detection=None,
        turn_detection="manual",
    )
    recognition._audio_transcript = "hello"
    recognition._last_final_transcript_time = time.time()
    return recognition, endpointing


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

    message_committed_fut = loop.create_future()
    speech_handle = SpeechHandle.create()
    speech_handle._user_message_committed_fut = message_committed_fut
    transcript_fut.set_result("hello")

    await asyncio.sleep(0)
    assert not commit_fut.done()

    assert recognition.turn_completion_fut is not None
    recognition.turn_completion_fut.set_result(speech_handle)
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
        assert recognition.turn_completion_fut is not None
        recognition.turn_completion_fut.set_result(None)
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
    transcript_fut.set_result("hello")
    await asyncio.sleep(0)

    commit_fut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await commit_fut

    assert recognition.turn_completion_fut is not None
    assert not recognition.turn_completion_fut.cancelled()
    recognition.turn_completion_fut.cancel()


@pytest.mark.asyncio
async def test_commit_user_turn_does_not_follow_a_later_turn_task() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    later_turn_task = asyncio.create_task(asyncio.Event().wait())
    activity._user_turn_completed_atask = later_turn_task
    try:
        transcript_fut.set_result("hello")
        assert recognition.turn_completion_fut is not None
        recognition.turn_completion_fut.set_result(None)
        assert await commit_fut == "hello"
    finally:
        later_turn_task.cancel()
        await asyncio.gather(later_turn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_commit_user_turn_waits_for_its_completion_after_user_resumes() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    transcript_fut.set_result("hello")
    await asyncio.sleep(0)

    cancelled_eou_task = asyncio.create_task(asyncio.Event().wait())
    recognition._end_of_turn_task = cancelled_eou_task
    cancelled_eou_task.cancel()
    await asyncio.sleep(0)
    assert not commit_fut.done()

    assert recognition.turn_completion_fut is not None
    recognition.turn_completion_fut.set_result(None)
    assert await commit_fut == "hello"


@pytest.mark.asyncio
async def test_commit_user_turn_propagates_eou_failure() -> None:
    loop = asyncio.get_running_loop()
    transcript_fut = loop.create_future()
    recognition = _AudioRecognitionStub(transcript_fut)
    activity = _create_activity()
    activity._audio_recognition = cast(Any, recognition)

    commit_fut = activity.commit_user_turn(transcript_timeout=2.0, stt_flush_duration=2.0)

    transcript_fut.set_result("hello")
    assert recognition.turn_completion_fut is not None
    recognition.turn_completion_fut.set_exception(RuntimeError("end-of-turn processing failed"))

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

    assert recognition.turn_completion_fut is not None
    recognition.turn_completion_fut.set_exception(RuntimeError("user-turn processing failed"))
    transcript_fut.set_result("hello")

    with pytest.raises(RuntimeError, match="user-turn processing failed"):
        await commit_fut


@pytest.mark.asyncio
async def test_manual_commit_survives_cancelled_eou_when_user_resumes() -> None:
    hooks = _TurnHooks(complete_turn=True)
    recognition, endpointing = _create_recognition(hooks, endpointing_delay=60.0)
    turn_completion_fut: asyncio.Future[None] = asyncio.Future()

    transcript_fut = recognition._commit_user_turn(
        audio_detached=False,
        transcript_timeout=2.0,
        skip_reply=True,
        turn_completion_fut=turn_completion_fut,
    )
    assert await transcript_fut == "hello"

    first_eou_task = recognition._end_of_turn_task
    assert first_eou_task is not None
    recognition._speaking = True
    first_eou_task.cancel()
    await asyncio.gather(first_eou_task, return_exceptions=True)
    assert not turn_completion_fut.done()

    recognition._speaking = False
    endpointing.update_options(min_delay=0.0, max_delay=0.0)
    recognition._run_eou_detection(llm.ChatContext.empty(), trigger="vad")

    await asyncio.wait_for(turn_completion_fut, timeout=1.0)
    assert len(hooks.turns) == 1


@pytest.mark.asyncio
async def test_manual_commit_is_bound_only_after_transcript_collection() -> None:
    hooks = _TurnHooks(complete_turn=True)
    recognition, _ = _create_recognition(hooks, endpointing_delay=0.0)
    recognition._last_final_transcript_time = None
    turn_completion_fut: asyncio.Future[None] = asyncio.Future()

    transcript_fut = recognition._commit_user_turn(
        audio_detached=False,
        transcript_timeout=2.0,
        skip_reply=True,
        turn_completion_fut=turn_completion_fut,
    )
    await asyncio.sleep(0)

    recognition._run_eou_detection(llm.ChatContext.empty(), trigger="vad")
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task
    assert hooks.turns[0].manual_turn_completion_fut is None
    assert not turn_completion_fut.done()

    recognition._audio_transcript = "world"
    recognition._last_final_transcript_time = time.time()
    recognition._final_transcript_received.set()

    assert await transcript_fut == "world"
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task
    await asyncio.wait_for(turn_completion_fut, timeout=1.0)
    assert hooks.turns[1].manual_turn_completion_fut is turn_completion_fut


@pytest.mark.asyncio
async def test_rejected_manual_turn_fails_completion() -> None:
    hooks = _TurnHooks(complete_turn=False, accept_turn=False)
    recognition, _ = _create_recognition(hooks, endpointing_delay=0.0)
    turn_completion_fut: asyncio.Future[None] = asyncio.Future()

    transcript_fut = recognition._commit_user_turn(
        audio_detached=False,
        transcript_timeout=2.0,
        skip_reply=True,
        turn_completion_fut=turn_completion_fut,
    )

    assert await transcript_fut == "hello"
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task
    with pytest.raises(RuntimeError, match="manual user turn was not committed"):
        await asyncio.wait_for(turn_completion_fut, timeout=1.0)
    assert recognition._pending_manual_turn is None


@pytest.mark.asyncio
async def test_turn_mode_change_cancels_pending_manual_completion() -> None:
    hooks = _TurnHooks(complete_turn=True)
    recognition, _ = _create_recognition(hooks, endpointing_delay=60.0)
    turn_completion_fut: asyncio.Future[None] = asyncio.Future()

    transcript_fut = recognition._commit_user_turn(
        audio_detached=False,
        transcript_timeout=2.0,
        skip_reply=True,
        turn_completion_fut=turn_completion_fut,
    )
    assert await transcript_fut == "hello"
    eou_task = recognition._end_of_turn_task
    assert eou_task is not None

    recognition._update_options(turn_detection="vad")

    with pytest.raises(asyncio.CancelledError):
        await turn_completion_fut
    assert recognition._pending_manual_turn is None
    await asyncio.gather(eou_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manual_commit_completion_is_not_reused_by_a_later_turn() -> None:
    hooks = _TurnHooks(complete_turn=False)
    recognition, _ = _create_recognition(hooks, endpointing_delay=0.0)
    turn_completion_fut: asyncio.Future[None] = asyncio.Future()

    transcript_fut = recognition._commit_user_turn(
        audio_detached=False,
        transcript_timeout=2.0,
        skip_reply=True,
        turn_completion_fut=turn_completion_fut,
    )
    assert await transcript_fut == "hello"
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task

    assert hooks.turns[0].manual_turn_completion_fut is turn_completion_fut
    assert recognition._pending_manual_turn is None

    recognition._audio_transcript = "later turn"
    recognition._run_eou_detection(llm.ChatContext.empty(), trigger="vad")
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task

    assert len(hooks.turns) == 2
    assert hooks.turns[1].manual_turn_completion_fut is None
    assert not turn_completion_fut.done()
    turn_completion_fut.cancel()


@pytest.mark.asyncio
async def test_manual_completion_follows_the_exact_turn_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity = _create_activity()
    activity._scheduling_paused = False
    completion_fut: asyncio.Future[SpeechHandle | None] = asyncio.Future()
    turn_gate = asyncio.Event()
    speech_handle = SpeechHandle.create()

    async def complete_turn(
        old_task: asyncio.Task[SpeechHandle | None] | None,
        info: _EndOfTurnInfo,
    ) -> SpeechHandle:
        await turn_gate.wait()
        return speech_handle

    monkeypatch.setattr(activity, "_user_turn_completed_task", complete_turn)
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
        manual_turn_completion_fut=completion_fut,
    )

    assert activity.on_end_of_turn(turn_info)
    exact_turn_task = activity._user_turn_completed_atask
    assert exact_turn_task is not None

    later_turn_task = asyncio.create_task(asyncio.Event().wait())
    activity._user_turn_completed_atask = later_turn_task
    try:
        turn_gate.set()
        assert await completion_fut is speech_handle
        assert not later_turn_task.done()
    finally:
        later_turn_task.cancel()
        await asyncio.gather(later_turn_task, return_exceptions=True)


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


@pytest.mark.asyncio
async def test_overlapping_turn_waits_for_previous_message_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity = _create_activity()
    previous_speech_handle = SpeechHandle.create()
    previous_speech_handle._user_message_committed_fut = asyncio.Future[None]()

    async def previous_turn() -> SpeechHandle:
        return previous_speech_handle

    previous_turn_task = asyncio.create_task(previous_turn())
    reached_turn_processing = asyncio.Event()

    def interrupt_background_speeches(*, force: bool) -> list[asyncio.Future[None]]:
        reached_turn_processing.set()
        return []

    monkeypatch.setattr(activity, "_interrupt_background_speeches", interrupt_background_speeches)
    turn_info = _EndOfTurnInfo(
        skip_reply=False,
        new_transcript="next turn",
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
    )

    turn_task = asyncio.create_task(
        activity._user_turn_completed_impl(previous_turn_task, turn_info)
    )
    await asyncio.sleep(0)
    assert not reached_turn_processing.is_set()

    previous_speech_handle._user_message_committed_fut.set_result(None)
    await turn_task
    assert reached_turn_processing.is_set()


@pytest.mark.asyncio
async def test_previous_reply_failure_does_not_drop_overlapping_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activity = _create_activity()
    previous_speech_handle = SpeechHandle.create()
    previous_speech_handle._user_message_committed_fut = asyncio.Future[None]()

    async def previous_turn() -> SpeechHandle:
        return previous_speech_handle

    previous_turn_task = asyncio.create_task(previous_turn())
    reached_turn_processing = asyncio.Event()

    def interrupt_background_speeches(*, force: bool) -> list[asyncio.Future[None]]:
        reached_turn_processing.set()
        return []

    monkeypatch.setattr(activity, "_interrupt_background_speeches", interrupt_background_speeches)
    turn_info = _EndOfTurnInfo(
        skip_reply=False,
        new_transcript="next turn",
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
    )

    previous_speech_handle._user_message_committed_fut.set_exception(
        RuntimeError("previous reply failed")
    )
    await activity._user_turn_completed_impl(previous_turn_task, turn_info)

    assert reached_turn_processing.is_set()
