from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    LanguageCode,
    ModelSettings,
    RealtimeInputMode,
    StopResponse,
    TurnHandlingOptions,
    llm,
    stt,
)
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics
from livekit.agents.voice.speech_handle import InputDetails, SpeechHandle
from livekit.agents.voice.turn import _StreamingTurnDetector

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _BufferedRealtimeSession(FakeRealtimeSession):
    """Fake provider buffer that records which audio each generation consumes."""

    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.provider_audio: list[rtc.AudioFrame] = []
        self.generated_audio: list[list[rtc.AudioFrame]] = []
        self.clear_audio_calls = 0
        self.commit_audio_calls = 0
        self.commit_audio_error: Exception | None = None
        self.clear_audio_error: Exception | None = None
        self.push_audio_error: Exception | None = None
        self.generate_reply_error: Exception | None = None
        self.resolve_replies_immediately = False

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.push_audio_error is not None:
            raise self.push_audio_error
        super().push_audio(frame)
        self.provider_audio.append(frame)

    def clear_audio(self) -> None:
        self.clear_audio_calls += 1
        if self.clear_audio_error is not None:
            raise self.clear_audio_error
        super().clear_audio()
        self.provider_audio.clear()

    def commit_audio(self) -> None:
        self.commit_audio_calls += 1
        if self.commit_audio_error is not None:
            raise self.commit_audio_error
        super().commit_audio()

    def generate_reply(self, **kwargs: Any) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if self.generate_reply_error is not None:
            raise self.generate_reply_error
        self.generated_audio.append(self.provider_audio.copy())
        self.provider_audio.clear()
        reply_fut = super().generate_reply(**kwargs)
        if self.resolve_replies_immediately:
            reply_fut.set_result(cast(llm.GenerationCreatedEvent, object()))
        return reply_fut


class _EditingAgent(Agent):
    def __init__(self, *, stop_first_turn: bool = False) -> None:
        super().__init__(instructions="test")
        self._stop_first_turn = stop_first_turn
        self.completed_turns = 0

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.completed_turns += 1
        if self._stop_first_turn and self.completed_turns == 1:
            raise StopResponse()
        new_message.content = [f"edited: {new_message.raw_text_content}"]


class _RejectingAgent(Agent):
    def __init__(self, *, stop_response: bool) -> None:
        super().__init__(instructions="test")
        self._stop_response = stop_response

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if self._stop_response:
            raise StopResponse()
        raise RuntimeError("reject finalized turn")


class _BlockingEmptyEditingAgent(Agent):
    def __init__(self, *, replacement: str | None) -> None:
        super().__init__(instructions="test")
        self.replacement = replacement
        self.hook_started = asyncio.Event()
        self.release_hook = asyncio.Event()
        self.completed_turns = 0

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        self.completed_turns += 1
        self.hook_started.set()
        await self.release_hook.wait()
        if self.replacement is not None:
            new_message.content = [self.replacement]


class _CustomSTTNodeAgent(Agent):
    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterator[SpeechEvent]:
        del model_settings
        async for _ in audio:
            yield SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
            yield SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="custom text", language=LanguageCode("en"))],
            )
            yield SpeechEvent(type=SpeechEventType.END_OF_SPEECH)
            return


def _session(
    *,
    mode: RealtimeInputMode = "text",
    capabilities: llm.RealtimeCapabilities | None = None,
    include_stt: bool = True,
) -> tuple[AgentSession, FakeRealtimeModel]:
    model = FakeRealtimeModel(
        capabilities=capabilities
        or fake_capabilities(turn_detection=False, can_disable_turn_detection=False)
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT() if include_stt else None,
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode=mode,
        ),
    )
    return session, model


def _activity(
    *, agent: Agent | None = None, mode: RealtimeInputMode = "text"
) -> tuple[AgentActivity, FakeRealtimeSession]:
    session, model = _session(mode=mode)
    activity = AgentActivity(agent or _EditingAgent(), session)
    rt_session = model.session()
    activity._rt_session = rt_session
    activity._scheduling_paused = False
    return activity, rt_session


def _eot(
    transcript: str,
    *,
    skip_reply: bool = False,
    reply_already_triggered: bool = False,
) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        reply_already_triggered=reply_already_triggered,
        new_transcript=transcript,
        transcript_confidence=0.91,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=1.0,
            stopped_speaking_at=2.0,
            transcription_delay=0.1,
            end_of_turn_delay=0.2,
        ),
        backchannel_over_agent=False,
    )


async def _complete_turn(
    activity: AgentActivity, info: _EndOfTurnInfo
) -> list[llm.ChatMessage | None]:
    captured: list[llm.ChatMessage | None] = []

    def _capture_generate_reply(**kwargs: Any) -> SpeechHandle:
        user_message = kwargs.get("user_message")
        captured.append(None if user_message is None else cast(llm.ChatMessage, user_message))
        return SpeechHandle.create()

    audio_input_ready_fut = None
    if (
        activity._rt_session is not None
        and activity._realtime_input_mode == "audio"
        and not activity._rt_turn_detection_enabled
        and not info.reply_already_triggered
    ):
        audio_input_ready_fut = activity._seal_realtime_audio_input()

    activity._generate_reply = _capture_generate_reply  # type: ignore[method-assign]
    task = asyncio.create_task(
        activity._user_turn_completed_task(None, info, audio_input_ready_fut)
    )
    activity._user_turn_completed_atask = task
    await task
    return captured


def _frame(fill: int = 0) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        bytes([fill]) * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


def _replace_realtime_session(activity: AgentActivity) -> _BufferedRealtimeSession:
    rt_session = _BufferedRealtimeSession(cast(FakeRealtimeModel, activity.llm))
    activity._rt_session = rt_session
    activity._started = True
    activity._audio_recognition = cast(
        Any, type("Recognition", (), {"_push_audio": lambda *args, **kwargs: None})()
    )
    return rt_session


def _create_audio_reply(
    activity: AgentActivity,
    *,
    authorize: bool,
    user_message: llm.ChatMessage | None = None,
) -> tuple[SpeechHandle, asyncio.Task[None]]:
    existing_tasks = set(activity._speech_tasks)
    speech_handle = activity._generate_reply(
        schedule_speech=False,
        input_details=InputDetails(modality="audio"),
        user_message=user_message,
    )
    new_tasks = set(activity._speech_tasks) - existing_tasks
    assert len(new_tasks) == 1
    reply_task = cast(asyncio.Task[None], new_tasks.pop())
    if authorize:
        speech_handle._authorize_generation()
    return speech_handle, reply_task


async def _wait_for_reply_count(
    rt_session: FakeRealtimeSession,
    reply_tasks: list[asyncio.Task[None]],
    count: int,
) -> None:
    async def _wait() -> None:
        while len(rt_session._reply_futs) < count:
            for task in reply_tasks:
                if task.done():
                    await task
                    raise AssertionError("reply task exited before creating its provider future")
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait(), timeout=1.0)


def _observe_authorization_wait(speech_handle: SpeechHandle) -> asyncio.Event:
    entered = asyncio.Event()
    original_wait = speech_handle._wait_for_authorization

    async def _wait_for_authorization() -> None:
        entered.set()
        await original_wait()

    speech_handle._wait_for_authorization = _wait_for_authorization  # type: ignore[method-assign]
    return entered


def _observe_future_wait(speech_handle: SpeechHandle, target: asyncio.Future[Any]) -> asyncio.Event:
    entered = asyncio.Event()
    original_wait = speech_handle.wait_if_not_interrupted

    async def _wait_if_not_interrupted(aw: list[asyncio.Future[Any]]) -> None:
        if target in aw:
            entered.set()
        await original_wait(aw)

    speech_handle.wait_if_not_interrupted = _wait_if_not_interrupted  # type: ignore[method-assign]
    return entered


async def test_finishing_realtime_reply_does_not_clear_next_audio_turn() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    first_response_active = asyncio.Event()
    finish_first_response = asyncio.Event()
    generation_calls = 0

    async def _generation_task(**_: Any) -> None:
        nonlocal generation_calls
        generation_calls += 1
        if generation_calls == 1:
            first_response_active.set()
            await finish_first_response.wait()

    activity._realtime_generation_task = _generation_task  # type: ignore[method-assign]

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()
    _, first_reply_task = _create_audio_reply(activity, authorize=True)
    await _wait_for_reply_count(rt_session, [first_reply_task], 1)
    rt_session._reply_futs[0].set_result(cast(llm.GenerationCreatedEvent, object()))
    await first_response_active.wait()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    activity._seal_realtime_audio_input()

    # A third turn starts while the completed second turn remains sealed. It must stay
    # deferred until the second generation consumes its own provider input.
    third_frame = _frame(3)
    activity._start_realtime_user_activity()
    activity.push_audio(third_frame)
    assert rt_session.provider_audio == [second_frame]
    assert len(activity._deferred_realtime_audio_inputs) == 1
    assert activity._deferred_realtime_audio_inputs[0].frames == [third_frame]

    finish_first_response.set()
    await first_reply_task

    assert rt_session.clear_audio_calls == 0
    assert rt_session.provider_audio == [second_frame]
    assert len(activity._deferred_realtime_audio_inputs) == 1
    assert activity._deferred_realtime_audio_inputs[0].frames == [third_frame]

    _, second_reply_task = _create_audio_reply(activity, authorize=True)
    await _wait_for_reply_count(rt_session, [second_reply_task], 2)

    assert rt_session.generated_audio == [[first_frame], [second_frame]]
    assert rt_session.provider_audio == []
    assert rt_session.generate_reply_calls == 2
    assert rt_session.user_activity_start_calls == 2
    assert len(activity._deferred_realtime_audio_inputs) == 1
    assert activity._deferred_realtime_audio_inputs[0].frames == [third_frame]

    rt_session._reply_futs[1].set_result(cast(llm.GenerationCreatedEvent, object()))
    await second_reply_task

    # The second reply's own finally block must not clear the third turn it advanced.
    assert rt_session.clear_audio_calls == 0
    assert rt_session.provider_audio == [third_frame]


async def test_audio_reply_created_before_seal_keeps_input_ownership() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    rt_session.resolve_replies_immediately = True

    async def _generation_task(**_: Any) -> None:
        return

    activity._realtime_generation_task = _generation_task  # type: ignore[method-assign]

    frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(frame)

    # The reply may be created while input is active, before endpoint processing seals it.
    # Sealing is part of the same logical input and must not invalidate that reply's owner.
    handle, reply_task = _create_audio_reply(activity, authorize=False)
    activity._seal_realtime_audio_input()
    handle._authorize_generation()
    await reply_task

    assert rt_session.generate_reply_calls == 1
    assert rt_session.generated_audio == [[frame]]
    assert rt_session.provider_audio == []


async def test_deferred_audio_before_activity_start_survives_input_advance() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    rt_session.resolve_replies_immediately = True

    async def _generation_task(**_: Any) -> None:
        return

    activity._realtime_generation_task = _generation_task  # type: ignore[method-assign]

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()
    first_handle, first_reply_task = _create_audio_reply(activity, authorize=False)

    # Input audio reaches the provider path before external VAD reports activity. If the
    # preceding input advances in that interval, this pre-VAD audio must remain buffered.
    second_frame = _frame(2)
    activity.push_audio(second_frame)
    first_handle._authorize_generation()
    await first_reply_task

    assert rt_session.generated_audio == [[first_frame]]
    assert rt_session.provider_audio == [second_frame]
    assert rt_session.user_activity_start_calls == 1
    assert not activity._rt_user_activity_started

    activity._start_realtime_user_activity()
    activity._seal_realtime_audio_input()
    _, second_reply_task = _create_audio_reply(activity, authorize=True)
    await second_reply_task

    assert rt_session.generated_audio == [[first_frame], [second_frame]]
    assert rt_session.user_activity_start_calls == 2


async def test_pause_resume_with_fresh_session_resets_manual_audio_activity() -> None:
    session, model = _session(mode="audio", include_stt=False)
    activity = AgentActivity(Agent(instructions="test"), session)
    await activity.start()

    try:
        first_rt_session = cast(FakeRealtimeSession, activity._rt_session)
        first_token = activity._rt_audio_input_token
        first_frame = _frame(1)

        activity._start_realtime_user_activity()
        activity.push_audio(first_frame)
        activity._seal_realtime_audio_input()
        activity._start_realtime_user_activity()
        activity.push_audio(_frame(2))

        assert activity._rt_user_activity_started
        assert activity._deferred_realtime_audio_inputs

        await activity.pause(blocked_tasks=[])

        assert not activity._rt_user_activity_started
        assert not activity._rt_audio_input_sealed
        assert not activity._deferred_realtime_audio_inputs
        assert activity._rt_audio_input_token is not first_token

        await activity.resume()
        second_rt_session = cast(FakeRealtimeSession, activity._rt_session)
        assert second_rt_session is not first_rt_session

        activity._start_realtime_user_activity()

        assert second_rt_session.user_activity_start_calls == 1
        assert activity._rt_user_activity_started
    finally:
        await activity.aclose()


async def test_stale_reply_cancellation_during_authorization_preserves_next_turn() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    activity._seal_realtime_audio_input()

    speech_handle, reply_task = _create_audio_reply(activity, authorize=False)
    authorization_entered = _observe_authorization_wait(speech_handle)
    await authorization_entered.wait()

    # Discard the reply task's turn and advance the already-buffered next turn.
    activity._clear_realtime_input()
    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == [second_frame]

    reply_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reply_task

    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == [second_frame]
    assert activity._rt_user_activity_started
    assert activity._rt_audio_input_sealed


async def test_stale_reply_setup_does_not_generate_from_next_audio_turn() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    rt_session.resolve_replies_immediately = True

    async def _generation_task(**_: Any) -> None:
        return

    activity._realtime_generation_task = _generation_task  # type: ignore[method-assign]

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    activity._seal_realtime_audio_input()

    user_message = llm.ChatMessage(role="user", content=["stale audio turn"])
    speech_handle, reply_task = _create_audio_reply(
        activity, authorize=False, user_message=user_message
    )
    authorization_entered = _observe_authorization_wait(speech_handle)
    await authorization_entered.wait()

    activity._clear_realtime_input()
    speech_handle._authorize_generation()
    await reply_task

    assert rt_session.generate_reply_calls == 0
    assert rt_session.generated_audio == []
    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == [second_frame]
    assert activity._rt_user_activity_started
    assert activity._rt_audio_input_sealed
    assert rt_session.chat_ctx.get_by_id(user_message.id) is None
    assert activity._agent.chat_ctx.get_by_id(user_message.id) is None


async def test_realtime_reply_setup_failure_clears_only_its_owned_audio_turn() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    rt_session.generate_reply_error = RuntimeError("setup failed")

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    activity._seal_realtime_audio_input()

    _, reply_task = _create_audio_reply(activity, authorize=True)
    with pytest.raises(RuntimeError, match="setup failed"):
        await reply_task

    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == [second_frame]
    assert activity._rt_user_activity_started
    assert activity._rt_audio_input_sealed


async def test_deferred_replay_failure_does_not_abort_created_provider_generation() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    generation_started = asyncio.Event()

    async def _generation_task(**_: Any) -> None:
        generation_started.set()

    activity._realtime_generation_task = _generation_task  # type: ignore[method-assign]

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    assert len(activity._deferred_realtime_audio_inputs) == 1
    rt_session.push_audio_error = RuntimeError("deferred replay failed")
    _, reply_task = _create_audio_reply(activity, authorize=True)
    await _wait_for_reply_count(rt_session, [reply_task], 1)

    provider_generation_fut = rt_session._reply_futs[0]
    provider_generation_fut.set_result(cast(llm.GenerationCreatedEvent, object()))
    await reply_task

    assert provider_generation_fut.done() and not provider_generation_fut.cancelled()
    assert generation_started.is_set()
    assert rt_session.generate_reply_calls == 1
    assert rt_session.generated_audio == [[first_frame]]
    assert not activity._deferred_realtime_audio_inputs
    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == []

    rt_session.push_audio_error = None
    next_frame = _frame(3)
    activity._start_realtime_user_activity()
    activity.push_audio(next_frame)

    assert rt_session.provider_audio == [next_frame]


async def test_deferred_replay_failure_does_not_report_unretrieved_future() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)

    activity._start_realtime_user_activity()
    activity.push_audio(_frame(1))
    activity._seal_realtime_audio_input()

    activity._start_realtime_user_activity()
    activity.push_audio(_frame(2))
    rt_session.push_audio_error = RuntimeError("deferred replay failed")

    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:

        def _fail_replay() -> None:
            with pytest.raises(RuntimeError, match="deferred replay failed"):
                activity._advance_realtime_audio_input()

        # Keep the failed ready future unowned, matching replay before the next EOU callback.
        _fail_replay()
        rt_session.push_audio_error = None
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    unretrieved = [
        context
        for context in loop_errors
        if context.get("message") == "Future exception was never retrieved"
        and str(context.get("exception")) == "deferred replay failed"
    ]
    assert unretrieved == []


def test_default_audio_mode_still_forwards_realtime_audio() -> None:
    activity, rt_session = _activity(mode="audio")
    recognition = cast(
        Any, type("Recognition", (), {"_push_audio": lambda *args, **kwargs: None})()
    )
    activity._audio_recognition = recognition
    activity._started = True

    frame = _frame()
    activity.push_audio(frame)

    assert rt_session.pushed_audio == [frame]


def test_text_mode_routes_audio_only_to_external_recognition() -> None:
    activity, rt_session = _activity()
    received: list[rtc.AudioFrame] = []

    class _Recognition:
        def _push_audio(self, frame: rtc.AudioFrame, *, stt_frame: rtc.AudioFrame | None) -> None:
            received.append(frame)

    activity._audio_recognition = cast(Any, _Recognition())
    activity._started = True

    frame = _frame()
    activity.push_audio(frame)

    assert received == [frame]
    assert rt_session.pushed_audio == []


async def test_finalized_edited_transcript_reaches_realtime_generation_once() -> None:
    activity, rt_session = _activity()

    captured = await _complete_turn(activity, _eot("hello world"))

    assert len(captured) == 1
    assert captured[0] is not None
    assert captured[0].raw_text_content == "edited: hello world"
    assert captured[0].transcript_confidence == 0.91
    assert captured[0].metrics["transcription_delay"] == 0.1
    assert rt_session.committed is False


async def test_two_text_turns_do_not_share_transcript_state() -> None:
    activity, _ = _activity()

    first = await _complete_turn(activity, _eot("one"))
    second = await _complete_turn(activity, _eot("two"))

    assert first[0] is not None and second[0] is not None
    assert [first[0].raw_text_content, second[0].raw_text_content] == [
        "edited: one",
        "edited: two",
    ]
    assert first[0].id != second[0].id


async def test_text_mode_skip_reply_keeps_local_transcript_without_generation() -> None:
    activity, rt_session = _activity()

    captured = await _complete_turn(activity, _eot("do not answer", skip_reply=True))

    assert captured == []
    assert rt_session.generate_reply_calls == 0
    assert rt_session.committed is False
    assert rt_session.audio_cleared is False
    messages = activity.agent.chat_ctx.messages()
    assert len(messages) == 1
    assert messages[0].raw_text_content == "do not answer"
    assert rt_session.chat_ctx.items == []


async def test_text_mode_skip_reply_stays_local_after_fallback_restart() -> None:
    primary = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    adapter = llm.RealtimeModelFallbackAdapter([primary])
    session = AgentSession(
        llm=adapter,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
        ),
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    rt_session = adapter.session(turn_detection_disabled=True)
    activity._rt_session = rt_session
    activity._scheduling_paused = False
    session._activity = activity
    session._agent = activity.agent
    rt_session._agent_session = session

    await _complete_turn(activity, _eot("keep local only", skip_reply=True))
    await adapter.restart_session()

    assert [message.raw_text_content for message in activity.agent.chat_ctx.messages()] == [
        "keep local only"
    ]
    assert rt_session.chat_ctx.messages() == []


async def test_text_mode_interrupt_preserves_provider_interruption_signal() -> None:
    activity, rt_session = _activity()

    await activity.interrupt()

    assert rt_session.interrupted is True
    assert rt_session.audio_cleared is False


async def test_audio_mode_skip_reply_discards_provider_audio_but_keeps_transcript() -> None:
    activity, rt_session = _activity(mode="audio")

    captured = await _complete_turn(activity, _eot("do not retain or answer", skip_reply=True))

    assert captured == []
    assert rt_session.generate_reply_calls == 0
    assert rt_session.audio_cleared is True
    messages = activity.agent.chat_ctx.messages()
    assert len(messages) == 1
    assert messages[0].raw_text_content == "do not retain or answer"
    assert rt_session.chat_ctx.items == []


async def test_short_external_transcript_retains_matching_realtime_audio() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    activity._session.options.interruption["min_words"] = 2
    current_speech = MagicMock()
    current_speech.allow_interruptions = True
    current_speech.interrupted = False
    activity._current_speech = cast(Any, current_speech)

    frame = _frame(1)
    input_token = activity._rt_audio_input_token
    activity._start_realtime_user_activity()
    activity.push_audio(frame)

    assert activity.on_end_of_turn(_eot("brief")) is False
    assert rt_session.provider_audio == [frame]
    assert rt_session.clear_audio_calls == 0
    assert activity._rt_audio_input_token is input_token
    assert activity._rt_user_activity_started


async def test_stop_response_discards_turn_and_next_turn_is_ready() -> None:
    activity, rt_session = _activity(agent=_EditingAgent(stop_first_turn=True))

    first = await _complete_turn(activity, _eot("stop"))
    second = await _complete_turn(activity, _eot("continue"))

    assert first == []
    assert rt_session.audio_cleared is False
    assert activity.agent.chat_ctx.items == []
    assert rt_session.chat_ctx.items == []
    assert len(second) == 1
    assert second[0] is not None
    assert second[0].raw_text_content == "edited: continue"


@pytest.mark.parametrize("stop_response", [True, False])
async def test_rejected_audio_turn_is_not_submitted_before_hook_finishes(
    stop_response: bool,
) -> None:
    activity, _ = _activity(agent=_RejectingAgent(stop_response=stop_response), mode="audio")
    rt_session = _replace_realtime_session(activity)
    frame = _frame(4)
    activity._start_realtime_user_activity()
    activity.push_audio(frame)

    captured = await _complete_turn(activity, _eot("reject this turn"))

    assert captured == []
    assert rt_session.commit_audio_calls == 0
    assert rt_session.generate_reply_calls == 0
    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == []


async def test_empty_text_turn_does_not_generate() -> None:
    activity, rt_session = _activity(agent=Agent(instructions="test"))

    captured = await _complete_turn(activity, _eot(""))

    assert captured == []
    assert rt_session.generate_reply_calls == 0
    assert rt_session.committed is False
    assert rt_session.audio_cleared is False


@pytest.mark.parametrize("replacement", ["replacement from hook", None])
async def test_empty_text_turn_runs_hook_before_interrupting_active_speech(
    replacement: str | None,
) -> None:
    agent = _BlockingEmptyEditingAgent(replacement=replacement)
    activity, _ = _activity(agent=agent)
    activity._session.options.interruption["min_words"] = 1
    current_speech = MagicMock()
    current_speech.allow_interruptions = True
    current_speech.interrupted = False
    current_speech.interrupt = AsyncMock()
    activity._current_speech = cast(Any, current_speech)
    activity._cancel_speech_pause = AsyncMock()  # type: ignore[method-assign]
    activity._interrupt_background_speeches = MagicMock(return_value=[])  # type: ignore[method-assign]
    captured: list[llm.ChatMessage] = []

    def _capture_generate_reply(**kwargs: Any) -> SpeechHandle:
        captured.append(cast(llm.ChatMessage, kwargs["user_message"]))
        return SpeechHandle.create()

    activity._generate_reply = _capture_generate_reply  # type: ignore[method-assign]

    assert activity.on_end_of_turn(_eot("")) is True
    await asyncio.wait_for(agent.hook_started.wait(), timeout=1.0)
    current_speech.interrupt.assert_not_awaited()

    agent.release_hook.set()
    assert activity._user_turn_completed_atask is not None
    await activity._user_turn_completed_atask

    assert agent.completed_turns == 1
    if replacement is None:
        current_speech.interrupt.assert_not_awaited()
        assert captured == []
    else:
        current_speech.interrupt.assert_awaited_once()
        assert len(captured) == 1
        assert captured[0].raw_text_content == replacement


def test_text_mode_requires_realtime_model() -> None:
    session = AgentSession(
        llm=FakeLLM(),
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad", realtime_input_mode="text"),
    )

    with pytest.raises(ValueError, match="RealtimeModel"):
        AgentActivity(Agent(instructions="test"), session)


def test_text_mode_requires_external_stt() -> None:
    session, _ = _session(include_stt=False)

    with pytest.raises(ValueError, match="STT"):
        AgentActivity(Agent(instructions="test"), session)


def test_text_mode_rejects_non_streaming_stt_without_vad_for_default_node() -> None:
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
                mutable_chat_context=True,
            )
        ),
        stt=FakeSTT(streaming=False),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )

    with pytest.raises(ValueError, match="non-streaming STT.*VAD"):
        AgentActivity(Agent(instructions="test"), session)


def test_text_mode_without_vad_accepts_pre_wrapped_stt() -> None:
    wrapped_stt = stt.StreamAdapter(
        stt=FakeSTT(streaming=False), vad=FakeVAD(fake_user_speeches=[])
    )
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
                mutable_chat_context=True,
            )
        ),
        stt=wrapped_stt,
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )

    activity = AgentActivity(Agent(instructions="test"), session)

    assert activity.stt is wrapped_stt
    assert activity._turn_detection == "stt"


@pytest.mark.parametrize("turn_detection", [None, "manual", "stt"])
def test_text_mode_without_vad_accepts_custom_stt_node(
    turn_detection: Any,
) -> None:
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
                mutable_chat_context=True,
            )
        ),
        stt=FakeSTT(streaming=False),
        vad=None,
        turn_handling=TurnHandlingOptions(
            realtime_input_mode="text", turn_detection=turn_detection
        ),
    )

    activity = AgentActivity(_CustomSTTNodeAgent(instructions="test"), session)

    assert activity._turn_detection == turn_detection


def test_text_mode_without_vad_requires_explicit_detection_for_custom_stt_node() -> None:
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
                mutable_chat_context=True,
            )
        ),
        stt=FakeSTT(streaming=False),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )

    with pytest.raises(ValueError, match="custom Agent.stt_node.*explicit turn_detection"):
        AgentActivity(_CustomSTTNodeAgent(instructions="test"), session)


async def test_custom_stt_node_preserves_resolved_stt_boundary_on_runtime_swap() -> None:
    old_stt = FakeSTT()
    new_stt = FakeSTT(streaming=False)
    agent = _CustomSTTNodeAgent(instructions="test", stt=old_stt)
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
                mutable_chat_context=True,
            )
        ),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        assert activity._turn_detection == "stt"

        agent.update_options(stt=new_stt)

        assert activity.stt is new_stt
        assert activity._turn_detection == "stt"
    finally:
        await session.aclose()


def test_text_mode_requires_mutable_realtime_context() -> None:
    session, _ = _session(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=False,
        )
    )

    with pytest.raises(ValueError, match="mutable chat context"):
        AgentActivity(Agent(instructions="test"), session)


def test_text_mode_rejects_server_side_turn_detection() -> None:
    session, _ = _session(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )

    with pytest.raises(ValueError, match="server-side turn detection"):
        AgentActivity(Agent(instructions="test"), session)


def test_realtime_input_mode_defaults_to_audio_and_rejects_unknown_values() -> None:
    assert AgentSession().options.realtime_input_mode == "audio"

    with pytest.raises(ValueError, match="realtime_input_mode"):
        AgentSession(
            turn_handling=cast(TurnHandlingOptions, {"realtime_input_mode": "unsupported"})
        )


def test_agent_can_override_session_realtime_input_mode() -> None:
    session, _ = _session(mode="audio")
    agent = Agent(
        instructions="test",
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )

    activity = AgentActivity(agent, session)

    assert session.options.realtime_input_mode == "audio"
    assert activity.realtime_input_mode == "text"


def test_text_mode_accepts_stt_turn_detection_and_disables_supported_server_detection() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="stt", realtime_input_mode="text"),
    )

    activity = AgentActivity(Agent(instructions="test"), session)

    assert activity._turn_detection == "stt"
    assert activity._rt_turn_detection_enabled is False


async def test_text_mode_without_vad_uses_streaming_stt_turn_detection() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)

    assert activity._turn_detection == "stt"
    assert activity._rt_turn_detection_enabled is False

    await activity.start()
    try:
        recognition = activity._audio_recognition
        assert recognition is not None
        assert recognition._turn_detection_mode == "stt"
    finally:
        await activity.aclose()


async def test_text_mode_without_vad_waits_for_streaming_stt_end_of_speech() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)

    await activity.start()
    try:
        recognition = activity._audio_recognition
        assert recognition is not None
        completed_turns: list[tuple[str, str]] = []

        def _capture_eou(*_: Any, **kwargs: Any) -> None:
            completed_turns.append((recognition._current_transcript, kwargs["trigger"]))

        recognition._run_eou_detection = _capture_eou  # type: ignore[method-assign]

        await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.START_OF_SPEECH))
        for transcript in ("hello", "world"):
            await recognition._on_stt_event(
                SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(text=transcript, language=LanguageCode("en"))],
                )
            )

        assert recognition._speaking
        assert completed_turns == []

        await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

        assert not recognition._speaking
        assert completed_turns == [("hello world", "stt")]
    finally:
        await activity.aclose()


async def test_text_mode_runtime_none_restores_automatic_stt_turn_detection() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )
    await session.start(Agent(instructions="test"))
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None

        session.update_options(turn_detection="manual")
        assert activity._turn_detection == "manual"
        assert recognition._turn_detection_mode == "manual"

        session.update_options(turn_detection=None)
        assert activity._turn_detection == "stt"
        assert recognition._turn_detection_mode == "stt"
    finally:
        await session.aclose()


async def test_text_mode_runtime_realtime_llm_is_rejected_without_losing_stt_boundary() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(realtime_input_mode="text"),
    )
    await session.start(Agent(instructions="test"))
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        original_setting = session._turn_detection
        original_explicit = session._turn_detection_explicit

        with pytest.raises(ValueError, match="realtime_llm.*text input"):
            session.update_options(turn_detection="realtime_llm")

        assert session._turn_detection is original_setting
        assert session._turn_detection_explicit is original_explicit
        assert activity._turn_detection == "stt"
        assert recognition._turn_detection_mode == "stt"

        completed_turns: list[tuple[str, str]] = []

        def _capture_eou(*_: Any, **kwargs: Any) -> None:
            completed_turns.append((recognition._current_transcript, kwargs["trigger"]))

        recognition._run_eou_detection = _capture_eou  # type: ignore[method-assign]
        await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.START_OF_SPEECH))
        await recognition._on_stt_event(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="still works", language=LanguageCode("en"))],
            )
        )
        await recognition._on_stt_event(SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

        assert completed_turns == [("still works", "stt")]
    finally:
        await session.aclose()


async def test_runtime_manual_mode_cancels_internal_empty_transcript_fallback() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
            endpointing={"max_delay": 60.0},
        ),
    )
    await session.start(Agent(instructions="test"))
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        assert recognition._finalize_empty_transcript_on_timeout is True
        recognition._arm_transcription_timeout(1.0, delay=0.0)
        timeout_handle = recognition._transcription_timeout_handle
        assert timeout_handle is not None

        session.update_options(turn_detection="manual")

        assert recognition._finalize_empty_transcript_on_timeout is False
        assert timeout_handle.cancelled()
        assert recognition._transcription_timeout_handle is None
    finally:
        await session.aclose()


async def test_runtime_manual_mode_preserves_explicit_transcription_timeout() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        transcription_timeout=60.0,
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
        ),
    )
    await session.start(Agent(instructions="test"))
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        recognition._arm_transcription_timeout(1.0, delay=0.0)
        timeout_handle = recognition._transcription_timeout_handle
        assert timeout_handle is not None

        session.update_options(turn_detection="manual")

        assert recognition._finalize_empty_transcript_on_timeout is False
        assert recognition._transcription_timeout_handle is timeout_handle
        assert not timeout_handle.cancelled()
    finally:
        await session.aclose()


async def test_runtime_vad_mode_enables_internal_empty_transcript_fallback() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="manual",
            realtime_input_mode="text",
            endpointing={"max_delay": 60.0},
        ),
    )
    await session.start(Agent(instructions="test"))
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        assert recognition._finalize_empty_transcript_on_timeout is False

        session.update_options(turn_detection="vad")
        recognition._arm_transcription_timeout(1.0, delay=0.0)

        assert recognition._finalize_empty_transcript_on_timeout is True
        assert recognition._transcription_timeout_handle is not None
    finally:
        await session.aclose()


async def test_transcription_timeout_clear_does_not_finalize_replaced_turn() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        transcription_timeout=60.0,
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
        ),
    )
    await session.start(Agent(instructions="test"))
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        recognition._user_turn_start = 1.0
        recognition._turn_speech_duration = 0.5
        recognition._turn_transcript_received = False
        recognition._finalize_empty_transcript_on_timeout = True
        run_eou_detection = MagicMock()
        recognition._run_eou_detection = run_eou_detection  # type: ignore[method-assign]
        session.on("user_transcription_timeout", lambda _: session.clear_user_turn())

        recognition._on_transcription_timeout()

        run_eou_detection.assert_not_called()
        assert model.active_session.generate_reply_calls == 0
    finally:
        await session.aclose()


@pytest.mark.parametrize("turn_detection", [None, "manual", "stt"])
def test_text_mode_without_vad_preserves_explicit_turn_detection(
    turn_detection: Any,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(
            turn_detection=turn_detection,
            realtime_input_mode="text",
        ),
    )

    activity = AgentActivity(Agent(instructions="test"), session)

    assert activity._turn_detection == turn_detection


async def test_text_mode_creates_session_with_supported_server_detection_disabled() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad", realtime_input_mode="text"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)

    await activity.start()
    try:
        assert model.active_session.turn_detection_disabled is True
    finally:
        await activity.aclose()


async def test_realtime_session_is_not_reused_across_input_modes() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    shared_stt = FakeSTT()
    shared_vad = FakeVAD(fake_user_speeches=[])

    def _make(mode: RealtimeInputMode) -> AgentActivity:
        session = AgentSession(
            llm=model,
            stt=shared_stt,
            vad=shared_vad,
            turn_handling=TurnHandlingOptions(turn_detection="vad", realtime_input_mode=mode),
        )
        return AgentActivity(Agent(instructions="test"), session)

    audio_activity = _make("audio")
    text_activity = _make("text")
    audio_activity._rt_session = model.session()

    resources = await audio_activity._detach_reusable_resources(text_activity)

    assert resources.rt_session is None


async def test_external_transcript_events_are_available_only_in_explicit_text_mode() -> None:
    event = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(text="external words", language=LanguageCode("en"))],
    )

    text_activity, _ = _activity(mode="text")
    text_events: list[object] = []
    text_activity._session._user_input_transcribed = text_events.append  # type: ignore[method-assign]
    text_activity.on_final_transcript(event)
    if text_activity._cancel_speech_pause_task is not None:
        await text_activity._cancel_speech_pause_task

    audio_activity, _ = _activity(mode="audio")
    audio_events: list[object] = []
    audio_activity._session._user_input_transcribed = audio_events.append  # type: ignore[method-assign]
    audio_activity.on_final_transcript(event)
    if audio_activity._cancel_speech_pause_task is not None:
        await audio_activity._cancel_speech_pause_task

    assert len(text_events) == 1
    assert audio_events == []


async def test_manual_commit_waits_for_external_transcript_in_text_mode() -> None:
    activity, rt_session = _activity(mode="text")
    received: list[dict[str, object]] = []

    class _Recognition:
        def _commit_user_turn(self, **kwargs: object) -> asyncio.Future[str]:
            received.append(kwargs)
            fut = asyncio.Future[str]()
            fut.set_result("final transcript")
            return fut

    activity._audio_recognition = cast(Any, _Recognition())

    transcript = await activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)

    assert transcript == "final transcript"
    assert received == [
        {
            "audio_detached": False,
            "transcript_timeout": 1.0,
            "stt_flush_duration": 0.1,
            "skip_reply": False,
            "reply_already_triggered": False,
        }
    ]
    assert rt_session.user_activity_started is False
    assert rt_session.committed is False
    assert rt_session.generate_reply_calls == 0


async def test_manual_audio_commit_triggers_once_before_external_transcript_flush() -> None:
    activity, rt_session = _activity(mode="audio")
    received: list[dict[str, object]] = []
    reply_calls: list[dict[str, object]] = []

    class _Recognition:
        def _commit_user_turn(self, **kwargs: object) -> asyncio.Future[str]:
            received.append(kwargs)
            fut = asyncio.Future[str]()
            fut.set_result("final transcript")
            return fut

    def _generate_reply(**kwargs: object) -> object:
        reply_calls.append(kwargs)
        return object()

    activity._audio_recognition = cast(Any, _Recognition())
    activity._session.generate_reply = _generate_reply  # type: ignore[method-assign]

    transcript = await activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)

    assert transcript == "final transcript"
    assert rt_session.user_activity_started is True
    assert rt_session.committed is True
    assert rt_session.audio_cleared is False
    assert activity._rt_audio_input_sealed
    assert reply_calls == [{"input_modality": "audio"}]
    assert received == [
        {
            "audio_detached": False,
            "transcript_timeout": 1.0,
            "stt_flush_duration": 0.1,
            "skip_reply": False,
            "reply_already_triggered": True,
        }
    ]


async def test_consecutive_manual_audio_commits_wait_for_their_owned_input() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    rt_session.resolve_replies_immediately = True
    activity._session._activity = activity

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _commit_user_turn(self, **_: object) -> asyncio.Future[str]:
            fut = asyncio.Future[str]()
            fut.set_result("final transcript")
            return fut

    async def _generation_task(**_: Any) -> None:
        return

    activity._audio_recognition = cast(Any, _Recognition())
    activity._realtime_generation_task = _generation_task  # type: ignore[method-assign]

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    first_tasks = set(activity._speech_tasks)
    first_transcript = activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)
    first_reply_tasks = set(activity._speech_tasks) - first_tasks
    assert len(first_reply_tasks) == 1
    first_reply_task = cast(asyncio.Task[None], first_reply_tasks.pop())
    first_handle = activity._speech_q[-1][2]

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    second_tasks = set(activity._speech_tasks)
    second_transcript = activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)
    second_reply_tasks = set(activity._speech_tasks) - second_tasks
    assert len(second_reply_tasks) == 1
    second_reply_task = cast(asyncio.Task[None], second_reply_tasks.pop())
    second_handle = activity._speech_q[-1][2]

    # Authorizing the later turn first must not let it commit or generate from the first
    # provider buffer. It remains bound to its deferred input until that input is current.
    second_input_ready = activity._deferred_realtime_audio_inputs[0].ready_fut
    second_waiting_for_input = _observe_future_wait(second_handle, second_input_ready)
    second_handle._authorize_generation()
    await asyncio.wait_for(second_waiting_for_input.wait(), timeout=1.0)
    assert rt_session.generate_reply_calls == 0
    first_handle._authorize_generation()

    await asyncio.wait_for(asyncio.gather(first_reply_task, second_reply_task), timeout=1.0)
    assert await first_transcript == "final transcript"
    assert await second_transcript == "final transcript"
    assert rt_session.commit_audio_calls == 2
    assert rt_session.generate_reply_calls == 2
    assert rt_session.generated_audio == [[first_frame], [second_frame]]


@pytest.mark.parametrize("failure_point", ["commit_audio", "generate_reply"])
async def test_manual_audio_commit_setup_failure_clears_owned_input(
    failure_point: str,
) -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _commit_user_turn(self, **_: object) -> asyncio.Future[str]:
            raise AssertionError("recognition must not run after synchronous setup failure")

    activity._audio_recognition = cast(Any, _Recognition())
    failure = RuntimeError(f"{failure_point} failed")
    if failure_point == "commit_audio":
        rt_session.commit_audio_error = failure
    else:

        def _fail_generate_reply(**_: object) -> object:
            raise failure

        activity._session.generate_reply = _fail_generate_reply  # type: ignore[method-assign]

    frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(frame)

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)

    assert rt_session.clear_audio_calls == 1
    assert rt_session.provider_audio == []
    assert not activity._rt_user_activity_started
    assert not activity._rt_audio_input_sealed
    assert not activity._deferred_realtime_audio_inputs


async def test_manual_audio_commit_skip_reply_discards_without_provider_commit() -> None:
    activity, rt_session = _activity(mode="audio")
    received: list[dict[str, object]] = []
    reply_calls = 0

    class _Recognition:
        def _commit_user_turn(self, **kwargs: object) -> asyncio.Future[str]:
            received.append(kwargs)
            fut = asyncio.Future[str]()
            fut.set_result("skipped transcript")
            return fut

    def _generate_reply() -> object:
        nonlocal reply_calls
        reply_calls += 1
        return object()

    activity._audio_recognition = cast(Any, _Recognition())
    activity._session.generate_reply = _generate_reply  # type: ignore[method-assign]

    await activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1, skip_reply=True)

    assert rt_session.user_activity_started is False
    assert rt_session.committed is False
    assert rt_session.audio_cleared is True
    assert reply_calls == 0
    assert received == [
        {
            "audio_detached": False,
            "transcript_timeout": 1.0,
            "stt_flush_duration": 0.1,
            "skip_reply": True,
            "reply_already_triggered": True,
        }
    ]


async def test_manual_skip_discards_deferred_turn_without_clearing_sealed_input() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    received: list[dict[str, object]] = []

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _commit_user_turn(self, **kwargs: object) -> asyncio.Future[str]:
            received.append(kwargs)
            fut = asyncio.Future[str]()
            fut.set_result("skipped transcript")
            return fut

    activity._audio_recognition = cast(Any, _Recognition())

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    assert rt_session.provider_audio == [first_frame]
    assert len(activity._deferred_realtime_audio_inputs) == 1

    transcript = activity.commit_user_turn(
        transcript_timeout=1.0,
        stt_flush_duration=0.1,
        skip_reply=True,
    )

    assert await transcript == "skipped transcript"
    assert rt_session.provider_audio == [first_frame]
    assert rt_session.clear_audio_calls == 0
    assert activity._rt_audio_input_sealed
    assert not activity._deferred_realtime_audio_inputs
    assert received[0]["reply_already_triggered"] is True

    captured = await _complete_turn(
        activity,
        _eot(
            "skipped transcript",
            skip_reply=True,
            reply_already_triggered=cast(bool, received[0]["reply_already_triggered"]),
        ),
    )

    assert captured == []
    assert rt_session.provider_audio == [first_frame]
    assert rt_session.clear_audio_calls == 0
    assert not activity._deferred_realtime_audio_inputs
    assert [message.raw_text_content for message in activity.agent.chat_ctx.messages()] == [
        "skipped transcript"
    ]


async def test_clear_user_turn_discards_deferred_turn_without_clearing_sealed_input() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    recognition_clears = 0

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _clear_user_turn(self) -> None:
            nonlocal recognition_clears
            recognition_clears += 1

    activity._audio_recognition = cast(Any, _Recognition())

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)

    activity.clear_user_turn()

    assert recognition_clears == 1
    assert rt_session.provider_audio == [first_frame]
    assert rt_session.clear_audio_calls == 0
    assert activity._rt_audio_input_sealed
    assert not activity._deferred_realtime_audio_inputs


async def test_paused_turn_discards_deferred_turn_without_clearing_sealed_input() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    activity._scheduling_paused = True

    assert activity.on_end_of_turn(_eot("rejected turn"))
    assert rt_session.provider_audio == [first_frame]
    assert rt_session.clear_audio_calls == 0
    assert activity._rt_audio_input_sealed
    assert not activity._deferred_realtime_audio_inputs


async def test_already_triggered_paused_turn_does_not_discard_newer_input() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)

    first_frame = _frame(1)
    activity._start_realtime_user_activity()
    activity.push_audio(first_frame)
    activity._seal_realtime_audio_input()

    second_frame = _frame(2)
    activity._start_realtime_user_activity()
    activity.push_audio(second_frame)
    activity._scheduling_paused = True

    assert activity.on_end_of_turn(_eot("already handled", reply_already_triggered=True))
    assert rt_session.provider_audio == [first_frame]
    assert rt_session.clear_audio_calls == 0
    assert len(activity._deferred_realtime_audio_inputs) == 1
    assert activity._deferred_realtime_audio_inputs[0].frames == [second_frame]


async def test_already_triggered_audio_eot_keeps_transcript_without_second_reply() -> None:
    activity, rt_session = _activity(mode="audio")
    activity.llm.capabilities.user_transcription = False
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append  # type: ignore[method-assign]

    captured = await _complete_turn(
        activity, _eot("flushed transcript", reply_already_triggered=True)
    )

    assert captured == []
    assert rt_session.generate_reply_calls == 0
    assert rt_session.audio_cleared is False
    assert [message.raw_text_content for message in activity.agent.chat_ctx.messages()] == [
        "flushed transcript"
    ]
    assert [message.id for message in added] == [activity.agent.chat_ctx.messages()[0].id]


async def test_provider_transcription_avoids_duplicate_manual_audio_user_message() -> None:
    activity, rt_session = _activity(mode="audio")
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append  # type: ignore[method-assign]

    captured = await _complete_turn(
        activity,
        _eot("external copy", reply_already_triggered=True),
    )

    assert captured == []
    assert activity.agent.chat_ctx.messages() == []
    assert added == []

    activity._on_input_audio_transcription_completed(
        llm.InputTranscriptionCompleted(
            item_id="provider-turn",
            transcript="provider copy",
            is_final=True,
        )
    )

    assert rt_session.generate_reply_calls == 0
    assert [message.raw_text_content for message in activity.agent.chat_ctx.messages()] == [
        "provider copy"
    ]
    assert [message.id for message in added] == ["provider-turn"]


async def test_finalized_turn_syncs_exact_message_and_starts_one_provider_generation() -> None:
    activity, rt_session = _activity(mode="text")
    added: list[llm.ChatMessage] = []
    activity._session._conversation_item_added = added.append  # type: ignore[method-assign]
    activity._authorization_allowed.set()
    activity._user_silence_event.set()
    reply_tasks: list[asyncio.Task[None]] = []

    def _start_realtime_reply(**kwargs: Any) -> SpeechHandle:
        handle = SpeechHandle.create()
        handle._authorize_generation()
        reply_tasks.append(
            asyncio.create_task(
                activity._realtime_reply_task(
                    speech_handle=handle,
                    model_settings=ModelSettings(),
                    user_message=cast(llm.ChatMessage, kwargs["user_message"]),
                )
            )
        )
        return handle

    activity._generate_reply = _start_realtime_reply  # type: ignore[method-assign]
    turn_task = asyncio.create_task(activity._user_turn_completed_task(None, _eot("original")))
    activity._user_turn_completed_atask = turn_task
    await turn_task

    async def _wait_for_reply() -> None:
        while not rt_session._reply_futs:
            if reply_tasks[0].done():
                await reply_tasks[0]
                raise AssertionError("reply task exited before creating its provider future")
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait_for_reply(), timeout=1.0)

    provider_message = rt_session.chat_ctx.messages()[0]
    local_message = activity.agent.chat_ctx.messages()[0]
    assert provider_message.id == local_message.id == added[0].id
    assert provider_message.raw_text_content == "edited: original"
    assert provider_message.transcript_confidence == 0.91
    assert provider_message.metrics["transcription_delay"] == 0.1
    assert rt_session.generate_reply_calls == 1
    assert rt_session.pushed_audio == []

    reply_tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await reply_tasks[0]
    assert rt_session._reply_futs[0].cancelled()


def test_text_mode_reselects_stt_after_explicit_unavailable_vad() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
        ),
    )

    activity = AgentActivity(Agent(instructions="test"), session)

    assert activity._turn_detection == "stt"


def test_adding_vad_restores_explicit_vad_after_temporary_stt_fallback() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=None,
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
        ),
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    assert activity._turn_detection == "stt"

    activity._update_models(new_vad=FakeVAD(fake_user_speeches=[]))

    assert activity._turn_detection == "vad"


def test_runtime_explicit_detector_warning_uses_prospective_setting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(realtime_input_mode="audio"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    detector = MagicMock(spec=_StreamingTurnDetector)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        activity.update_options(
            turn_detection=detector,
            session_turn_detection_explicit=True,
        )

    assert "ignoring the turn_detection setting" in caplog.text


def test_runtime_policy_update_preserves_turn_that_already_owns_audio() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    activity._started = True
    owning_turn = activity._rt_audio_input_token
    owning_policy = owning_turn.policy
    frame = _frame(10)

    activity.push_audio(frame)
    activity._audio_recognition = None
    activity.update_options(
        turn_detection="manual",
        session_turn_detection_explicit=True,
    )

    assert activity._turn_policy.turn_detection == "manual"
    assert activity._rt_audio_input_token is owning_turn
    assert owning_turn.policy is owning_policy
    assert rt_session.provider_audio == [frame]


def test_removing_vad_reselects_streaming_stt_boundary_in_text_mode() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=True,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
        ),
    )
    activity = AgentActivity(Agent(instructions="test"), session)

    activity._update_models(new_vad=None)

    assert activity.vad is None
    assert activity._turn_detection == "stt"


async def test_unknown_text_sync_still_generates_once() -> None:
    activity, _ = _activity(mode="text")
    rt_session = _replace_realtime_session(activity)
    rt_session.resolve_replies_immediately = True
    activity._authorization_allowed.set()
    activity._user_silence_event.set()

    async def _timeout_without_mirror_update(_: llm.ChatContext) -> None:
        raise llm.RealtimeError("provider acknowledgement timed out")

    rt_session.update_chat_ctx = _timeout_without_mirror_update  # type: ignore[method-assign]
    activity._realtime_generation_task = AsyncMock()  # type: ignore[method-assign]
    handle = SpeechHandle.create()
    handle._authorize_generation()
    message = llm.ChatMessage(role="user", content=["final external text"])

    await activity._realtime_reply_task(
        speech_handle=handle,
        model_settings=ModelSettings(),
        user_message=message,
    )

    assert rt_session.generate_reply_calls == 1
    assert activity.agent.chat_ctx.get_by_id(message.id) is message


@pytest.mark.parametrize("skip_reply", [False, True])
async def test_server_detected_manual_commit_never_seals_or_clears_provider_audio(
    skip_reply: bool,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=False,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        turn_handling=TurnHandlingOptions(realtime_input_mode="audio"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    rt_session = _BufferedRealtimeSession(model)
    activity._rt_session = rt_session
    activity._started = True
    generated: list[dict[str, object]] = []

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _commit_user_turn(self, **kwargs: object) -> asyncio.Future[str]:
            assert kwargs["reply_already_triggered"] is True
            fut = asyncio.Future[str]()
            fut.set_result("provider-owned")
            return fut

    activity._audio_recognition = cast(Any, _Recognition())
    activity._session.generate_reply = (  # type: ignore[method-assign]
        lambda **kwargs: generated.append(kwargs)
    )
    frame = _frame(7)
    activity.push_audio(frame)

    await activity.commit_user_turn(
        transcript_timeout=1.0,
        stt_flush_duration=0.1,
        skip_reply=skip_reply,
    )

    assert activity._rt_turn_detection_enabled
    assert rt_session.commit_audio_calls == 1
    assert rt_session.clear_audio_calls == 0
    assert rt_session.provider_audio == [frame]
    assert not activity._rt_audio_input_sealed
    assert generated == ([] if skip_reply else [{"input_modality": "audio"}])


async def test_clear_user_turn_does_not_clear_provider_owned_audio() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=False,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        turn_handling=TurnHandlingOptions(realtime_input_mode="audio"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    rt_session = _BufferedRealtimeSession(model)
    activity._rt_session = rt_session
    activity._started = True
    recognition_clears = 0

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _clear_user_turn(self) -> None:
            nonlocal recognition_clears
            recognition_clears += 1

    activity._audio_recognition = cast(Any, _Recognition())
    frame = _frame(8)
    activity.push_audio(frame)

    activity.clear_user_turn()

    assert recognition_clears == 1
    assert rt_session.clear_audio_calls == 0
    assert rt_session.provider_audio == [frame]


async def test_blocked_delayed_eou_does_not_clear_provider_owned_audio() -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=True,
            can_disable_turn_detection=False,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(),
        turn_handling=TurnHandlingOptions(realtime_input_mode="audio"),
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    rt_session = _BufferedRealtimeSession(model)
    activity._rt_session = rt_session
    activity._started = True
    activity._scheduling_paused = True
    frame = _frame(9)
    activity.push_audio(frame)

    assert activity.on_end_of_turn(_eot("late detector verdict"))

    assert rt_session.clear_audio_calls == 0
    assert rt_session.provider_audio == [frame]


async def test_clear_user_turn_does_not_discard_submitted_audio_reply() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    recognition_clears = 0

    class _Recognition:
        def _push_audio(self, *_: object, **__: object) -> None:
            pass

        def _commit_user_turn(self, **_: object) -> asyncio.Future[str]:
            fut = asyncio.Future[str]()
            fut.set_result("committed")
            return fut

        def _clear_user_turn(self) -> None:
            nonlocal recognition_clears
            recognition_clears += 1

    activity._audio_recognition = cast(Any, _Recognition())
    activity._session.generate_reply = lambda **_: object()  # type: ignore[method-assign]
    frame = _frame(3)
    activity._start_realtime_user_activity()
    activity.push_audio(frame)
    await activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)

    activity.clear_user_turn()

    assert recognition_clears == 1
    assert rt_session.clear_audio_calls == 0
    assert rt_session.provider_audio == [frame]


async def test_clear_audio_preserves_primary_error_when_deferred_replay_also_fails() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    activity._start_realtime_user_activity()
    activity.push_audio(_frame(1))
    activity._seal_realtime_audio_input()
    activity._start_realtime_user_activity()
    activity.push_audio(_frame(2))
    deferred_ready = activity._seal_realtime_audio_input()
    primary = RuntimeError("primary clear failure")
    secondary = RuntimeError("secondary replay failure")
    rt_session.clear_audio_error = primary
    rt_session.push_audio_error = secondary

    with pytest.raises(RuntimeError, match="primary clear failure") as exc_info:
        activity._clear_realtime_input()

    assert exc_info.value is primary
    assert deferred_ready.done()
    assert deferred_ready.exception() is secondary


async def test_cancellation_after_generation_creation_interrupts_owned_provider_output() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    activity._authorization_allowed.set()
    activity._user_silence_event.set()
    handle = SpeechHandle.create()
    handle._authorize_generation()
    owner = activity._rt_audio_input_token
    reply_task: asyncio.Task[None] | None = None

    def _create_then_cancel(**_: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
        assert reply_task is not None
        rt_session.generate_reply_calls += 1
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        generation = MagicMock(spec=llm.GenerationCreatedEvent)
        generation.user_initiated = True
        fut.set_result(generation)
        activity._on_generation_created(generation)
        reply_task.cancel()
        return fut

    cast(Any, rt_session).generate_reply = _create_then_cancel
    reply_task = asyncio.create_task(
        activity._realtime_reply_task(
            speech_handle=handle,
            model_settings=ModelSettings(),
            realtime_audio_input_owner=owner,
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await reply_task

    assert rt_session.interrupted is True


async def test_cancellation_does_not_interrupt_newer_provider_output() -> None:
    activity, _ = _activity(mode="audio")
    rt_session = _replace_realtime_session(activity)
    activity._authorization_allowed.set()
    activity._user_silence_event.set()
    handle = SpeechHandle.create()
    handle._authorize_generation()
    owner = activity._rt_audio_input_token
    reply_task: asyncio.Task[None] | None = None

    def _create_then_supersede(**_: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
        assert reply_task is not None
        rt_session.generate_reply_calls += 1
        owned_generation = MagicMock(spec=llm.GenerationCreatedEvent)
        owned_generation.user_initiated = True
        newer_generation = MagicMock(spec=llm.GenerationCreatedEvent)
        newer_generation.user_initiated = True
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        fut.set_result(owned_generation)
        activity._on_generation_created(owned_generation)
        activity._on_generation_created(newer_generation)
        reply_task.cancel()
        return fut

    cast(Any, rt_session).generate_reply = _create_then_supersede
    reply_task = asyncio.create_task(
        activity._realtime_reply_task(
            speech_handle=handle,
            model_settings=ModelSettings(),
            realtime_audio_input_owner=owner,
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await reply_task

    assert rt_session.interrupted is False
