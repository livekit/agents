from __future__ import annotations

import asyncio
from typing import Any, cast

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
)
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


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

    activity._generate_reply = _capture_generate_reply  # type: ignore[method-assign]
    task = asyncio.create_task(activity._user_turn_completed_task(None, info))
    activity._user_turn_completed_atask = task
    await task
    return captured


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(bytes(320), sample_rate=16000, num_channels=1, samples_per_channel=160)


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


async def test_text_mode_skip_reply_clears_input_without_generation() -> None:
    activity, rt_session = _activity()

    captured = await _complete_turn(activity, _eot("do not answer", skip_reply=True))

    assert captured == []
    assert rt_session.generate_reply_calls == 0
    assert rt_session.committed is False
    assert rt_session.audio_cleared is False
    assert activity.agent.chat_ctx.items == []
    assert rt_session.chat_ctx.items == []


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


async def test_empty_text_turn_does_not_generate() -> None:
    activity, rt_session = _activity(agent=Agent(instructions="test"))

    captured = await _complete_turn(activity, _eot(""))

    assert captured == []
    assert rt_session.generate_reply_calls == 0
    assert rt_session.committed is False
    assert rt_session.audio_cleared is False


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
    reply_calls = 0

    class _Recognition:
        def _commit_user_turn(self, **kwargs: object) -> asyncio.Future[str]:
            received.append(kwargs)
            fut = asyncio.Future[str]()
            fut.set_result("final transcript")
            return fut

    def _generate_reply() -> object:
        nonlocal reply_calls
        reply_calls += 1
        return object()

    activity._audio_recognition = cast(Any, _Recognition())
    activity._session.generate_reply = _generate_reply  # type: ignore[method-assign]

    transcript = await activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.1)

    assert transcript == "final transcript"
    assert rt_session.user_activity_started is True
    assert rt_session.committed is True
    assert rt_session.audio_cleared is False
    assert reply_calls == 1
    assert received == [
        {
            "audio_detached": False,
            "transcript_timeout": 1.0,
            "stt_flush_duration": 0.1,
            "skip_reply": False,
            "reply_already_triggered": True,
        }
    ]


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
            "reply_already_triggered": False,
        }
    ]


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
