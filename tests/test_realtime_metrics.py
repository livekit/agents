"""Metrics a realtime turn reports: end-to-end latency, message timing, usage and redaction."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, LanguageCode, llm, stt, utils, vad
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
)

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

# `_vad` patches the process-wide `time.time`
pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture
async def realtime() -> AsyncIterator[tuple[AgentSession, FakeRealtimeModel]]:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=True))
    async with AgentSession(llm=model, vad=FakeVAD()) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="test"))
        yield session, model


@pytest.fixture
async def manual_realtime() -> AsyncIterator[tuple[AgentSession, FakeRealtimeModel]]:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(turn_detection=False, user_transcription=False)
    )
    async with AgentSession(
        llm=model,
        stt=FakeSTT(),
        vad=FakeVAD(),
        turn_handling={
            "turn_detection": "manual",
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
        },
    ) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        await session.start(Agent(instructions="test"))
        yield session, model


async def _vad(
    session: AgentSession,
    event_type: vad.VADEventType,
    *,
    at: float,
    speech: float = 0.0,
    silence: float = 0.0,
    inference_duration: float = 0.0,
) -> None:
    assert session._activity is not None
    assert session._activity._audio_recognition is not None
    with patch("time.time", return_value=at):
        await session._activity._audio_recognition._on_vad_event(
            vad.VADEvent(
                type=event_type,
                timestamp=at,
                samples_index=0,
                speech_duration=0.5,
                silence_duration=silence,
                raw_accumulated_speech=speech,
                raw_accumulated_silence=silence,
                inference_duration=inference_duration,
            )
        )


def _start_reply(
    session: AgentSession,
    model: FakeRealtimeModel,
    *,
    requested: SpeechHandle | None = None,
    additional_message: llm.MessageGeneration | None = None,
    response_id: str | None = None,
) -> tuple[SpeechHandle, utils.aio.Chan[rtc.AudioFrame]]:
    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    text = utils.aio.Chan[str]()
    audio = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])
    messages.send_nowait(
        llm.MessageGeneration(
            message_id=utils.shortuuid("message_"),
            text_stream=text,
            audio_stream=audio,
            modalities=modalities,
        )
    )
    text.send_nowait("Hello")
    text.close()
    if additional_message is not None:
        messages.send_nowait(additional_message)
    messages.close()
    functions.close()
    speeches: list[SpeechCreatedEvent] = []
    session.on("speech_created", speeches.append)
    event = llm.GenerationCreatedEvent(
        message_stream=messages,
        function_stream=functions,
        user_initiated=requested is not None,
        response_id=response_id or utils.shortuuid("response_"),
    )
    model.active_session.emit("generation_created", event)
    session.off("speech_created", speeches.append)
    if requested is not None:
        model.active_session._reply_futs[-1].set_result(event)
        return requested, audio
    assert len(speeches) == 1
    return speeches[0].speech_handle, audio


async def _finish_reply(
    session: AgentSession, handle: SpeechHandle, audio: utils.aio.Chan[rtc.AudioFrame]
) -> llm.ChatMessage:
    audio.send_nowait(
        rtc.AudioFrame(
            data=b"\x00\x00" * 240,
            sample_rate=24000,
            num_channels=1,
            samples_per_channel=240,
        )
    )
    audio.close()
    await handle
    return next(
        item
        for item in reversed(session.history.items)
        if isinstance(item, llm.ChatMessage) and item.role == "assistant"
    )


@pytest.mark.parametrize(
    "end_event", [vad.VADEventType.END_OF_SPEECH, vad.VADEventType.INFERENCE_DONE]
)
async def test_realtime_e2e_uses_local_speech_end_without_changing_ttft(
    realtime: tuple[AgentSession, FakeRealtimeModel], end_event: vad.VADEventType
) -> None:
    session, model = realtime
    collected: list[MetricsCollectedEvent] = []
    session.on("metrics_collected", collected.append)
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.INFERENCE_DONE, at=now - 0.4, speech=0.6)
    await _vad(session, end_event, at=now, silence=0.3, inference_duration=0.1)
    handle, audio = _start_reply(session, model)
    provider_metrics = RealtimeModelMetrics(
        request_id="provider-response",
        timestamp=now,
        ttft=0.05,
        input_token_details=RealtimeModelMetrics.InputTokenDetails(),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
    )
    model.active_session.emit("metrics_collected", provider_metrics)
    reply = await _finish_reply(session, handle, audio)

    assert reply.metrics["e2e_latency"] == pytest.approx(
        reply.metrics["started_speaking_at"] - (now - 0.4)
    )
    assert session._early_assistant_metrics["e2e_latency"] == reply.metrics["e2e_latency"]
    assert collected[-1].metrics is provider_metrics
    assert provider_metrics.ttft == 0.05


async def test_realtime_e2e_uses_last_speech_segment_before_playback(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 10.0)
    handle, audio = _start_reply(session, model)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now - 8.0, silence=0.5)
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 7.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now, silence=0.5)
    assert session._activity is not None
    assert session._activity._audio_recognition is not None
    recognition = session._activity._audio_recognition
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task
    assert recognition.last_speaking_time is None
    reply = await _finish_reply(session, handle, audio)

    assert reply.metrics["e2e_latency"] == pytest.approx(
        reply.metrics["started_speaking_at"] - (now - 0.5)
    )


async def test_realtime_e2e_is_not_reused_after_late_vad_eos(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.INFERENCE_DONE, at=now - 0.2, speech=0.8)
    await _vad(session, vad.VADEventType.INFERENCE_DONE, at=now, silence=0.2)
    first = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" in first.metrics

    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now + 0.4, silence=0.6)
    second = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in second.metrics


async def test_realtime_e2e_uses_new_speech_after_a_reply(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now, silence=0.4)
    first = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" in first.metrics

    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    second = await _finish_reply(session, *_start_reply(session, model))
    assert second.metrics["e2e_latency"] == pytest.approx(
        second.metrics["started_speaking_at"] - now
    )


async def test_realtime_e2e_does_not_use_previous_turn_when_user_resumes(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 3.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now - 1.0)
    assert session._activity is not None
    assert session._activity._audio_recognition is not None
    recognition = session._activity._audio_recognition
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task

    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now)
    reply = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in reply.metrics


@pytest.mark.parametrize("committed", [False, True])
async def test_realtime_e2e_does_not_use_cleared_user_speech(
    realtime: tuple[AgentSession, FakeRealtimeModel], committed: bool
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    if committed:
        assert session._activity is not None
        assert session._activity._audio_recognition is not None
        assert session._activity._audio_recognition._end_of_turn_task is not None
        await session._activity._audio_recognition._end_of_turn_task
    session.clear_user_turn()

    reply = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in reply.metrics


async def test_realtime_e2e_does_not_use_skipped_user_speech(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    await session.commit_user_turn(skip_reply=True)

    reply = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in reply.metrics


@pytest.mark.parametrize("committed_before_generation", [False, True])
async def test_realtime_e2e_uses_latest_turn_before_playback(
    realtime: tuple[AgentSession, FakeRealtimeModel], committed_before_generation: bool
) -> None:
    session, model = realtime
    assert session._activity is not None
    assert session._activity._audio_recognition is not None
    recognition = session._activity._audio_recognition
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 6.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now - 4.0)
    assert recognition._end_of_turn_task is not None
    if committed_before_generation:
        await recognition._end_of_turn_task
    first_handle, first_audio = _start_reply(session, model)
    await recognition._end_of_turn_task

    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 3.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now - 1.0)
    assert recognition._end_of_turn_task is not None
    await recognition._end_of_turn_task

    first = await _finish_reply(session, first_handle, first_audio)
    second = await _finish_reply(session, *_start_reply(session, model))
    assert first.metrics["e2e_latency"] == pytest.approx(
        first.metrics["started_speaking_at"] - (now - 1.0)
    )
    assert "e2e_latency" not in second.metrics


async def test_realtime_manual_commit_preserves_speech_timing(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    speeches: list[SpeechCreatedEvent] = []
    session.on("speech_created", speeches.append)
    await session.commit_user_turn()
    for _ in range(100):
        if model.active_session._reply_futs:
            break
        await asyncio.sleep(0)
    assert model.active_session._reply_futs
    assert len(speeches) == 1
    reply = await _finish_reply(
        session, *_start_reply(session, model, requested=speeches[0].speech_handle)
    )
    assert reply.metrics["e2e_latency"] == pytest.approx(reply.metrics["started_speaking_at"] - now)


@pytest.mark.parametrize("phase", ["pending_delivery", "delivered"])
@pytest.mark.parametrize("next_speech", ["none", "before_clear", "after_clear"])
async def test_realtime_manual_clear_preserves_latest_speech_timing(
    manual_realtime: tuple[AgentSession, FakeRealtimeModel], phase: str, next_speech: str
) -> None:
    session, model = manual_realtime
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None

    async def final(text: str) -> None:
        await recognition._on_stt_event(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text=text, language=LanguageCode("en"))],
            )
        )

    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 0.2)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    await final("weather")
    speeches: list[SpeechCreatedEvent] = []
    session.on("speech_created", speeches.append)
    assert await session.commit_user_turn() == "weather"
    assert recognition._end_of_turn_task is not None
    assert not recognition._end_of_turn_task.done()
    if next_speech == "before_clear":
        await _vad(session, vad.VADEventType.INFERENCE_DONE, at=time.time(), speech=0.03)
        await final("next turn")
    stopped_speaking_at = recognition.last_speaking_time
    assert stopped_speaking_at is not None
    session.clear_user_turn()
    if next_speech == "after_clear":
        await _vad(session, vad.VADEventType.INFERENCE_DONE, at=time.time(), speech=0.03)
        await final("next turn")
        stopped_speaking_at = recognition.last_speaking_time
        assert stopped_speaking_at is not None

    if phase == "delivered":
        await recognition._end_of_turn_task
    for _ in range(100):
        if model.active_session._reply_futs:
            break
        await asyncio.sleep(0)
    assert model.active_session._reply_futs
    assert len(speeches) == 1
    reply = await _finish_reply(
        session, *_start_reply(session, model, requested=speeches[0].speech_handle)
    )

    assert reply.metrics["e2e_latency"] == pytest.approx(
        reply.metrics["started_speaking_at"] - stopped_speaking_at
    )


async def test_realtime_manual_clear_does_not_restore_skipped_speech_timing(
    manual_realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = manual_realtime
    assert session._activity is not None
    recognition = session._activity._audio_recognition
    assert recognition is not None
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 0.2)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    await recognition._on_stt_event(
        stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="weather", language=LanguageCode("en"))],
        )
    )
    assert await session.commit_user_turn(skip_reply=True) == "weather"
    session.clear_user_turn()
    reply = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in reply.metrics


async def test_realtime_e2e_is_reported_only_on_first_played_message(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    text = utils.aio.Chan[str]()
    text.send_nowait("Another message")
    text.close()
    audio = utils.aio.Chan[rtc.AudioFrame]()
    audio.send_nowait(rtc.AudioFrame.create(24000, 1, 240))
    audio.close()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])
    additional_message = llm.MessageGeneration(
        message_id=utils.shortuuid("message_"),
        text_stream=text,
        audio_stream=audio,
        modalities=modalities,
    )
    await _finish_reply(
        session, *_start_reply(session, model, additional_message=additional_message)
    )
    replies = [
        item
        for item in session.history.items
        if isinstance(item, llm.ChatMessage) and item.role == "assistant"
    ]
    assert len(replies) == 2
    assert "e2e_latency" in replies[0].metrics
    assert "e2e_latency" not in replies[1].metrics


@pytest.mark.parametrize("speaking", [False, True])
async def test_realtime_e2e_requires_local_speech_timing(
    realtime: tuple[AgentSession, FakeRealtimeModel], speaking: bool
) -> None:
    session, model = realtime
    if speaking:
        await _vad(session, vad.VADEventType.START_OF_SPEECH, at=time.time() - 0.5)
    else:
        model.active_session.emit("input_speech_started", llm.InputSpeechStartedEvent())
        model.active_session.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )
    reply = await _finish_reply(session, *_start_reply(session, model))

    assert "e2e_latency" not in reply.metrics


async def test_realtime_e2e_includes_wait_before_first_audio(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)

    silent_handle, silent_audio = _start_reply(session, model)
    silent_audio.close()
    await silent_handle

    reply = await _finish_reply(session, *_start_reply(session, model))
    assert reply.metrics["e2e_latency"] == pytest.approx(reply.metrics["started_speaking_at"] - now)


async def test_realtime_e2e_survives_interruption_before_playback(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    interrupted, audio = _start_reply(session, model)
    await interrupted.interrupt()
    audio.close()

    reply = await _finish_reply(session, *_start_reply(session, model))
    assert reply.metrics["e2e_latency"] == pytest.approx(reply.metrics["started_speaking_at"] - now)


async def test_typed_realtime_request_discards_pending_speech_timing(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    requested = session.generate_reply(user_input="Hello", input_modality="text")
    for _ in range(100):
        if model.active_session._reply_futs:
            break
        await asyncio.sleep(0)
    assert model.active_session._reply_futs

    reply = await _finish_reply(session, *_start_reply(session, model, requested=requested))
    assert "e2e_latency" not in reply.metrics
    follow_up = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in follow_up.metrics


async def test_realtime_response_id_is_available_on_assistant_message(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    reply = await _finish_reply(
        session, *_start_reply(session, model, response_id="provider-response-id")
    )
    assert reply.metrics["provider_request_ids"] == ["provider-response-id"]


async def _transcribed_user_messages(
    *transcripts: llm.InputTranscriptionCompleted,
) -> list[llm.ChatMessage]:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    conversation_events: list[ConversationItemAddedEvent] = []

    async with AgentSession(llm=model) as session:
        session.on("conversation_item_added", conversation_events.append)
        await session.start(Agent(instructions="test"))

        for transcript in transcripts:
            model.active_session.emit("input_audio_transcription_completed", transcript)
        await asyncio.sleep(0)

    return [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "user"
    ]


async def test_realtime_late_transcripts_keep_their_own_turn_start() -> None:
    # the provider withholds a transcript until its reply is done generating
    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1", transcript="first", is_final=True, turn_started_at=1_000_000.0
        ),
        llm.InputTranscriptionCompleted(
            item_id="item_2", transcript="second", is_final=True, turn_started_at=1_005_000.0
        ),
    )

    assert [msg.created_at for msg in messages] == [1_000_000.0, 1_005_000.0]
    assert [msg.metrics["started_speaking_at"] for msg in messages] == [1_000_000.0, 1_005_000.0]


async def test_realtime_user_message_falls_back_to_delivery_time() -> None:
    before = time.time()

    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1", transcript="text injected turn", is_final=True
        )
    )

    assert len(messages) == 1
    assert before <= messages[0].created_at <= time.time()
    assert "started_speaking_at" not in messages[0].metrics


async def test_realtime_user_turn_is_ordered_before_the_reply_it_prompted() -> None:
    """A duplex model answers over the caller, so the transcript arrives after its own reply."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async with AgentSession(llm=model) as session:
        agent = Agent(instructions="test")
        await session.start(agent)

        agent._chat_ctx.add_message(
            role="assistant", content="answering", id="reply", created_at=1_000_500.0
        )

        model.active_session.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id="user_turn",
                transcript="what is my name?",
                is_final=True,
                turn_started_at=1_000_000.0,
            ),
        )
        await asyncio.sleep(0)

        ids = [item.id for item in agent.chat_ctx.items if item.type == "message"]
        assert ids.index("user_turn") < ids.index("reply")


_REALTIME_AUDIO_REDACTION_WARNING = (
    "RealtimeModel user turns lack complete speech timestamps, so audio redaction may be "
    "inaccurate; disable audio recording to prevent redaction leak."
)


@pytest.mark.parametrize(
    ("project_redaction", "session_redaction", "audio_recording", "stt", "server_td", "warns"),
    [
        (True, False, True, False, True, True),
        (False, True, True, False, True, True),
        (True, False, True, True, True, True),
        (True, False, True, False, False, True),
        (False, False, True, False, True, False),
        (True, False, False, False, True, False),
    ],
    ids=[
        "project",
        "session",
        "with-stt",
        "without-server-turn-detection",
        "redaction-disabled",
        "audio-recording-disabled",
    ],
)
async def test_realtime_audio_redaction_warns_only_for_redacted_audio_recording(
    caplog: pytest.LogCaptureFixture,
    project_redaction: bool,
    session_redaction: bool,
    audio_recording: bool,
    stt: bool,
    server_td: bool,
    warns: bool,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(turn_detection=server_td, audio_output=False)
    )
    job_ctx = MagicMock()
    job_ctx.job.enable_recording = True
    job_ctx.job.enable_redaction = project_redaction
    job_ctx.job.id = "test-job-id"
    job_ctx.job.room.sid = "test-room-sid"
    job_ctx.job.agent_name = "test-agent"
    job_ctx.room.name = "test-room"
    job_ctx._primary_agent_session = None
    job_ctx.session_directory = Path("/tmp/test-session")

    with (
        caplog.at_level(logging.WARNING, logger="livekit.agents"),
        patch("livekit.agents.voice.agent_session.get_job_context", return_value=job_ctx),
    ):
        async with AgentSession(
            llm=model,
            stt=FakeSTT() if stt else None,
            vad=None,
            turn_handling={"turn_detection": None},
        ) as session:
            await session.start(
                Agent(instructions="test"),
                record={"audio": audio_recording, "redaction": session_redaction},
            )

    assert (_REALTIME_AUDIO_REDACTION_WARNING in caplog.messages) is warns


async def test_realtime_audio_redaction_warning_is_emitted_once_per_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        async with AgentSession(
            llm=model, vad=None, turn_handling={"turn_detection": None}
        ) as session:
            await session.start(
                Agent(instructions="first"),
                record={"redaction": True},
            )
            session.update_agent(Agent(instructions="second"))
            assert session._update_activity_atask is not None
            await session._update_activity_atask

    assert caplog.messages.count(_REALTIME_AUDIO_REDACTION_WARNING) == 1


async def test_usage_reported_as_the_session_closes_is_still_collected() -> None:
    """A session-billed model only learns its final usage as the connection drains."""
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))
        model.active_session.closing_metrics = RealtimeModelMetrics(
            request_id="sess_1",
            timestamp=time.time(),
            session_duration=12.5,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(audio_tokens=90),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(audio_tokens=40),
            metadata=Metadata(model_name="fake-live", model_provider="fake.provider"),
        )

    (usage,) = [u for u in session.usage.model_usage if u.provider == "fake.provider"]
    assert usage.input_tokens == 100
    assert usage.input_audio_tokens == 90
    assert usage.output_tokens == 50
    assert usage.output_audio_tokens == 40
    assert usage.session_duration == 12.5
