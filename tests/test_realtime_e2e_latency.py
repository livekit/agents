import asyncio
import time
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils, vad
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.events import MetricsCollectedEvent, SpeechCreatedEvent

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


@pytest.fixture
async def realtime() -> AsyncIterator[tuple[AgentSession, FakeRealtimeModel]]:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=True))
    async with AgentSession(llm=model, vad=FakeVAD()) as session:
        session.output.audio = FakeAudioOutput()
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
    session: AgentSession, model: FakeRealtimeModel, *, requested: SpeechHandle | None = None
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
    messages.close()
    functions.close()
    speeches: list[SpeechCreatedEvent] = []
    session.on("speech_created", speeches.append)
    event = llm.GenerationCreatedEvent(
        message_stream=messages,
        function_stream=functions,
        user_initiated=requested is not None,
        response_id=utils.shortuuid("response_"),
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


async def test_realtime_e2e_does_not_use_cleared_user_speech(
    realtime: tuple[AgentSession, FakeRealtimeModel],
) -> None:
    session, model = realtime
    now = time.time()
    await _vad(session, vad.VADEventType.START_OF_SPEECH, at=now - 1.0)
    await _vad(session, vad.VADEventType.END_OF_SPEECH, at=now)
    session.clear_user_turn()

    reply = await _finish_reply(session, *_start_reply(session, model))
    assert "e2e_latency" not in reply.metrics


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
