"""What a realtime turn puts on its spans: the conversation on ``realtime_inference``, the
caller's own input on ``agent_turn``, and the caller's transcript on a ``user_turn`` span the
model's server-side turn detection would otherwise leave unrecorded."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


def _span(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert spans, f"no {name} span, got {sorted({s.name for s in exporter.get_finished_spans()})}"
    return spans[0]


async def _run_one_generation(session: AgentSession, model: FakeRealtimeModel) -> None:
    """Drive one assistant generation from the reply request to the closed streams."""
    for _ in range(50):
        if model.active_session._reply_futs:
            break
        await asyncio.sleep(0)
    assert model.active_session._reply_futs, "the session never asked the model for a reply"

    message_ch = utils.aio.Chan[llm.MessageGeneration]()
    function_ch = utils.aio.Chan[llm.FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])

    message_ch.send_nowait(
        llm.MessageGeneration(
            message_id="msg-1",
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    function_ch.close()
    text_ch.send_nowait("the weather today is sunny")
    audio_ch.send_nowait(
        rtc.AudioFrame(
            data=b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
        )
    )
    model.active_session._reply_futs[0].set_result(
        llm.GenerationCreatedEvent(
            message_stream=message_ch,
            function_stream=function_ch,
            user_initiated=True,
        )
    )
    for _ in range(50):
        await asyncio.sleep(0)
    text_ch.close()
    audio_ch.close()
    for _ in range(200):
        await asyncio.sleep(0)


async def test_realtime_turn_records_the_conversation(span_exporter: InMemorySpanExporter) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="be concise"))
        session.generate_reply(user_input="what is the weather", instructions="answer in one line")
        await _run_one_generation(session, model)

    inference = _span(span_exporter, "realtime_inference")
    assert inference.attributes is not None
    system = json.loads(inference.attributes[trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS])
    # both what the session was given and what this one response asked for
    assert [p["content"] for p in system] == ["be concise", "answer in one line"]
    inputs = json.loads(inference.attributes[trace_types.ATTR_GEN_AI_INPUT_MESSAGES])
    assert [m["role"] for m in inputs], "the history the model was given is empty"
    outputs = json.loads(inference.attributes[trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES])
    assert outputs[0]["role"] == "assistant"
    assert "the weather today is sunny" in json.dumps(outputs)
    assert outputs[0]["finish_reason"] == "stop"

    turn = _span(span_exporter, "agent_turn")
    assert turn.attributes is not None
    assert turn.attributes[trace_types.ATTR_INSTRUCTIONS] == "answer in one line"
    assert turn.attributes[trace_types.ATTR_USER_INPUT] == "what is the weather"


async def test_realtime_transcript_opens_a_user_turn(span_exporter: InMemorySpanExporter) -> None:
    model = FakeRealtimeModel()
    started_at = time.time() - 3.0
    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        await session.start(Agent(instructions="be concise"))
        model.active_session.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id="item-1",
                transcript="what is the weather",
                is_final=True,
                confidence=0.9,
                turn_started_at=started_at,
            ),
        )
        for _ in range(20):
            await asyncio.sleep(0)

    turn = _span(span_exporter, "user_turn")
    assert turn.attributes is not None
    assert turn.attributes[trace_types.ATTR_USER_TRANSCRIPT] == "what is the weather"
    assert turn.attributes[trace_types.ATTR_TRANSCRIPT_CONFIDENCE] == 0.9
    # the realtime model is the transcriber, named the way an STT model would be
    assert turn.attributes[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == model.model
    assert turn.attributes[trace_types.ATTR_GEN_AI_PROVIDER_NAME] == model.provider
    # back-dated to where the provider says the turn began, not to where the transcript arrived
    assert turn.start_time is not None
    assert abs(turn.start_time / 1_000_000_000 - started_at) < 0.05
    assert trace_types.ATTR_USER_TURN_START_ESTIMATED not in turn.attributes


async def test_user_turn_start_is_marked_when_the_provider_gives_none(
    span_exporter: InMemorySpanExporter,
) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        await session.start(Agent(instructions="be concise"))
        model.active_session.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id="item-1", transcript="hello", is_final=True),
        )
        for _ in range(20):
            await asyncio.sleep(0)

    turn = _span(span_exporter, "user_turn")
    assert turn.attributes is not None
    assert turn.attributes[trace_types.ATTR_USER_TURN_START_ESTIMATED] is True


def _transcribe(model: FakeRealtimeModel, transcript: str, started_at: float) -> None:
    model.active_session.emit(
        "input_audio_transcription_completed",
        llm.InputTranscriptionCompleted(
            item_id=transcript, transcript=transcript, is_final=True, turn_started_at=started_at
        ),
    )


async def _commit_vad_turn(session: AgentSession) -> None:
    """Commit the open user turn the way a VAD end of speech does, and finish its reply task."""
    recognition = session._activity._audio_recognition
    assert recognition is not None
    recognition._run_eou_detection(session._activity._agent._chat_ctx, trigger="vad")
    assert recognition._end_of_turn_task is not None
    await asyncio.wait_for(recognition._end_of_turn_task, timeout=5)
    if (task := session._activity._user_turn_completed_atask) is not None:
        await asyncio.wait_for(task, timeout=5)


def _user_turns(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    spans = [s for s in exporter.get_finished_spans() if s.name == "user_turn"]
    return {str((s.attributes or {}).get(trace_types.ATTR_USER_TRANSCRIPT)): s for s in spans}


@pytest.mark.parametrize("before_commit", [True, False])
async def test_vad_turn_and_realtime_transcript_share_one_span(
    span_exporter: InMemorySpanExporter, before_commit: bool
) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(
        llm=model, vad=FakeVAD(fake_user_speeches=[]), aec_warmup_duration=None
    ) as session:
        await session.start(Agent(instructions="be concise"))
        assert session._activity._audio_recognition is not None
        speech_start = time.time() - 2.0
        session._activity._audio_recognition._ensure_user_turn_span(start_time=speech_start)
        if before_commit:
            _transcribe(model, "account balance", speech_start + 0.2)
        await _commit_vad_turn(session)
        if not before_commit:
            _transcribe(model, "account balance", speech_start + 0.2)

    # one turn, with the VAD's timing and the model's text, however the two are ordered
    assert list(_user_turns(span_exporter)) == ["account balance"]


async def test_late_transcript_lands_on_its_own_turn(span_exporter: InMemorySpanExporter) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(
        llm=model, vad=FakeVAD(fake_user_speeches=[]), aec_warmup_duration=None
    ) as session:
        await session.start(Agent(instructions="be concise"))
        recognition = session._activity._audio_recognition
        assert recognition is not None
        first_start = time.time() - 5.0
        recognition._ensure_user_turn_span(start_time=first_start)
        await _commit_vad_turn(session)
        recognition._ensure_user_turn_span(start_time=time.time())

        # the first turn's transcript arrives while the second turn is open
        _transcribe(model, "account balance", first_start + 0.2)
        await _commit_vad_turn(session)

    turns = _user_turns(span_exporter)
    assert sorted(turns) == ["None", "account balance"]
    assert turns["account balance"].start_time == int(first_start * 1_000_000_000)


@pytest.mark.parametrize("before_stop", [True, False])
async def test_model_speech_events_bound_the_user_turn(
    span_exporter: InMemorySpanExporter, before_stop: bool
) -> None:
    # gpt-live sends the final transcript before its end of speech, openai after it
    model = FakeRealtimeModel()
    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        await session.start(Agent(instructions="be concise"))
        started_at = time.time() - 2.0
        model.active_session.emit(
            "input_speech_started", llm.InputSpeechStartedEvent(started_at=started_at)
        )
        if before_stop:
            _transcribe(model, "account balance", started_at)
        model.active_session.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
        )
        if not before_stop:
            _transcribe(model, "account balance", started_at)

    turns = _user_turns(span_exporter)
    assert list(turns) == ["account balance"]
    turn = turns["account balance"]
    assert turn.start_time == int(started_at * 1_000_000_000)
    speaking = _span(span_exporter, "user_speaking")
    assert speaking.parent is not None and speaking.parent.span_id == turn.context.span_id
