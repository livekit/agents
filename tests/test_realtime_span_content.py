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
