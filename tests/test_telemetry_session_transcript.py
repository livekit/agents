from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN_CONTEXT, NonRecordingSpan

from livekit.agents import Agent, llm
from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def capture_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen_ai, "_capture_content", True)
    monkeypatch.setattr(gen_ai, "_capture_input_delta", False)
    monkeypatch.setattr(gen_ai, "_input_capture_version", 0)


@pytest.fixture
def exporter(request: pytest.FixtureRequest) -> Iterator[InMemorySpanExporter]:
    original = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider, allow_pii=getattr(request, "param", True))
    try:
        yield exporter
    finally:
        set_tracer_provider(original)
        provider.shutdown()


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("delta", [True, False])
async def test_session_close_exports_the_full_transcript(
    exporter: InMemorySpanExporter, capture: bool, delta: bool
) -> None:
    gen_ai.set_capture_content(capture)
    gen_ai.set_capture_input_delta(delta)
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello")
    actions.add_llm("Hi there")
    actions.add_tts(0.5)
    actions.add_user_speech(5.0, 6.0, "Goodbye")
    actions.add_llm("Bye")
    actions.add_tts(0.5)
    session = create_session(actions)
    await asyncio.wait_for(run_session(session, Agent(instructions="private instructions")), 60)
    [root] = [span for span in exporter.get_finished_spans() if span.name == "agent_session"]
    assert root.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert "gen_ai.input.messages" not in root.attributes
    if capture:
        expected = [
            {"role": role, "parts": [{"type": "text", "content": text}]}
            for role, text in [
                ("user", "Hello"),
                ("assistant", "Hi there"),
                ("user", "Goodbye"),
                ("assistant", "Bye"),
            ]
        ]
        assert json.loads(root.attributes["gen_ai.output.messages"]) == expected
    else:
        assert "gen_ai.output.messages" not in root.attributes


@pytest.mark.parametrize("exporter", [False], indirect=True)
def test_session_transcript_respects_pii_filtering(exporter: InMemorySpanExporter) -> None:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="private text")
    with tracer.start_as_current_span("agent_session") as span:
        gen_ai.record_session_transcript(span, ctx)
    assert "gen_ai.output.messages" not in exporter.get_finished_spans()[0].attributes


@pytest.mark.parametrize("recording,capture", [(False, True), (True, False)])
def test_disabled_session_capture_skips_transcript_construction(
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    recording: bool,
    capture: bool,
) -> None:
    gen_ai.set_capture_content(capture)

    def unexpected(*args: object) -> None:
        raise AssertionError("session transcript was constructed")

    monkeypatch.setattr(gen_ai, "to_input_messages", unexpected)
    span = (
        tracer.start_span("agent_session") if recording else NonRecordingSpan(INVALID_SPAN_CONTEXT)
    )
    gen_ai.record_session_transcript(span, llm.ChatContext.empty())
    span.end()


def test_session_transcript_has_no_default_message_limit(exporter: InMemorySpanExporter) -> None:
    ctx = llm.ChatContext.empty()
    for index in range(101):
        ctx.add_message(role="user", content=str(index))
    with tracer.start_as_current_span("agent_session") as span:
        gen_ai.record_session_transcript(span, ctx)
    [span] = exporter.get_finished_spans()
    assert len(json.loads(span.attributes["gen_ai.output.messages"])) == 101
