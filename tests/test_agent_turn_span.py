"""One ``agent_turn`` span per speech handle.

A reply that calls a tool runs two generations (LLM steps) in two tasks; they used to be two
``agent_turn`` spans linked only by ``lk.parent_generation_id``. The speech handle now owns a
single span for its whole life: each generation is an event on it, tool and inference spans
nest under it, and it ends with the speech."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer

from .fake_session import FakeActions, create_session, run_session

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


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _children(
    exporter: InMemorySpanExporter, parent: ReadableSpan, name: str
) -> list[ReadableSpan]:
    return [
        s
        for s in _spans(exporter, name)
        if s.parent is not None and s.parent.span_id == parent.context.span_id
    ]


class _WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        return f"sunny in {location}"


async def test_tool_call_is_one_agent_turn(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, "What's the weather in Tokyo?")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_llm(content="It is sunny in Tokyo.", input="sunny in Tokyo")
    actions.add_tts(1.0)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(run_session(session, _WeatherAgent(), drain_delay=1.0), timeout=60)

    [root] = _spans(span_exporter, "agent_session")
    turns = _spans(span_exporter, "agent_turn")
    assert len(turns) == 1, [(t.attributes or {}).get(trace_types.ATTR_SPEECH_ID) for t in turns]
    [turn] = turns
    assert turn.parent is not None and turn.parent.span_id == root.context.span_id

    attrs = turn.attributes or {}
    speech_id = attrs[trace_types.ATTR_SPEECH_ID]
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 2
    assert attrs[trace_types.ATTR_AGENT_TURN_ID] == f"{speech_id}_2"
    generations = [e for e in turn.events if e.name == "generation"]
    assert [(e.attributes or {})[trace_types.ATTR_AGENT_TURN_ID] for e in generations] == [
        f"{speech_id}_1",
        f"{speech_id}_2",
    ]
    assert trace_types.ATTR_AGENT_PARENT_TURN_ID not in (generations[0].attributes or {})
    assert (generations[1].attributes or {})[trace_types.ATTR_AGENT_PARENT_TURN_ID] == (
        f"{speech_id}_1"
    )

    # both generations' inference, the tool between them, and the speech all nest under it
    assert len(_children(span_exporter, turn, "llm_node")) == 2
    [tool] = _children(span_exporter, turn, "function_tool")
    [tts] = _children(span_exporter, turn, "tts_node")
    [speaking] = _children(span_exporter, turn, "agent_speaking")
    assert tool.start_time < tts.start_time
    # and the turn covers everything, ending with the speech rather than with the first step
    for child in (tool, tts, speaking):
        assert child.end_time is not None and turn.end_time is not None
        assert turn.start_time <= child.start_time and child.end_time <= turn.end_time


async def test_plain_reply_is_one_generation(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello", stt_delay=0.1)
    actions.add_llm("Hi there", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)

    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="test"), drain_delay=1.0), timeout=60
    )

    [turn] = _spans(span_exporter, "agent_turn")
    attrs = turn.attributes or {}
    assert attrs[trace_types.ATTR_GENERATION_COUNT] == 1
    assert attrs[trace_types.ATTR_AGENT_TURN_ID] == f"{attrs[trace_types.ATTR_SPEECH_ID]}_1"
    assert len([e for e in turn.events if e.name == "generation"]) == 1
    assert trace_types.ATTR_AGENT_PARENT_TURN_ID not in attrs
