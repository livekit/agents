"""The shape of a session's span tree, as a backend sees it."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent
from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60


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


@pytest.fixture
def in_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    # the fake session runs without a JobContext, so there is no room sid to read
    monkeypatch.setattr(gen_ai, "_conversation_id", lambda: "RM_test")


@pytest.fixture
async def spans(span_exporter: InMemorySpanExporter, in_conversation: None) -> list[ReadableSpan]:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "my policy number is 12345", stt_delay=0.2)
    actions.add_llm("Thank you, I have that.", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)
    actions.add_user_speech(6.0, 8.0, "the car is a blue sedan", stt_delay=0.2)
    actions.add_llm("Noted, a blue sedan.", ttft=0.1, duration=0.3)
    actions.add_tts(2.0, ttfb=0.2, duration=0.3)

    session = create_session(actions)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="take a claim")), SESSION_TIMEOUT
    )
    await session.aclose()
    return list(span_exporter.get_finished_spans())


def _by_name(spans: list[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [span for span in spans if span.name == name]


def _operation(span: ReadableSpan) -> str | None:
    return (span.attributes or {}).get("gen_ai.operation.name")


async def test_the_session_is_one_trace_with_one_root(spans: list[ReadableSpan]) -> None:
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1

    roots = [span for span in spans if span.parent is None]
    assert [span.name for span in roots] == ["agent_session"]
    assert _operation(roots[0]) == "invoke_workflow"


async def test_one_inference_span_per_llm_call(spans: list[ReadableSpan]) -> None:
    inference = [span for span in spans if _operation(span) == "chat"]
    assert [span.name for span in inference] == ["llm_request", "llm_request"]

    # the wrappers stay in the tree, they just do not claim the operation
    assert _by_name(spans, "llm_node")
    for span in _by_name(spans, "llm_node"):
        assert "gen_ai.usage.input_tokens" not in (span.attributes or {})


async def test_every_span_is_grouped_into_the_session(spans: list[ReadableSpan]) -> None:
    missing = [
        span.name for span in spans if "gen_ai.conversation.id" not in (span.attributes or {})
    ]
    assert not missing


async def test_the_speech_turns_are_genai_operations(spans: list[ReadableSpan]) -> None:
    assert {_operation(span) for span in _by_name(spans, "user_turn")} == {"transcribe"}
    assert {_operation(span) for span in _by_name(spans, "tts_node")} == {"synthesize"}


async def test_the_speech_turns_report_their_text(spans: list[ReadableSpan]) -> None:
    """The STT transcript and the words handed to the TTS, in the GenAI content
    attributes a backend renders — not only in the `lk.pii.*` ones."""

    def _contents(span: ReadableSpan, attribute: str) -> list[str]:
        raw = (span.attributes or {}).get(attribute)
        return (
            [part["content"] for message in json.loads(raw) for part in message["parts"]]
            if raw
            else []
        )

    said = [
        text
        for span in _by_name(spans, "user_turn")
        for text in _contents(span, "gen_ai.output.messages")
    ]
    assert said == ["my policy number is 12345", "the car is a blue sedan"]

    spoken = [
        text
        for span in _by_name(spans, "tts_node")
        for text in _contents(span, "gen_ai.input.messages")
    ]
    assert "".join(spoken) == "Thank you, I have that.Noted, a blue sedan."


async def test_each_turn_reports_its_own_input_and_output(spans: list[ReadableSpan]) -> None:
    turns = _by_name(spans, "agent_turn")
    assert len(turns) == 2

    for span in turns:
        attributes = span.attributes or {}
        assert _operation(span) == "invoke_agent"
        assert json.loads(attributes["gen_ai.input.messages"])
        assert json.loads(attributes["gen_ai.output.messages"])

    first = json.loads((turns[0].attributes or {})["gen_ai.input.messages"])
    assert first[0]["parts"][0]["content"] == "my policy number is 12345"


async def test_the_root_carries_the_whole_conversation(spans: list[ReadableSpan]) -> None:
    attributes = _by_name(spans, "agent_session")[0].attributes or {}

    contents = [
        part["content"]
        for message in json.loads(attributes["gen_ai.input.messages"])
        for part in message["parts"]
    ]
    assert contents == [
        "my policy number is 12345",
        "Thank you, I have that.",
        "the car is a blue sedan",
    ]

    output = json.loads(attributes["gen_ai.output.messages"])
    assert output[0]["role"] == "assistant"
    assert output[0]["parts"][0]["content"] == "Noted, a blue sedan."


async def test_tokens_sum_once_over_the_inference_spans(spans: list[ReadableSpan]) -> None:
    totals = [
        (span.attributes or {}).get("gen_ai.usage.input_tokens")
        for span in spans
        if _operation(span) == "chat"
    ]
    assert len(totals) == 2

    everything_claiming_usage = [
        span.name for span in spans if "gen_ai.usage.input_tokens" in (span.attributes or {})
    ]
    assert everything_claiming_usage == ["llm_request", "llm_request"]


async def test_content_limits_apply_to_turns_but_not_to_the_session(
    span_exporter: InMemorySpanExporter,
) -> None:
    gen_ai.set_capture_system_instructions(False)
    gen_ai.set_max_input_messages(1)
    try:
        actions = FakeActions()
        actions.add_user_speech(0.5, 2.5, "first", stt_delay=0.2)
        actions.add_llm("one", ttft=0.1, duration=0.3)
        actions.add_tts(2.0, ttfb=0.2, duration=0.3)
        actions.add_user_speech(6.0, 8.0, "second", stt_delay=0.2)
        actions.add_llm("two", ttft=0.1, duration=0.3)
        actions.add_tts(2.0, ttfb=0.2, duration=0.3)

        session = create_session(actions)
        await asyncio.wait_for(
            run_session(session, Agent(instructions="take a claim")), SESSION_TIMEOUT
        )
        await session.aclose()
    finally:
        gen_ai.set_capture_system_instructions(True)
        gen_ai.set_max_input_messages(0)

    spans = list(span_exporter.get_finished_spans())

    for span in [span for span in spans if _operation(span) == "chat"]:
        attributes = span.attributes or {}
        assert "gen_ai.system_instructions" not in attributes
        assert len(json.loads(attributes["gen_ai.input.messages"])) == 1

    truncated = [
        span.name for span in spans if "lk.gen_ai.input.messages_dropped" in (span.attributes or {})
    ]
    assert "llm_request" in truncated

    root = _by_name(spans, "agent_session")[0].attributes or {}
    assert len(json.loads(root["gen_ai.input.messages"])) == 3
    assert "lk.gen_ai.input.messages_dropped" not in root
