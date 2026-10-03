"""RecordingOptions.input_delta: LLM spans record only what changed since the last
committed generation — the new conversation items, and the system instructions only when
they differ."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, AgentSession, RunContext, function_tool, llm
from livekit.agents.llm import FunctionToolCall
from livekit.agents.telemetry import gen_ai, set_tracer_provider, trace_types, tracer

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

SITE = gen_ai.INPUT_DELTA_SITE_LLM_REQUEST


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


def _ctx(*items: tuple[str, str, str]) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    for item_id, role, text in items:
        ctx.add_message(role=role, content=text, id=item_id)  # type: ignore[arg-type]
    return ctx


def _delta(
    scope: gen_ai.InputDeltaScope, ctx: llm.ChatContext
) -> tuple[gen_ai.InputDelta, trace.Span]:
    with tracer.start_as_current_span("llm_request") as span:
        return scope.delta(SITE, ctx, span), span


def _ids(delta: gen_ai.InputDelta) -> list[str]:
    return [item.id for item in delta.chat_ctx.items]


SYS = ("sys", "system", "be nice")


def test_first_generation_is_full(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    ctx = _ctx(SYS, ("u1", "user", "hi"))
    delta, _ = _delta(tracker.begin(), ctx)
    assert delta.chat_ctx is ctx
    assert delta.messages_base is None and delta.instructions_base is None


def test_appended_items_are_a_delta(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    _, span1 = _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))
    scope1.commit()

    ctx2 = _ctx(SYS, ("u1", "user", "hi"), ("a1", "assistant", "hello"), ("u2", "user", "bye"))
    delta, _ = _delta(tracker.begin(), ctx2)
    # instructions are unchanged, so only the previous agent turn and the new user turn
    assert _ids(delta) == ["a1", "u2"]
    assert delta.messages_base is not None
    assert [item_id for item_id, _ in delta.messages_base.item_keys] == ["u1"]
    assert delta.messages_base.span_context == span1.get_span_context()
    assert delta.instructions_base is not None
    assert delta.instructions_base.span_context == span1.get_span_context()


def test_changed_instructions_are_recorded_with_the_delta(
    span_exporter: InMemorySpanExporter,
) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))
    scope1.commit()

    ctx2 = _ctx(("sys", "system", "be terse"), ("u1", "user", "hi"), ("u2", "user", "bye"))
    delta, _ = _delta(tracker.begin(), ctx2)
    assert _ids(delta) == ["sys", "u2"]
    assert delta.messages_base is not None
    assert delta.instructions_base is None


def test_one_off_system_message_does_not_break_the_delta(
    span_exporter: InMemorySpanExporter,
) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))
    scope1.commit()

    # generate_reply(instructions=...) appends a system message after the conversation
    ctx2 = _ctx(SYS, ("u1", "user", "hi"), ("u2", "user", "bye"), ("tmp", "system", "greet"))
    delta, _ = _delta(tracker.begin(), ctx2)
    assert delta.messages_base is not None
    assert delta.instructions_base is None
    assert _ids(delta) == ["sys", "tmp", "u2"]


@pytest.mark.parametrize(
    "items",
    [
        [("u2", "user", "bye")],  # u1 removed
        [("u0", "user", "inserted"), ("u1", "user", "hi")],  # inserted before
    ],
    ids=["removed", "inserted"],
)
def test_manipulated_conversation_is_full(
    span_exporter: InMemorySpanExporter, items: list[tuple[str, str, str]]
) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))
    scope1.commit()

    delta, _ = _delta(tracker.begin(), _ctx(SYS, *items))
    assert delta.messages_base is None
    # the instructions are still the base's
    assert delta.instructions_base is not None
    assert _ids(delta) == [item[0] for item in items]


def test_edited_item_makes_messages_full(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    ctx = _ctx(SYS, ("u1", "user", "hi"))
    scope1 = tracker.begin()
    _delta(scope1, ctx)
    scope1.commit()

    # edited in place, keeping its id
    ctx.items[1].content = ["hello"]  # type: ignore[union-attr]
    ctx.add_message(role="user", content="bye", id="u2")
    scope2 = tracker.begin()
    delta, span2 = _delta(scope2, ctx)
    assert delta.messages_base is None
    assert delta.instructions_base is not None
    assert _ids(delta) == ["u1", "u2"]
    scope2.commit()

    # the next committed generation re-establishes the baseline
    ctx.add_message(role="user", content="again", id="u3")
    delta, _ = _delta(tracker.begin(), ctx)
    assert delta.messages_base is not None
    assert delta.messages_base.span_context == span2.get_span_context()
    assert _ids(delta) == ["u3"]


def test_copied_context_is_still_a_delta(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    ctx = _ctx(SYS, ("u1", "user", "hi"))
    scope1 = tracker.begin()
    _delta(scope1, ctx)
    scope1.commit()

    # each turn runs on a copy, and update_chat_ctx(ctx.copy()) replaces the history
    copy = ctx.copy()
    copy.add_message(role="user", content="bye", id="u2")
    delta, _ = _delta(tracker.begin(), copy)
    assert delta.messages_base is not None
    assert _ids(delta) == ["u2"]


def test_preemptive_transcript_edit_is_recorded_again(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    ctx = _ctx(SYS, ("u1", "user", "hi"))
    scope1 = tracker.begin()
    _, span1 = _delta(scope1, ctx)
    # an adopted preemptive generation's user message gets the final transcript before the
    # speech is scheduled, but span1 still contains the preliminary transcript
    ctx.items[1].content = ["Hi."]  # type: ignore[union-attr]
    scope1.commit()

    ctx.add_message(role="user", content="bye", id="u2")
    delta, _ = _delta(tracker.begin(), ctx)
    assert delta.messages_base is None
    assert delta.instructions_base is not None
    assert delta.instructions_base.span_context == span1.get_span_context()
    assert _ids(delta) == ["u1", "u2"]


def test_states_are_independent(span_exporter: InMemorySpanExporter) -> None:
    a, b = gen_ai.InputDeltaTracker(), gen_ai.InputDeltaTracker()
    scope = a.begin()
    _delta(scope, _ctx(SYS, ("u1", "user", "hi")))
    scope.commit()

    delta, _ = _delta(b.begin(), _ctx(SYS, ("u1", "user", "hi"), ("u2", "user", "bye")))
    assert delta.messages_base is None and delta.instructions_base is None


def test_uncommitted_generation_does_not_move_the_baseline(
    span_exporter: InMemorySpanExporter,
) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    _, span1 = _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))
    scope1.commit()

    # a preemptive generation that gets discarded
    _delta(tracker.begin(), _ctx(SYS, ("u1", "user", "hi"), ("p2", "user", "by")))

    delta, _ = _delta(tracker.begin(), _ctx(SYS, ("u1", "user", "hi"), ("u2", "user", "bye")))
    assert delta.messages_base is not None
    assert delta.messages_base.span_context == span1.get_span_context()
    assert _ids(delta) == ["u2"]


def test_select_after_commit_becomes_the_baseline(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    scope1.commit()  # scheduled before the llm_request span started
    _, span1 = _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))

    delta, _ = _delta(tracker.begin(), _ctx(SYS, ("u1", "user", "hi"), ("u2", "user", "bye")))
    assert delta.messages_base is not None
    assert delta.messages_base.span_context == span1.get_span_context()


def test_instructions_base_points_at_the_span_that_recorded_them(
    span_exporter: InMemorySpanExporter,
) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope1 = tracker.begin()
    _, span1 = _delta(scope1, _ctx(SYS, ("u1", "user", "hi")))
    scope1.commit()
    scope2 = tracker.begin()
    _delta(scope2, _ctx(SYS, ("u1", "user", "hi"), ("u2", "user", "bye")))
    scope2.commit()

    delta, _ = _delta(tracker.begin(), _ctx(SYS, ("u1", "user", "hi"), ("u2", "user", "bye")))
    assert delta.instructions_base is not None
    assert delta.instructions_base.span_context == span1.get_span_context()


# -- session ---------------------------------------------------------------------------


class _WeatherAgent(Agent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions="You are a helpful assistant.", **kwargs)

    @function_tool
    async def get_weather(self, context: RunContext, location: str) -> str:
        return f"sunny in {location}"


def _llm_requests(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    spans = [s for s in exporter.get_finished_spans() if s.name == "llm_request"]
    return sorted(spans, key=lambda s: s.start_time or 0)


def _input_texts(span: ReadableSpan) -> list[Any]:
    messages = json.loads((span.attributes or {})[trace_types.ATTR_GEN_AI_INPUT_MESSAGES])
    return [(m["role"], m["parts"][0].get("content", m["parts"][0]["type"])) for m in messages]


def _two_turns_with_a_tool() -> FakeActions:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello", stt_delay=0.1)
    actions.add_llm("Hi there", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)
    actions.add_user_speech(4.0, 5.0, "What's the weather in Tokyo?", stt_delay=0.1)
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name="get_weather", arguments='{"location": "Tokyo"}', call_id="1")
        ],
    )
    actions.add_llm(content="It is sunny in Tokyo.", input="sunny in Tokyo")
    actions.add_tts(1.0)
    return actions


async def test_session_records_incremental_input(span_exporter: InMemorySpanExporter) -> None:
    session = create_session(_two_turns_with_a_tool(), speed_factor=2.0)
    await asyncio.wait_for(
        run_session(
            session,
            _WeatherAgent(),
            drain_delay=1.0,
            record={"traces": True, "input_delta": True},
        ),
        timeout=60,
    )

    first, second, tool_step = _llm_requests(span_exporter)
    attrs = [s.attributes or {} for s in (first, second, tool_step)]

    # the first generation is recorded in full
    assert trace_types.ATTR_INPUT_DELTA not in attrs[0]
    assert trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS in attrs[0]
    assert _input_texts(first) == [("user", "Hello")]

    # the second holds the previous agent turn and the new user turn
    assert attrs[1][trace_types.ATTR_INPUT_DELTA] is True
    assert attrs[1][trace_types.ATTR_INPUT_BASE_SPAN_ID] == trace.format_span_id(
        first.context.span_id
    )
    assert trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS not in attrs[1]
    assert attrs[1][trace_types.ATTR_INPUT_INSTRUCTIONS_BASE_SPAN_ID] == trace.format_span_id(
        first.context.span_id
    )
    assert _input_texts(second) == [
        ("assistant", "Hi there"),
        ("user", "What's the weather in Tokyo?"),
    ]
    assert [link.context.span_id for link in second.links] == [first.context.span_id]

    # the tool step holds the tool call and its output
    assert attrs[2][trace_types.ATTR_INPUT_BASE_SPAN_ID] == trace.format_span_id(
        second.context.span_id
    )
    assert [role for role, _ in _input_texts(tool_step)] == ["assistant", "tool"]


async def test_session_without_input_delta_records_full_input(
    span_exporter: InMemorySpanExporter,
) -> None:
    session = create_session(_two_turns_with_a_tool(), speed_factor=2.0)
    await asyncio.wait_for(
        run_session(session, _WeatherAgent(), drain_delay=1.0, record={"traces": True}),
        timeout=60,
    )

    spans = _llm_requests(span_exporter)
    assert len(spans) == 3
    for span in spans:
        attrs = span.attributes or {}
        assert not any(key.startswith("lk.input.") for key in attrs)
        assert trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS in attrs
    assert _input_texts(spans[1])[0] == ("user", "Hello")


async def test_fallback_records_input_on_the_provider_span_only(
    span_exporter: InMemorySpanExporter,
) -> None:
    session: AgentSession = create_session(_two_turns_with_a_tool(), speed_factor=2.0)
    assert isinstance(session.llm, llm.LLM)
    agent = _WeatherAgent(llm=llm.FallbackAdapter([session.llm]))
    await asyncio.wait_for(
        run_session(session, agent, drain_delay=1.0, record={"traces": True, "input_delta": True}),
        timeout=60,
    )

    requests = _llm_requests(span_exporter)
    with_input = [
        s for s in requests if trace_types.ATTR_GEN_AI_INPUT_MESSAGES in (s.attributes or {})
    ]
    assert len(with_input) == 3
    # the deltas chain through the provider spans, never through the fallback wrapper
    for span in with_input:
        assert (span.attributes or {})[trace_types.ATTR_GEN_AI_OPERATION_NAME] == "chat"
    assert (with_input[1].attributes or {})[trace_types.ATTR_INPUT_BASE_SPAN_ID] == (
        trace.format_span_id(with_input[0].context.span_id)
    )
