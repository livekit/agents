"""RecordingOptions.input_delta: each LLM span continues the input recorded for the last
committed generation, recording what follows the prefix they share, and the instructions
only when they differ."""

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


NODE = gen_ai.INPUT_DELTA_SITE_LLM_NODE
REQUEST = gen_ai.INPUT_DELTA_SITE_LLM_REQUEST
INSTR_ID = "lk.agent_task.instructions"
INSTR = (INSTR_ID, "system", "be brief")
INSTR2 = (INSTR_ID, "system", "be detailed")


def _call(item_id: str, call_id: str) -> llm.FunctionCall:
    return llm.FunctionCall(id=item_id, call_id=call_id, name="f", arguments="{}")


def _output(item_id: str, call_id: str) -> llm.FunctionCallOutput:
    return llm.FunctionCallOutput(
        id=item_id, call_id=call_id, name="f", output="ok", is_error=False
    )


def _ctx(*items: tuple[str, str, str] | llm.ChatItem) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    for item in items:
        if isinstance(item, tuple):
            item_id, role, text = item
            ctx.add_message(role=role, content=text, id=item_id)  # type: ignore[arg-type]
        else:
            ctx.items.append(item)
    return ctx


def _delta(
    scope: gen_ai.InputDeltaScope, ctx: llm.ChatContext, site: gen_ai.InputDeltaSite = REQUEST
) -> tuple[gen_ai.InputDelta, trace.Span]:
    with tracer.start_as_current_span(site.name) as span:
        return scope.delta(site, ctx, span), span


def _committed(tracker: gen_ai.InputDeltaTracker, ctx: llm.ChatContext) -> dict[str, Any]:
    """Record ``ctx`` at both sites as one committed generation."""
    scope = tracker.begin()
    spans = {site.name: _delta(scope, ctx, site)[1] for site in (NODE, REQUEST)}
    scope.commit()
    return spans


def _ids(items: list[Any]) -> list[str]:
    return [item.id for item in items]


def test_instructions_are_only_the_instructions_message() -> None:
    ctx = _ctx(INSTR, ("u1", "user", "hi"), ("x", "system", "greet"), ("G", "system", "guide"))
    assert gen_ai.to_system_instructions(ctx) == [{"type": "text", "content": "be brief"}]
    assert [m["role"] for m in gen_ai.to_input_messages(ctx)] == ["user", "system", "system"]

    # even right after the instructions, another system message is not one of them
    ctx = _ctx(INSTR, ("G", "system", "guide"), ("u1", "user", "hi"))
    assert gen_ai.to_system_instructions(ctx) == [{"type": "text", "content": "be brief"}]

    # without an instructions message, the system message the context starts with is
    ctx = _ctx(("s", "system", "you are a bot"), ("G", "system", "guide"), ("u1", "user", "hi"))
    assert gen_ai.to_system_instructions(ctx) == [{"type": "text", "content": "you are a bot"}]
    assert [m["role"] for m in gen_ai.to_input_messages(ctx)] == ["system", "user"]


def test_first_generation_is_full(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    ctx = _ctx(INSTR, ("u1", "user", "hi"))
    for site in (NODE, REQUEST):
        delta, _ = _delta(tracker.begin(), ctx, site)
        assert delta.base is None
        assert _ids(delta.chat_ctx.items) == [INSTR_ID, "u1"]
        assert _ids(delta.instructions) == [INSTR_ID]


def test_appended_turn_continues_the_parent(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    parent = _committed(tracker, _ctx(INSTR, ("u1", "user", "hi")))
    ctx = _ctx(INSTR, ("u1", "user", "hi"), ("a1", "assistant", "hello"), ("u2", "user", "bye"))

    scope = tracker.begin()
    node, _ = _delta(scope, ctx, NODE)
    assert node.base == parent["llm_node"].get_span_context()
    assert node.base_items == 2
    assert _ids(node.chat_ctx.items) == ["a1", "u2"]

    request, _ = _delta(scope, ctx, REQUEST)
    assert request.base == parent["llm_request"].get_span_context()
    assert request.base_messages == 1
    assert request.instructions == []
    assert _ids(request.conversation) == ["a1", "u2"]


def test_changed_instructions_are_recorded_again(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    _committed(tracker, _ctx(INSTR, ("u1", "user", "hi")))
    ctx = _ctx(INSTR2, ("u1", "user", "hi"), ("u2", "user", "bye"))

    scope = tracker.begin()
    # gen_ai keeps them apart: the messages still continue the parent
    request, _ = _delta(scope, ctx, REQUEST)
    assert request.base is not None and request.base_messages == 1
    assert _ids(request.instructions) == [INSTR_ID]
    assert _ids(request.conversation) == ["u2"]
    # lk.pii.chat_ctx starts with them, so nothing is shared
    node, _ = _delta(scope, ctx, NODE)
    assert node.base is None
    assert _ids(node.chat_ctx.items) == [INSTR_ID, "u1", "u2"]


def test_edit_records_from_the_edited_item(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    _committed(
        tracker, _ctx(INSTR, ("u1", "user", "hi"), ("a1", "assistant", "hey"), ("u2", "user", "x"))
    )
    ctx = _ctx(INSTR, ("u1", "user", "hi"), ("a1", "assistant", "HEY"), ("u2", "user", "x"))
    request, _ = _delta(tracker.begin(), ctx, REQUEST)
    assert request.base_messages == 1
    assert _ids(request.conversation) == ["a1", "u2"]


def test_tool_call_is_never_split_from_its_message(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    # the parent ended with an assistant message; this turn adds tool calls to it
    _committed(tracker, _ctx(INSTR, ("u1", "user", "hi"), ("a1", "assistant", "checking")))
    ctx = _ctx(
        INSTR,
        ("u1", "user", "hi"),
        ("a1", "assistant", "checking"),
        _call("fc", "c1"),
        _output("fo", "c1"),
    )
    request, _ = _delta(tracker.begin(), ctx, REQUEST)
    # "a1" and its tool call are one gen_ai message, so the cut moves before "a1"
    assert request.base_messages == 1
    assert _ids(request.conversation) == ["a1", "fc", "fo"]
    assert [m["role"] for m in request.input_messages()] == ["assistant", "tool"]


def test_uncommitted_generation_does_not_move_the_parent(
    span_exporter: InMemorySpanExporter,
) -> None:
    tracker = gen_ai.InputDeltaTracker()
    parent = _committed(tracker, _ctx(INSTR, ("u1", "user", "hi")))
    # a preemptive generation that gets discarded
    _delta(tracker.begin(), _ctx(INSTR, ("u1", "user", "hi"), ("p2", "user", "by")))

    ctx = _ctx(INSTR, ("u1", "user", "hi"), ("u2", "user", "bye"))
    request, _ = _delta(tracker.begin(), ctx)
    assert request.base == parent["llm_request"].get_span_context()
    assert _ids(request.conversation) == ["u2"]


def test_span_after_commit_becomes_the_parent(span_exporter: InMemorySpanExporter) -> None:
    tracker = gen_ai.InputDeltaTracker()
    scope = tracker.begin()
    scope.commit()  # scheduled before the llm_request span started
    _, span1 = _delta(scope, _ctx(INSTR, ("u1", "user", "hi")))

    request, _ = _delta(tracker.begin(), _ctx(INSTR, ("u1", "user", "hi"), ("u2", "user", "x")))
    assert request.base == span1.get_span_context()


def test_trackers_are_independent(span_exporter: InMemorySpanExporter) -> None:
    a, b = gen_ai.InputDeltaTracker(), gen_ai.InputDeltaTracker()
    _committed(a, _ctx(INSTR, ("u1", "user", "hi")))
    request, _ = _delta(b.begin(), _ctx(INSTR, ("u1", "user", "hi"), ("u2", "user", "x")))
    assert request.base is None


def _turn(*items: tuple[str, str, str] | llm.ChatItem, guide: bool = True) -> llm.ChatContext:
    """A turn as the framework sends it: the instructions, the agent's initial config, the
    history, then the expressive guide, added fresh at the end of every reply."""
    ctx = _ctx(*items, *([("G", "system", "markup guide")] if guide else []))
    at = 1 if ctx.items and ctx.items[0].id == INSTR_ID else 0
    ctx.items.insert(at, llm.AgentConfigUpdate(id="cfg", instructions="initial"))
    return ctx


def test_rebuilt_input_matches_what_was_sent(span_exporter: InMemorySpanExporter) -> None:
    """Following lk.input.* from each span back to a full one reproduces the input the
    model received, in order, for both gen_ai.* and lk.pii.chat_ctx."""
    a1, u1 = ("a1", "assistant", "hello"), ("u1", "user", "hi")
    a2, u2 = ("a2", "assistant", "sure"), ("u2", "user", "weather?")
    a3, u3 = ("a3", "assistant", "sunny"), ("u3", "user", "tomorrow?")
    a4, u4 = ("a4", "assistant", "rain"), ("u4", "user", "thanks")
    s = ("S", "system", "doctor has slots at 3pm")  # added through update_chat_ctx: persistent
    tool = [_call("fc", "c1"), _output("fo", "c1")]
    turns = [
        # greeting: a generate_reply(instructions=...) message, no conversation yet
        _turn(INSTR, ("x", "system", "greet the user")),
        _turn(INSTR, a1, u1),
        _turn(INSTR, a1, u1, a2, u2),
        # a tool step reuses the turn's context: the guide stays before the tool items
        _ctx(*_turn(INSTR, a1, u1, a2, u2).items, *tool),
        # a persistent system message mid-history, and a per-turn RAG assistant message
        _turn(INSTR, a1, u1, a2, u2, *tool, s, a3, u3, ("R", "assistant", "rag: docs")),
        _turn(INSTR, a1, u1, a2, u2, *tool, s, a3, u3, a4, u4, ("R2", "assistant", "rag: more")),
        # update_instructions together with a new turn
        _turn(INSTR2, a1, u1, a2, u2, *tool, s, a3, u3, a4, u4, ("a5", "assistant", "ok")),
        # an earlier message edited
        _turn(INSTR2, a1, ("u1", "user", "hey"), a2, u2, *tool, s, a3, u3, a4, u4),
        # a standalone context: no instructions message, a guide right after the system prompt
        _ctx(("sp", "system", "you are a bot"), ("G", "system", "guide"), u1),
    ]

    tracker = gen_ai.InputDeltaTracker()
    recorded: dict[int, gen_ai.InputDelta] = {}
    shape: dict[str, list[tuple[bool, bool]]] = {"llm_node": [], "llm_request": []}
    for ctx in turns:
        scope = tracker.begin()
        latest: dict[str, gen_ai.InputDelta] = {}
        for site in (NODE, REQUEST):
            delta, span = _delta(scope, ctx, site)
            recorded[span.get_span_context().span_id] = delta
            latest[site.name] = delta
            shape[site.name].append((delta.base is not None, bool(delta.instructions)))
        scope.commit()

        def parent(delta: gen_ai.InputDelta) -> gen_ai.InputDelta:
            assert delta.base is not None
            return recorded[delta.base.span_id]

        def chat_items(delta: gen_ai.InputDelta) -> list[Any]:
            if delta.base_items is None:
                return list(delta.chat_ctx.items)
            return chat_items(parent(delta))[: delta.base_items] + list(delta.chat_ctx.items)

        def messages(delta: gen_ai.InputDelta) -> list[Any]:
            if delta.base_messages is None:
                return delta.input_messages()
            return messages(parent(delta))[: delta.base_messages] + delta.input_messages()

        def instructions(delta: gen_ai.InputDelta) -> list[Any]:
            if delta.instructions or delta.base is None:
                return delta.system_instructions()
            return instructions(parent(delta))

        rebuilt = [(i.id, i._fingerprint()) for i in chat_items(latest["llm_node"])]
        assert rebuilt == [(i.id, i._fingerprint()) for i in ctx.items]
        assert messages(latest["llm_request"]) == gen_ai.to_input_messages(ctx)
        assert instructions(latest["llm_request"]) == gen_ai.to_system_instructions(ctx)
        # the instructions attribute never carries another system message
        assert len(instructions(latest["llm_request"])) == 1

    # (continues a parent, records instructions) per turn: the cases actually happened
    assert shape["llm_request"] == [
        (False, True),  # greeting
        (True, False),
        (True, False),
        (True, False),  # tool step
        (True, False),  # persistent S and a RAG message
        (True, False),
        (True, True),  # instructions changed: recorded again, messages continue
        (True, False),  # edited message
        (False, True),  # standalone context
    ]
    assert [has_base for has_base, _ in shape["llm_node"]] == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]


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
    assert attrs[1][trace_types.ATTR_INPUT_BASE_MESSAGES] == 1
    assert trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS not in attrs[1]
    assert _input_texts(second) == [
        ("assistant", "Hi there"),
        ("user", "What's the weather in Tokyo?"),
    ]
    assert [link.context.span_id for link in second.links] == [first.context.span_id]

    # the tool step holds the tool call and its output
    assert attrs[2][trace_types.ATTR_INPUT_BASE_SPAN_ID] == trace.format_span_id(
        second.context.span_id
    )
    assert attrs[2][trace_types.ATTR_INPUT_BASE_MESSAGES] == 3
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
