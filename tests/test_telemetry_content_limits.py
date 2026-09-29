from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.telemetry import gen_ai, pii, set_tracer_provider, trace_types, tracer
from livekit.agents.voice.generation import perform_llm_inference
from livekit.agents.voice.io import ModelSettings

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def capture_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen_ai, "_capture_content", True)
    monkeypatch.setattr(gen_ai, "_capture_system_instructions", True)
    monkeypatch.setattr(gen_ai, "_max_input_messages", 0)
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


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5])
@pytest.mark.parametrize("limit", [0, 1, 4])
def test_limits_keep_short_histories_and_report_drops(
    exporter: InMemorySpanExporter, count: int, limit: int
) -> None:
    gen_ai.set_max_input_messages(limit)
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": str(i)}]} for i in range(count)
    ]
    original = json.dumps(messages)
    with tracer.start_as_current_span("request") as span:
        gen_ai.set_content_attributes(span, input_messages=messages)
    [span] = exporter.get_finished_spans()
    attrs = span.attributes
    assert json.loads(attrs.get("gen_ai.input.messages", "[]")) == (
        messages[-limit:] if limit else messages
    )
    assert json.dumps(messages) == original
    dropped = max(0, count - limit) if limit else 0
    assert attrs.get("lk.gen_ai.input.messages_dropped", 0) == dropped
    if not dropped:
        assert "lk.gen_ai.input.messages_dropped" not in attrs


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gen_ai.set_max_input_messages(-1)


@pytest.mark.parametrize("exporter", [True, False], indirect=True)
@pytest.mark.parametrize("control", ["content", "instructions", "history", "delta"])
def test_export_controls_cover_legacy_payloads_before_pii_restore(
    exporter: InMemorySpanExporter, control: str
) -> None:
    if control == "content":
        gen_ai.set_capture_content(False)
    elif control == "instructions":
        gen_ai.set_capture_system_instructions(False)
    elif control == "history":
        gen_ai.set_max_input_messages(1)
    else:
        gen_ai.set_capture_input_delta(True)
    messages = gen_ai.to_speech_messages("older", role="user") + gen_ai.to_speech_messages(
        "latest", role="user"
    )
    attributes = {
        trace_types.ATTR_CHAT_CTX: "full history and instructions",
        trace_types.ATTR_INSTRUCTIONS: "instructions",
        trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS: '[{"type":"text","content":"instructions"}]',
        trace_types.ATTR_GEN_AI_INPUT_MESSAGES: json.dumps(messages),
        trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES: json.dumps(messages[-1:]),
        trace_types.ATTR_RESPONSE_TEXT: "response",
        trace_types.ATTR_FUNCTION_TOOL_ARGS: "arguments",
        trace_types.ATTR_FUNCTION_TOOL_OUTPUT: "result",
        trace_types.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS: "arguments",
        trace_types.ATTR_GEN_AI_TOOL_CALL_RESULT: "result",
        "gen_ai.usage.input_tokens": 42,
    }
    with tracer.start_as_current_span("request", attributes=attributes) as span:
        span.add_event("custom", attributes)
        span.add_event(trace_types.EVENT_GEN_AI_USER_MESSAGE, {"content": "old event"})
    [span] = exporter.get_finished_spans()
    restored = pii.restore_pii(span)
    assert trace_types.ATTR_CHAT_CTX not in restored.attributes
    assert restored.attributes["gen_ai.usage.input_tokens"] == 42
    assert [event.name for event in restored.events] == ["custom"]
    for attrs in [restored.attributes, restored.events[0].attributes]:
        assert trace_types.ATTR_CHAT_CTX not in attrs
        if control == "content":
            assert attrs == {"gen_ai.usage.input_tokens": 42}
        elif control == "instructions":
            assert trace_types.ATTR_INSTRUCTIONS not in attrs
            assert trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS not in attrs
            assert json.loads(attrs[trace_types.ATTR_GEN_AI_INPUT_MESSAGES]) == messages
        elif control == "history":
            assert json.loads(attrs[trace_types.ATTR_GEN_AI_INPUT_MESSAGES]) == messages[-1:]
            assert attrs[trace_types.ATTR_GEN_AI_INPUT_MESSAGES_DROPPED] == 1
        else:
            assert json.loads(attrs[trace_types.ATTR_GEN_AI_INPUT_MESSAGES]) == messages


async def test_export_limits_do_not_change_the_model_request(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen_ai.set_capture_system_instructions(False)
    gen_ai.set_max_input_messages(1)
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="instructions")
    ctx.add_message(role="user", content="earlier")
    ctx.add_message(role="assistant", content="response")
    ctx.add_message(role="user", content="latest")
    original = ctx.items.copy()

    def node(chat_ctx: llm.ChatContext, tools: list[llm.Tool], settings: ModelSettings) -> str:
        assert chat_ctx.items == original
        return "answer"

    def forbid_legacy_payload(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy full-chat payload was constructed")

    monkeypatch.setattr(llm.ChatContext, "to_dict", forbid_legacy_payload)
    task, _ = perform_llm_inference(
        node=node, chat_ctx=ctx, tool_ctx=llm.ToolContext([]), model_settings=ModelSettings()
    )
    await task
    [span] = [span for span in exporter.get_finished_spans() if span.name == "llm_node"]
    assert trace_types.ATTR_CHAT_CTX not in span.attributes
    assert trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes
    assert json.loads(span.attributes[trace_types.ATTR_GEN_AI_INPUT_MESSAGES]) == [
        {"role": "user", "parts": [{"type": "text", "content": "latest"}]}
    ]
    assert ctx.items == original


def test_truncated_tool_result_keeps_its_identity(exporter: InMemorySpanExporter) -> None:
    gen_ai.set_max_input_messages(1)
    ctx = llm.ChatContext(
        [
            llm.FunctionCall(name="lookup", call_id="call_1", arguments="{}"),
            llm.FunctionCallOutput(name="lookup", call_id="call_1", output="found", is_error=False),
        ]
    )
    with tracer.start_as_current_span("request") as span:
        gen_ai.set_content_attributes(span, input_messages=gen_ai.to_input_messages(ctx))
    [span] = exporter.get_finished_spans()
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call_1",
                    "name": "lookup",
                    "response": "found",
                }
            ],
        }
    ]
    assert span.attributes["lk.gen_ai.input.messages_dropped"] == 1


@pytest.mark.parametrize("kind", ["custom", "llm", "fallback"])
@pytest.mark.parametrize("exporter", [True, False], indirect=True)
async def test_delta_inputs_preserve_model_requests_and_detect_changes(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    gen_ai.set_capture_input_delta(True)
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="instructions")
    first = ctx.add_message(role="user", content="first")
    model: llm.LLM = FakeLLM()
    if kind == "fallback":
        model = llm.FallbackAdapter([model])

    def forbid_legacy_payload(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy full-chat payload was constructed")

    monkeypatch.setattr(llm.ChatContext, "to_dict", forbid_legacy_payload)

    async def request() -> list[dict[str, Any]]:
        original = [item.model_dump() for item in ctx.items]
        before = len(exporter.get_finished_spans())

        def node(
            chat_ctx: llm.ChatContext, tools: list[llm.Tool], settings: ModelSettings
        ) -> str | llm.LLMStream:
            assert chat_ctx is ctx
            assert [item.model_dump() for item in chat_ctx.items] == original
            return "answer" if kind == "custom" else model.chat(chat_ctx=chat_ctx)

        task, _ = perform_llm_inference(
            node=node, chat_ctx=ctx, tool_ctx=llm.ToolContext([]), model_settings=ModelSettings()
        )
        await task
        assert [item.model_dump() for item in ctx.items] == original
        spans = [pii.restore_pii(span) for span in exporter.get_finished_spans()[before:]]
        assert all(trace_types.ATTR_CHAT_CTX not in span.attributes for span in spans)
        [inference] = [
            span for span in spans if span.attributes.get("gen_ai.operation.name") == "chat"
        ]
        assert inference.attributes["lk.gen_ai.input.messages_mode"] == "delta"
        assert json.loads(inference.attributes["gen_ai.system_instructions"]) == [
            {"type": "text", "content": "instructions"}
        ]
        return json.loads(inference.attributes["gen_ai.input.messages"])

    async with model:
        assert await request() == gen_ai.to_speech_messages("first", role="user")
        ctx.add_message(role="assistant", content="answer")
        ctx.add_message(role="user", content="next")
        assert await request() == (
            gen_ai.to_speech_messages("answer", role="assistant")
            + gen_ai.to_speech_messages("next", role="user")
        )
        tool_items = [
            llm.FunctionCall(name="lookup", call_id="call_1", arguments="{}"),
            llm.FunctionCallOutput(name="lookup", call_id="call_1", output="found", is_error=False),
        ]
        ctx.items.extend(tool_items)
        assert await request() == gen_ai.to_input_messages(llm.ChatContext(tool_items))
        assert await request() == []
        first.content = ["corrected"]
        assert await request() == gen_ai.to_speech_messages("corrected", role="user")
        ctx.add_message(role="user", content="corrected")
        assert await request() == gen_ai.to_speech_messages("corrected", role="user")
        # Removing old history must not make the retained messages appear new.
        ctx.items = ctx.items[:1] + ctx.items[-1:]
        assert await request() == []
        # A separate standalone context starts a new baseline, even with the same IDs.
        ctx = ctx.copy()
        assert await request() == gen_ai.to_speech_messages("corrected", role="user")


@pytest.mark.parametrize("delta", [True, False])
@pytest.mark.parametrize("custom_node", [True, False])
async def test_session_delta_survives_context_copies_and_isolates_sessions(
    exporter: InMemorySpanExporter, delta: bool, custom_node: bool
) -> None:
    gen_ai.set_capture_input_delta(delta)
    shared = llm.ChatContext.empty()
    shared.add_message(role="user", content="earlier")
    shared.add_message(role="assistant", content="reply")

    class TestAgent(Agent):
        async def llm_node(
            self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
        ) -> str:
            assert chat_ctx.items[1].text_content == "earlier"
            return "answer"

    async def run() -> None:
        model = FakeLLM(
            fake_responses=[
                FakeLLMResponse(input=text, content="answer", ttft=0, duration=0)
                for text in ("first", "second")
            ]
        )
        async with AgentSession(llm=model) as session:
            cls = TestAgent if custom_node else Agent
            await session.start(cls(instructions="test", chat_ctx=shared.copy()))
            await session.generate_reply(user_input="first")
            await session.generate_reply(user_input="second")

    await asyncio.wait_for(asyncio.gather(run(), run()), 10)
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "chat"
    ]
    assert len(spans) == 4
    by_trace: dict[int, list[ReadableSpan]] = {}
    for span in spans:
        by_trace.setdefault(span.context.trace_id, []).append(span)
    assert len(by_trace) == 2
    initial = gen_ai.to_input_messages(shared)
    for first, second in by_trace.values():
        assert json.loads(first.attributes["gen_ai.input.messages"]) == (
            initial + gen_ai.to_speech_messages("first", role="user")
        )
        new = gen_ai.to_speech_messages("answer", role="assistant") + gen_ai.to_speech_messages(
            "second", role="user"
        )
        assert json.loads(second.attributes["gen_ai.input.messages"]) == (
            new if delta else initial + gen_ai.to_speech_messages("first", role="user") + new
        )


def test_delta_limit_and_disabled_capture_do_not_consume_unrecorded_history(
    exporter: InMemorySpanExporter,
) -> None:
    gen_ai.set_capture_input_delta(True)
    gen_ai.set_max_input_messages(1)
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="earlier")
    ctx.add_message(role="user", content="latest")
    gen_ai.set_capture_content(False)
    with tracer.start_as_current_span("disabled") as span:
        gen_ai.record_llm_input_messages(span, ctx)
    gen_ai.set_capture_content(True)
    with tracer.start_as_current_span("enabled") as span:
        gen_ai.record_llm_input_messages(span, ctx)
    with tracer.start_as_current_span("unchanged") as span:
        gen_ai.record_llm_input_messages(span, ctx)
    disabled, enabled, unchanged = exporter.get_finished_spans()
    assert "gen_ai.input.messages" not in disabled.attributes
    assert json.loads(enabled.attributes["gen_ai.input.messages"]) == gen_ai.to_speech_messages(
        "latest", role="user"
    )
    assert enabled.attributes["lk.gen_ai.input.messages_dropped"] == 1
    assert json.loads(unchanged.attributes["gen_ai.input.messages"]) == []


def test_resuming_capture_resets_delta_after_a_suppressed_export(
    exporter: InMemorySpanExporter,
) -> None:
    gen_ai.set_capture_input_delta(True)
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="first")
    with tracer.start_as_current_span("suppressed") as span:
        gen_ai.record_llm_input_messages(span, ctx)
        gen_ai.set_capture_content(False)
    gen_ai.set_capture_content(True)
    with tracer.start_as_current_span("resumed") as span:
        gen_ai.record_llm_input_messages(span, ctx)
    suppressed, resumed = exporter.get_finished_spans()
    assert "gen_ai.input.messages" not in suppressed.attributes
    assert json.loads(resumed.attributes["gen_ai.input.messages"]) == gen_ai.to_speech_messages(
        "first", role="user"
    )
