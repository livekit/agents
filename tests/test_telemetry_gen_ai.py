from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import llm
from livekit.agents.telemetry import gen_ai, trace_types

pytestmark = pytest.mark.unit

# The shapes asserted here follow the gen_ai.input.messages / output.messages /
# system_instructions / tool.definitions JSON schemas from
# https://github.com/open-telemetry/semantic-conventions-genai, so builder drift is
# caught here rather than by a backend that silently drops the span.


def _chat_ctx() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content=["You are a helpful agent."])
    ctx.add_message(role="user", content=["What's the weather in Paris?"])
    ctx.insert(llm.FunctionCall(call_id="call_1", name="get_weather", arguments='{"loc": "Paris"}'))
    ctx.insert(
        llm.FunctionCallOutput(
            call_id="call_1", name="get_weather", output='{"temp": 14}', is_error=False
        )
    )
    ctx.insert(llm.AgentHandoff(new_agent_id="agent-2"))
    ctx.add_message(role="assistant", content=["It's 14 degrees in Paris."])
    return ctx


def _exporting_span(name: str = "llm_request") -> tuple[Any, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__).start_span(name), exporter


def _attributes(exporter: InMemorySpanExporter) -> Any:
    return exporter.get_finished_spans()[0].attributes


@llm.function_tool
async def _get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return "sunny"


def test_builders_produce_the_conventions_shapes() -> None:
    chat_ctx = _chat_ctx()

    # instructions are reported separately from history, not as a system message
    assert gen_ai.to_system_instructions(chat_ctx) == [
        {"type": "text", "content": "You are a helpful agent."}
    ]

    messages = gen_ai.to_input_messages(chat_ctx)
    # the agent handoff is not conversational and is skipped
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["parts"][0] == {
        "type": "tool_call",
        "id": "call_1",
        "name": "get_weather",
        "arguments": {"loc": "Paris"},
    }
    # a serialized payload is deserialized, as the convention asks of instrumentations
    assert messages[2]["parts"][0]["response"] == {"temp": 14}

    call = llm.FunctionCall(call_id="call_9", name="lookup", arguments='{"q": "x"}')
    output = gen_ai.to_output_messages(text="one moment", function_calls=[call])
    assert output[0]["role"] == "assistant"
    assert [p["type"] for p in output[0]["parts"]] == ["text", "tool_call"]
    assert gen_ai.to_output_messages(text="", function_calls=[]) == []

    # `parameters` is omitted: the convention marks it NOT RECOMMENDED by default
    assert gen_ai.to_tool_definitions([_get_weather]) == [
        {
            "type": "function",
            "name": "_get_weather",
            "description": "Get the current weather in a given location.",
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, trace_types.GenAIFinishReason.STOP),
        ({"interrupted": True}, trace_types.GenAIFinishReason.ERROR),
        (
            {"function_calls": [llm.FunctionCall(call_id="c", name="n", arguments="{}")]},
            trace_types.GenAIFinishReason.TOOL_CALL,
        ),
        # a tool call emitted before the generation failed is not a successful handoff
        (
            {
                "function_calls": [llm.FunctionCall(call_id="c", name="n", arguments="{}")],
                "interrupted": True,
            },
            trace_types.GenAIFinishReason.ERROR,
        ),
    ],
)
def test_finish_reasons_use_the_conventions_values(kwargs: dict, expected: str) -> None:
    assert gen_ai.finish_reason_for(**kwargs) == expected


def test_inference_span_uses_the_registry_names() -> None:
    span, exporter = _exporting_span()
    gen_ai.set_request_attributes(
        span,
        operation=trace_types.GenAIOperationName.CHAT,
        # a LiveKit plugin id, normalized to the registry spelling
        provider="bedrock",
        model="claude-sonnet-4",
        stream=True,
    )
    gen_ai.set_response_attributes(
        span, response_id="resp_1", finish_reasons=["stop"], time_to_first_chunk=0.4
    )
    gen_ai.set_content_attributes(span, input_messages=[{"role": "user", "parts": []}])
    span.end()

    attrs = _attributes(exporter)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "aws.bedrock"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-4"
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.response.time_to_first_chunk"] == 0.4
    # attributes cannot hold structured values, so content is a JSON string
    assert json.loads(attrs["gen_ai.input.messages"]) == [{"role": "user", "parts": []}]


def test_content_capture_can_be_turned_off() -> None:
    span, exporter = _exporting_span()
    gen_ai.set_capture_content(False)
    try:
        gen_ai.set_request_attributes(
            span, operation=trace_types.GenAIOperationName.CHAT, model="gpt-4o"
        )
        gen_ai.set_content_attributes(span, input_messages=[{"role": "user", "parts": []}])
    finally:
        gen_ai.set_capture_content(True)
    span.end()

    attrs = _attributes(exporter)
    assert "gen_ai.input.messages" not in attrs
    assert attrs["gen_ai.request.model"] == "gpt-4o"


def test_usage_details_are_reported_alongside_the_totals() -> None:
    span, exporter = _exporting_span()
    gen_ai.set_usage_attributes(
        span,
        llm.CompletionUsage(
            completion_tokens=180,
            prompt_tokens=300,
            prompt_cached_tokens=40,
            cache_creation_tokens=25,
            reasoning_tokens=50,
            total_tokens=480,
        ),
    )
    span.end()

    # per the convention the detailed counts are subsets of the totals, never added
    attrs = _attributes(exporter)
    assert attrs["gen_ai.usage.input_tokens"] == 300
    assert attrs["gen_ai.usage.output_tokens"] == 180
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 40
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 25
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 50


def test_execute_tool_span() -> None:
    span, exporter = _exporting_span("function_tool")
    gen_ai.set_tool_attributes(
        span,
        name="get_weather",
        call_id="call_1",
        description="Get the weather",
        arguments='{"location": "Paris"}',
    )
    gen_ai.set_tool_result(span, result='{"temp": 14}', is_error=False)
    span.end()

    attrs = _attributes(exporter)
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "get_weather"
    assert attrs["gen_ai.tool.call.id"] == "call_1"
    assert attrs["gen_ai.tool.type"] == "function"
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"location": "Paris"}
    assert json.loads(attrs["gen_ai.tool.call.result"]) == {"temp": 14}


def test_error_type_is_low_cardinality() -> None:
    from livekit.agents import APIStatusError

    span, exporter = _exporting_span("function_tool")
    gen_ai.set_tool_result(span, result="boom", is_error=True)
    span.end()
    attrs = _attributes(exporter)
    # "the result returned by the tool call (if any and if execution was successful)"
    assert "gen_ai.tool.call.result" not in attrs
    assert attrs["error.type"] == "tool_error"

    span, exporter = _exporting_span()
    gen_ai.set_error_type(span, APIStatusError("rate limited", status_code=429))
    span.end()
    # a status code identifies the failure better than the exception class
    assert _attributes(exporter)["error.type"] == "429"


# Every `provider` value a plugin actually reports, gathered from the LLM and realtime
# plugins. The convention makes the registry spelling mandatory for a provider it
# enumerates, so a plugin returning a display name or a base-URL host must still land on it.
@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        # base-URL hosts, from the OpenAI-compatible clients
        ("api.openai.com", "openai"),
        ("api.anthropic.com", "anthropic"),
        ("api.mistral.ai", "mistral_ai"),
        ("api.groq.com", "groq"),
        ("api.x.ai", "x_ai"),
        ("api.deepseek.com", "deepseek"),
        ("my-co.openai.azure.com", "azure.ai.openai"),
        ("bedrock-runtime.us-east-1.amazonaws.com", "aws.bedrock"),
        # display names
        ("AWS Bedrock", "aws.bedrock"),
        ("Amazon", "aws.bedrock"),
        ("MistralAI", "mistral_ai"),
        ("Vertex AI", "gcp.vertex_ai"),
        ("Vertex AI Model Garden", "gcp.vertex_ai"),
        ("Gemini", "gcp.gemini"),
        ("google", "gcp.gen_ai"),
        ("Google Cloud Platform", "gcp.gen_ai"),
        ("xAI", "x_ai"),
        ("Perplexity", "perplexity"),
        ("Groq", "groq"),
        # outside the registry: the convention allows a custom value, so it passes through
        ("Baseten", "Baseten"),
        ("MiniMax", "MiniMax"),
        ("api.cerebras.ai", "api.cerebras.ai"),
    ],
)
def test_plugin_providers_resolve_to_the_registry(reported: str, expected: str) -> None:
    assert trace_types.gen_ai_provider_name(reported) == expected


def test_the_provider_tables_only_target_registry_values() -> None:
    # a typo in a mapping would emit a value no backend recognises
    targets = {
        *trace_types._PROVIDER_BY_NAME.values(),
        *trace_types._PROVIDER_BY_HOST.values(),
        *(v for _, v in trace_types._PROVIDER_BY_HOST_SUFFIX),
    }
    assert targets <= trace_types.GEN_AI_PROVIDER_NAMES


def test_tool_spans_carry_the_conversation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # execute_tool is a first-class GenAI operation; a backend that groups by
    # gen_ai.conversation.id would otherwise drop tool spans out of the session view
    monkeypatch.setattr(gen_ai, "_conversation_id", lambda: "room-42")

    span, exporter = _exporting_span("function_tool")
    gen_ai.set_tool_attributes(span, name="get_weather", call_id="call_1")
    span.end()

    assert _attributes(exporter)["gen_ai.conversation.id"] == "room-42"
