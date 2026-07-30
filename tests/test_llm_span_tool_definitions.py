"""Hermetic tests: the llm_node span records full tool definitions (#6620).

The span used to carry only tool *names* (``lk.function_tools``); descriptions
and parameter schemas never reached OTel backends, so traces could not show
what the LLM was actually offered on a turn.
"""

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.llm import ChatContext, ToolContext, function_tool
from livekit.agents.telemetry import set_tracer_provider, trace_types
from livekit.agents.voice.generation import _function_tool_definitions, perform_llm_inference
from livekit.agents.voice.io import ModelSettings

pytestmark = pytest.mark.unit


@function_tool
async def get_weather(location: str) -> str:
    """Look up the current weather for a location."""
    return "sunny"


@function_tool(
    raw_schema={
        "name": "transfer_call",
        "description": "Transfer the call to a human operator.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}},
    }
)
async def transfer_call(raw_arguments: dict) -> str:
    return "ok"


class TestFunctionToolDefinitions:
    def test_function_tool_definition_shape(self) -> None:
        defs = _function_tool_definitions(ToolContext([get_weather]))
        assert len(defs) == 1
        assert defs[0]["name"] == "get_weather"
        assert "weather" in defs[0]["description"].lower()
        assert defs[0]["parameters"]["properties"]["location"]["type"] == "string"

    def test_raw_tool_definition_shape(self) -> None:
        defs = _function_tool_definitions(ToolContext([transfer_call]))
        assert len(defs) == 1
        assert defs[0]["name"] == "transfer_call"
        assert defs[0]["description"] == "Transfer the call to a human operator."
        assert "reason" in defs[0]["parameters"]["properties"]

    def test_definitions_are_json_serializable(self) -> None:
        defs = _function_tool_definitions(ToolContext([get_weather, transfer_call]))
        parsed = json.loads(json.dumps(defs))
        assert {d["name"] for d in parsed} == {"get_weather", "transfer_call"}


class TestLLMSpanAttributes:
    async def test_llm_node_span_carries_tool_definitions(self) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        set_tracer_provider(provider)
        try:

            async def fake_node(chat_ctx, tools, model_settings):
                async def gen():
                    yield "hello"

                return gen()

            chat_ctx = ChatContext.empty()
            chat_ctx.add_message(role="user", content="hi")

            llm_task, data = perform_llm_inference(
                node=fake_node,
                chat_ctx=chat_ctx,
                tool_ctx=ToolContext([get_weather]),
                model_settings=ModelSettings(),
            )
            await llm_task

            spans = exporter.get_finished_spans()
            llm_span = next(s for s in spans if s.name == "llm_node")

            # existing name-only attribute is unchanged
            assert list(llm_span.attributes[trace_types.ATTR_FUNCTION_TOOLS]) == ["get_weather"]

            defs = json.loads(llm_span.attributes[trace_types.ATTR_FUNCTION_TOOL_DEFINITIONS])
            assert defs[0]["name"] == "get_weather"
            assert "weather" in defs[0]["description"].lower()
            assert defs[0]["parameters"]["properties"]["location"]["type"] == "string"
        finally:
            # restore an exporter-less provider so spans from other tests go nowhere
            set_tracer_provider(TracerProvider())
