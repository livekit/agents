"""Standard SDK spans inside application callbacks through the generated AgentHost adapter."""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent, ModelRetry, RunContext
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def _response(input: str, content: str = "", tool: str | None = None) -> FakeLLMResponse:
    return FakeLLMResponse(
        input=input,
        content=content,
        ttft=0,
        duration=0,
        tool_calls=[FunctionToolCall(name=tool, arguments="{}", call_id="call")] if tool else [],
    )


def _parent(trace_id: int, span_id: int) -> trace.NonRecordingSpan:
    return trace.NonRecordingSpan(trace.SpanContext(trace_id, span_id, False, trace.TraceFlags(1)))


async def test_standard_sdk_callback_parentage_is_captured_before_export_for_concurrent_runs() -> (
    None
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sdk = trace.get_tracer("customer", tracer_provider=provider)
    agent = Agent[str, str](
        instructions="desk",
        name="desk",
        llm=FakeLLM(fake_responses=[_response("begin", tool="lookup"), _response("found", "Done")]),
    )

    @agent.instructions
    async def instructions(ctx: RunContext[str]) -> str:
        with sdk.start_as_current_span("instructions") as span:
            span.set_attribute("job", ctx.context)
            await asyncio.sleep(0)
        return "Use lookup."

    @agent.tool
    async def lookup(ctx: RunContext[str]) -> str:
        """Look up the customer's record."""
        with sdk.start_as_current_span("tool") as span:
            span.set_attribute("job", ctx.context)
            await asyncio.sleep(0)
        return "found"

    async def run(job: str, trace_id: int, span_id: int) -> None:
        # No span-start record was exported for this logical parent.
        parent = _parent(trace_id, span_id)
        with trace.use_span(parent):
            result = await agent.run("begin", context=job)
            assert result.status == "completed" and result.output == "Done"
            assert trace.get_current_span() is parent

    previous = trace.get_current_span()
    try:
        await asyncio.gather(run("one", 101, 11), run("two", 202, 22))
        spans = exporter.get_finished_spans()
        assert {(s.name, s.attributes["job"]) for s in spans} == {
            ("instructions", "one"),
            ("instructions", "two"),
            ("tool", "one"),
            ("tool", "two"),
        }
        for span in spans:
            expected = {"one": (101, 11), "two": (202, 22)}[span.attributes["job"]]
            assert (span.context.trace_id, span.parent.span_id) == expected
        assert trace.get_current_span() is previous
    finally:
        provider.shutdown()


async def test_generated_callback_context_restores_after_tool_failure_and_cancellation() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sdk = trace.get_tracer("customer", tracer_provider=provider)
    entered, settled = asyncio.Event(), asyncio.Event()
    agent = Agent[str, str](
        instructions="desk",
        name="desk",
        llm=FakeLLM(
            fake_responses=[_response("begin", tool="lookup"), _response("retry", "Recovered")]
        ),
    )

    @agent.tool
    async def lookup(ctx: RunContext[str]) -> str:
        """Look up the customer's record."""
        with sdk.start_as_current_span("tool"):
            if ctx.context == "failure":
                raise ModelRetry("retry")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                settled.set()
        return "found"

    parent = _parent(303, 33)
    previous = trace.get_current_span()
    try:
        with trace.use_span(parent):
            result = await agent.run("begin", context="failure")
            assert result.status == "completed" and result.output == "Recovered"
            assert trace.get_current_span() is parent
            async with agent.run_stream("begin", context="cancel") as stream:
                events = stream.stream_events()
                assert (await anext(events)).type == "tool_call"
                await entered.wait()
                stream.cancel()
                await events.aclose()
            with pytest.raises(asyncio.CancelledError):
                await stream.result()
            assert settled.is_set()
            assert trace.get_current_span() is parent
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert all((s.context.trace_id, s.parent.span_id) == (303, 33) for s in spans)
        assert trace.get_current_span() is previous
    finally:
        provider.shutdown()
