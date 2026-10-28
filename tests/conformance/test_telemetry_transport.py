"""Trace identity crosses the real job IPC boundary independently for each invocation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import JobContext, ipc
from livekit.agents._generated.telemetry_transport import TraceCarrier
from livekit.agents.ipc import JobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.agents.telemetry._context import tracer
from livekit.protocol import agent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "wire_parent",
    ["active", None, "malformed", "x" * 4096],
    ids=["active", "missing", "malformed", "oversized"],
)
async def test_job_ipc_preserves_each_parent_and_restores_the_callers_context(
    monkeypatch: pytest.MonkeyPatch,
    own_executor: Callable[[JobExecutor], JobExecutor],
    wire_parent: str | None,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sdk = trace.get_tracer("customer-ipc", tracer_provider=provider)
    monkeypatch.setattr(tracer, "_tracer", sdk)
    if wire_parent != "active":
        monkeypatch.setattr(
            ipc, "capture_trace_carrier", lambda: TraceCarrier(traceparent=wire_parent)
        )

    async def entrypoint(ctx: JobContext) -> None:
        with sdk.start_as_current_span("customer") as span:
            span.set_attribute("job", ctx.job.id)
            await asyncio.sleep(0)
        ctx.shutdown("done")

    async def run(job_id: str, trace_id: int, span_id: int) -> None:
        executor = JobExecutor(
            thread=True,
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=None,
            simulation_end_fnc=None,
            initialize_timeout=10,
            close_timeout=10,
            session_end_timeout=1,
            ping_interval=0,
            high_ping_threshold=1,
            http_proxy=None,
            loop=asyncio.get_running_loop(),
        )
        own_executor(executor)
        try:
            await executor.start()
            await executor.initialize()
            parent = trace.NonRecordingSpan(
                trace.SpanContext(trace_id, span_id, False, trace.TraceFlags(1))
            )
            with trace.use_span(parent):
                await executor.launch_job(
                    RunningJobInfo(
                        JobAcceptArguments("", "", ""),
                        agent.Job(id=job_id),
                        "",
                        "",
                        "",
                        True,
                    )
                )
                assert trace.get_current_span() is parent
            await asyncio.wait_for(executor.join(), 10)
        finally:
            await executor.aclose()

    previous = trace.get_current_span()
    try:
        await asyncio.gather(run("one", 101, 11), run("two", 202, 22))
        spans = exporter.get_finished_spans()
        for job_id, expected in [("one", (101, 11)), ("two", (202, 22))]:
            root = next(
                s
                for s in spans
                if s.name == "job_entrypoint" and s.attributes["lk.job_id"] == job_id
            )
            child = next(s for s in spans if s.name == "customer" and s.attributes["job"] == job_id)
            if wire_parent == "active":
                assert root.parent is not None
                assert (root.context.trace_id, root.parent.span_id) == expected
            else:
                assert root.parent is None
                assert root.context.trace_id != expected[0]
            assert child.context.trace_id == root.context.trace_id
            assert child.parent.span_id == root.context.span_id
        assert trace.get_current_span() is previous
    finally:
        provider.shutdown()
