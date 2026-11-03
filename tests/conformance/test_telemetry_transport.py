"""Trace identity crosses the real job IPC boundary independently for each invocation: the
native `job_entrypoint` continues the span current at `launch_job`, and the entrypoint runs
under it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import JobContext
from livekit.agents.ipc import JobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("active", [True, False], ids=["active", "missing"])
async def test_job_ipc_preserves_each_parent_and_restores_the_callers_context(
    own_executor: Callable[[JobExecutor], JobExecutor], active: bool
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sdk = trace.get_tracer("customer-ipc", tracer_provider=provider)

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
            parent = (
                trace.NonRecordingSpan(
                    trace.SpanContext(trace_id, span_id, False, trace.TraceFlags(1))
                )
                if active
                else trace.INVALID_SPAN
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
        roots = set()
        for job_id, (trace_id, span_id) in [("one", (101, 11)), ("two", (202, 22))]:
            child = next(s for s in spans if s.name == "customer" and s.attributes["job"] == job_id)
            # the entrypoint runs under its job's native root, never the dispatcher's span itself
            assert child.parent is not None and child.parent.span_id != span_id
            roots.add(child.parent.span_id)
            assert (child.context.trace_id == trace_id) == active
        assert len(roots) == 2, "two jobs shared one root"
        assert trace.get_current_span() is previous
    finally:
        provider.shutdown()
