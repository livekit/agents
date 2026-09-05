"""Event loop blocking detector (``telemetry.loop_monitor``).

The monitor runs a heartbeat on the loop under test and a watchdog thread that samples the loop
thread's stack, so these tests use real wall-clock time with short thresholds. Durations are
asserted with generous tolerances to stay stable on loaded CI machines."""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.telemetry import loop_monitor, set_tracer_provider, trace_types, tracer
from livekit.agents.telemetry.loop_monitor import (
    ENV_ERROR_THRESHOLD_MS,
    ENV_WARN_THRESHOLD_MS,
    SPAN_NAME,
    BlockedReport,
    EventLoopMonitor,
    LoopMonitorThresholds,
    _RateLimiter,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

WARN = 0.03
ERROR = 0.15
TICK = 0.005


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
async def monitor() -> Iterator[EventLoopMonitor]:  # type: ignore[misc]
    m = EventLoopMonitor(
        asyncio.get_running_loop(), warn_threshold=WARN, error_threshold=ERROR, tick_interval=TICK
    )
    reports: list[BlockedReport] = []
    m._on_report = reports.append
    m.reports = reports  # type: ignore[attr-defined]
    m.start()
    # let the heartbeat arm and settle before the test blocks the loop
    await asyncio.sleep(WARN)
    try:
        yield m
    finally:
        m.stop()


def _blocked_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == SPAN_NAME]


def _block_loop_synchronously(duration: float) -> None:
    # a deliberately blocking call on the loop thread, the thing the monitor exists to catch
    time.sleep(duration)


async def _settle() -> None:
    # give the late heartbeat a chance to run and report
    await asyncio.sleep(TICK * 4)


async def test_blocking_call_is_reported_as_span(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    _block_loop_synchronously(0.2)
    await _settle()

    spans = _blocked_spans(span_exporter)
    assert len(spans) == 1
    span = spans[0]
    attrs = span.attributes or {}

    duration = attrs[trace_types.ATTR_BLOCKING_DURATION]
    assert isinstance(duration, float)
    # resolution is one tick; allow scheduler slop on top
    assert 0.2 - TICK - 0.01 <= duration <= 0.2 + 0.1
    assert attrs[trace_types.ATTR_BLOCKING_THRESHOLD] == WARN
    assert attrs[trace_types.ATTR_BLOCKING_SEVERITY] == "error"
    assert span.status.status_code == trace.StatusCode.ERROR

    # the span is back-dated so it covers the block on the timeline
    assert span.end_time is not None and span.start_time is not None
    assert abs((span.end_time - span.start_time) / 1e9 - duration) < 1e-6

    # the watchdog sampled the loop thread while it was stuck in time.sleep
    stack = attrs[trace_types.ATTR_BLOCKING_STACK]
    assert isinstance(stack, str)
    assert "_block_loop_synchronously" in stack
    assert "time.sleep" in stack
    # the sample names the task that was running
    assert attrs.get(trace_types.ATTR_BLOCKING_TASK)
    # time.sleep is a wait, not compute: the loop thread burned almost no CPU
    cpu = attrs[trace_types.ATTR_BLOCKING_CPU_TIME]
    assert isinstance(cpu, float) and cpu < duration / 2


async def test_block_between_thresholds_is_a_warning(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    _block_loop_synchronously(0.07)
    await _settle()

    spans = _blocked_spans(span_exporter)
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs[trace_types.ATTR_BLOCKING_SEVERITY] == "warning"
    assert spans[0].status.status_code == trace.StatusCode.UNSET


async def test_cooperative_work_is_not_reported(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    # plenty of short callbacks, none long enough to matter
    for _ in range(40):
        time.sleep(0.002)
        await asyncio.sleep(0)
    await asyncio.sleep(WARN * 2)

    assert _blocked_spans(span_exporter) == []
    assert monitor.reports == []  # type: ignore[attr-defined]


async def test_report_context_parents_span_and_carries_job_context(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    # emulate job_proc_lazy_main handing over the job_entrypoint span's context
    with tracer.start_as_current_span("job_entrypoint") as parent:
        monitor.set_report_context(contextvars.copy_context())

    _block_loop_synchronously(0.07)
    await _settle()

    spans = _blocked_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == parent.get_span_context().span_id


async def test_stop_is_idempotent_and_quiets_the_monitor(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    monitor.stop()
    monitor.stop()
    _block_loop_synchronously(0.07)
    await _settle()
    assert _blocked_spans(span_exporter) == []


def test_rate_limiter_counts_suppressed() -> None:
    limiter = _RateLimiter(2)
    assert limiter.allow(100.0)
    assert limiter.allow(100.1)
    assert not limiter.allow(100.2)
    assert not limiter.allow(100.3)
    assert limiter.take_suppressed() == 2
    assert limiter.take_suppressed() == 0
    # the window slides
    assert limiter.allow(161.0)


def test_thresholds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_WARN_THRESHOLD_MS, raising=False)
    monkeypatch.delenv(ENV_ERROR_THRESHOLD_MS, raising=False)
    t = LoopMonitorThresholds.from_env()
    assert t == LoopMonitorThresholds(warn=0.05, error=0.5)

    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "100")
    monkeypatch.setenv(ENV_ERROR_THRESHOLD_MS, "1000")
    assert LoopMonitorThresholds.from_env() == LoopMonitorThresholds(warn=0.1, error=1.0)

    # zero disables
    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "0")
    assert LoopMonitorThresholds.from_env() is None

    # garbage falls back to the default rather than disabling
    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "fast")
    assert LoopMonitorThresholds.from_env() == LoopMonitorThresholds(warn=0.05, error=1.0)

    # error below warn is clamped up to warn
    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "200")
    monkeypatch.setenv(ENV_ERROR_THRESHOLD_MS, "20")
    assert LoopMonitorThresholds.from_env() == LoopMonitorThresholds(warn=0.2, error=0.2)


async def test_registry_starts_once_per_loop_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.get_running_loop()
    thresholds = LoopMonitorThresholds(warn=WARN, error=ERROR)
    m = loop_monitor.start_monitoring(loop, thresholds=thresholds)
    try:
        assert m is not None
        assert loop_monitor.get_monitor(loop) is m
        assert loop_monitor.start_monitoring(loop, thresholds=thresholds) is None
    finally:
        loop_monitor.stop_monitoring(loop)
    assert loop_monitor.get_monitor(loop) is None

    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "0")
    assert loop_monitor.start_monitoring(loop) is None


def test_constructor_validates_thresholds() -> None:
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError):
            EventLoopMonitor(loop, warn_threshold=0)
        with pytest.raises(ValueError):
            EventLoopMonitor(loop, warn_threshold=0.1, error_threshold=0.05)
        with pytest.raises(ValueError):
            EventLoopMonitor(loop, warn_threshold=0.01, tick_interval=0.05)
    finally:
        loop.close()
