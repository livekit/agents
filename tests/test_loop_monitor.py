"""Event loop blocking detector (``telemetry.loop_monitor``).

The monitor runs a heartbeat on the loop under test and a watchdog thread that samples the loop
thread's stack, so these tests use real wall-clock time with short thresholds. Durations are
asserted with generous tolerances to stay stable on loaded CI machines."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import inspect
import linecache
import math
import os
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents._generated.telemetry import LoopMonitorEmission
from livekit.agents.telemetry import loop_monitor, set_tracer_provider, trace_types, tracer
from livekit.agents.telemetry.loop_monitor import (
    SPAN_NAME,
    BlockedReport,
    EventLoopMonitor,
    LoopMonitorThresholds,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

ENV_WARN_THRESHOLD_MS = "LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS"
ENV_ERROR_THRESHOLD_MS = "LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS"

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


def _fake_job_context(session: object | None = None) -> SimpleNamespace:
    """The JobContext surface the monitor touches."""
    return SimpleNamespace(
        _primary_agent_session=session,
        # read by the span processors main installs (PII stripping, job attribution)
        _redaction_enabled=False,
        _telemetry_state=None,
        job=SimpleNamespace(id="AJ_test", room=SimpleNamespace(sid="RM_test")),
    )


def _job_report_context(job_ctx: object) -> contextvars.Context:
    from livekit.agents.job import _JobContextVar

    token = _JobContextVar.set(job_ctx)  # type: ignore[arg-type]
    try:
        return contextvars.copy_context()
    finally:
        _JobContextVar.reset(token)


@pytest.fixture
async def monitor(span_exporter: InMemorySpanExporter) -> Iterator[EventLoopMonitor]:  # type: ignore[misc]
    """A monitor reporting into a fake job whose primary session is already running, so spans
    are emitted immediately under a live ``agent_session`` root (the common case). Tests for
    the pre-session and no-job paths replace the report context themselves."""
    m = EventLoopMonitor(
        asyncio.get_running_loop(), warn_threshold=WARN, error_threshold=ERROR, tick_interval=TICK
    )
    reports: list[BlockedReport] = []
    m._on_report = reports.append
    m.reports = reports  # type: ignore[attr-defined]

    root = tracer.start_span("agent_session")
    session = SimpleNamespace(
        _root_span_context=trace.set_span_in_context(root),
        _record_loop_stall=lambda duration, *, timestamp_ns: None,
    )
    m.set_report_context(_job_report_context(_fake_job_context(session=session)))
    m.session_root = root  # type: ignore[attr-defined]

    m.start()
    # let the heartbeat arm and settle before the test blocks the loop
    await asyncio.sleep(WARN)
    # Startup may itself be delayed by the host scheduler; each case observes its own workload.
    reports.clear()
    span_exporter.clear()
    try:
        yield m
    finally:
        m.stop()
        root.end()


def _blocked_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == SPAN_NAME]


def _block_loop_synchronously(duration: float) -> None:
    # a deliberately blocking call on the loop thread, the thing the monitor exists to catch
    time.sleep(duration)


_SLEEP_CALL = "line {}, in _block_loop_synchronously".format(
    _block_loop_synchronously.__code__.co_firstlineno
    + next(
        i
        for i, ln in enumerate(inspect.getsource(_block_loop_synchronously).splitlines())
        if "time.sleep" in ln
    )
)


async def _settle() -> None:
    # give the late heartbeat a chance to run and report
    await asyncio.sleep(TICK * 4)


def _describe(reports: list[BlockedReport]) -> str:
    """Everything a failure needs to be diagnosed from CI output alone."""
    return "\n".join(
        f"{r.duration * 1000:.0f}ms (gc {r.gc_time * 1000:.0f}ms, cpu {r.cpu_time * 1000:.0f}ms, "
        f"watchdog gap {r.watchdog_gap * 1000:.0f}ms, task {r.task_name}) "
        + " | ".join(s.replace(chr(10), " / ") for s in r.stacks)
        for r in reports
    )


def _loop_blocks(monitor: EventLoopMonitor) -> list[BlockedReport]:
    """Reports attributable to this loop. A loaded CI host can deschedule the whole process
    for tens of milliseconds; the monitor reports that too (the loop really did stall) but
    tags it, and it is not what the negative tests here are checking for."""
    return [r for r in monitor.reports if not r.process_descheduled]  # type: ignore[attr-defined]


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
    assert _SLEEP_CALL in stack
    # each sample says when in the stall it was taken: it is a sample, not a profile
    assert "loop thread sampled" in stack and "ms into the stall" in stack
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

    assert _loop_blocks(monitor) == []


async def test_stall_before_the_session_lands_in_the_job_trace(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    """Before session.start() there is no session root to resolve; the stall is a child of
    whatever the job's reporting context holds, job_entrypoint in a job process, at its
    original time."""
    job_ctx = _fake_job_context(session=None)
    entrypoint = tracer.start_span("job_entrypoint")
    with tracer.use_span(entrypoint, end_on_exit=False):
        monitor.set_report_context(_job_report_context(job_ctx))

    _block_loop_synchronously(0.07)
    await _settle()
    entrypoint.end()

    [span] = _blocked_spans(span_exporter)
    assert span.parent is not None
    assert span.parent.span_id == entrypoint.get_span_context().span_id
    assert span.start_time is not None and span.end_time is not None
    assert span.end_time <= entrypoint.end_time  # type: ignore[operator]
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_BLOCKING_SEVERITY] == "warning"
    assert _SLEEP_CALL in attrs[trace_types.ATTR_BLOCKING_STACK]


async def test_stall_without_a_job_is_log_only(
    span_exporter: InMemorySpanExporter,
    monitor: EventLoopMonitor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # no job context in the report context: the worker process, or a bare loop. There will
    # never be a session to attach a span to, so the log carries it.
    monitor.set_report_context(None)
    with caplog.at_level("WARNING", logger="livekit.agents"):
        _block_loop_synchronously(0.07)
        await _settle()
    assert _blocked_spans(span_exporter) == []
    assert len(monitor.reports) == 1  # type: ignore[attr-defined]
    assert any("event loop blocked for" in r.getMessage() for r in caplog.records)


async def test_stall_during_a_session_lands_under_it_and_is_summarised(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    from livekit.agents import Agent, VoiceSession

    from .fake_llm import FakeLLM

    session = VoiceSession(llm=FakeLLM(), vad=None, turn_handling={"turn_detection": None})

    job_ctx = _fake_job_context(session=session)
    monitor.set_report_context(_job_report_context(job_ctx))

    blocked = False

    def _block_once(ev: object) -> None:
        nonlocal blocked
        if not blocked:
            blocked = True
            _block_loop_synchronously(0.07)  # synchronous work inside a session callback

    session.on("agent_state_changed", _block_once)
    try:
        await session.start(Agent(instructions="test"))
        await _settle()
        assert blocked, "session startup must exercise its state callback"
    finally:
        await session.aclose()

    [root] = [s for s in span_exporter.get_finished_spans() if s.name == "agent_session"]
    stalls = [
        span
        for span in _blocked_spans(span_exporter)
        if _SLEEP_CALL in (span.attributes or {}).get(trace_types.ATTR_BLOCKING_STACK, "")
    ]
    assert stalls, "the stall during the session was not emitted"
    for stall in stalls:
        assert stall.parent is not None and stall.parent.span_id == root.context.span_id

    # the session span carries the summary so the session list can flag it
    session_stalls = [
        span
        for span in _blocked_spans(span_exporter)
        if span.parent is not None and span.parent.span_id == root.context.span_id
    ]
    attrs = root.attributes or {}
    assert attrs[trace_types.ATTR_BLOCKING_COUNT] == len(session_stalls)
    assert attrs[trace_types.ATTR_BLOCKING_MAX_DURATION] >= 0.05
    assert (
        attrs[trace_types.ATTR_BLOCKING_TOTAL_DURATION]
        >= attrs[trace_types.ATTR_BLOCKING_MAX_DURATION]
    )
    events = [e for e in root.events if e.name == SPAN_NAME]
    assert len(events) == len(session_stalls)


async def test_stop_is_idempotent_and_quiets_the_monitor(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    monitor.stop()
    monitor.stop()
    _block_loop_synchronously(0.07)
    await _settle()
    assert _blocked_spans(span_exporter) == []


async def test_idle_loop_is_not_reported(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    # an idle loop sits in select() until the heartbeat is due; the timer fires on time
    await asyncio.sleep(WARN * 12)
    assert _loop_blocks(monitor) == []


async def test_blocking_work_in_an_executor_is_not_reported(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    # the whole point of run_in_executor / to_thread: the loop keeps ticking
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, time.sleep, WARN * 6)
    await asyncio.to_thread(time.sleep, WARN * 6)
    await _settle()
    assert _loop_blocks(monitor) == []


async def test_sustained_cooperative_load_is_not_reported(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    """Cooperative CPU slices keep timers progressing and do not report blocked worker tasks.

    Host scheduler stalls elsewhere on the loop remain legitimate observations; they are
    independent of whether these workers yield between their small CPU slices.
    """
    loop = asyncio.get_running_loop()
    fired: list[int] = []

    async def worker(n: int) -> None:
        for _ in range(120):
            sum(range(1000))
            await asyncio.sleep(0)
        fired.append(n)

    handles = [loop.call_later(i * 0.01, fired.append, 1000 + i) for i in range(40)]
    names = {f"cooperative-{n}" for n in range(2)}
    tasks = [loop.create_task(worker(n), name=f"cooperative-{n}") for n in range(2)]
    await asyncio.gather(*tasks)
    await loop.run_in_executor(None, time.sleep, 0.05)
    await asyncio.sleep(0.45)
    for handle in handles:
        handle.cancel()
    await _settle()

    assert len([value for value in fired if value < 1000]) == 2
    assert any(value >= 1000 for value in fired), "timers must progress while workers cooperate"
    blocked_workers = [report for report in _loop_blocks(monitor) if report.task_name in names]
    assert blocked_workers == [], _describe(blocked_workers)


async def test_one_iteration_of_many_ready_callbacks_is_one_stall(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    """A burst of ready callbacks runs within one loop iteration, so the heartbeat cannot fire
    in between: that is reported as a single stall of the burst's length, and the stack sample
    lands in one of the callbacks."""
    loop = asyncio.get_running_loop()
    for _ in range(60):
        loop.call_soon(time.sleep, 0.002)  # ~120 ms of back-to-back callbacks
    await asyncio.sleep(0)
    await _settle()

    spans = _blocked_spans(span_exporter)
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    duration = attrs[trace_types.ATTR_BLOCKING_DURATION]
    assert isinstance(duration, float) and 0.09 <= duration <= 0.3
    # time.sleep is a C function scheduled directly, so it has no frame of its own: the
    # innermost frame is asyncio's dispatch of the callback, which must be kept
    stack = attrs[trace_types.ATTR_BLOCKING_STACK]
    assert isinstance(stack, str)
    assert os.path.join("asyncio", "events.py") in stack and stack.rstrip().endswith("in _run")


async def test_gc_pause_is_attributed(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    import gc

    # a large heap of container objects makes a full collection measurably slow
    heap = [{"i": i, "l": [i]} for i in range(400_000)]
    gc.collect()  # baseline, outside the measurement window
    await _settle()
    before = len(monitor.reports)  # type: ignore[attr-defined]

    start = time.perf_counter()
    gc.collect()
    elapsed = time.perf_counter() - start
    await _settle()
    del heap

    reports = [report for report in monitor.reports[before:] if report.gc_time > 0]  # type: ignore[attr-defined]
    if elapsed < WARN:
        pytest.skip(f"gc.collect took only {elapsed * 1000:.0f}ms on this machine")
    assert len(reports) == 1
    report = reports[0]
    # the pause is a GC pause, and the span says so instead of blaming the coroutine
    assert report.gc_time > 0
    assert report.gc_time <= report.duration + 1e-3
    assert report.gc_time >= report.duration * 0.5


async def test_worker_mode_logs_and_records_the_metric_without_spans(
    span_exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    m = EventLoopMonitor(
        asyncio.get_running_loop(),
        warn_threshold=WARN,
        error_threshold=ERROR,
        tick_interval=TICK,
        emit_spans=False,
    )
    reports: list[BlockedReport] = []
    m._on_report = reports.append
    m.start()
    await asyncio.sleep(WARN)
    try:
        with caplog.at_level("WARNING", logger="livekit.agents"):
            _block_loop_synchronously(0.08)
            await _settle()
    finally:
        m.stop()

    assert _blocked_spans(span_exporter) == []
    assert len(reports) == 1
    assert any("event loop blocked for" in r.getMessage() for r in caplog.records)


async def test_gil_holding_native_call_is_reported_end_to_end(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    # one C call that holds the GIL throughout; sized well above WARN on any machine
    cpu_before = time.thread_time()
    sum_line = sys._getframe().f_lineno + 1
    sum(range(20_000_000))
    work_cpu = time.thread_time() - cpu_before
    await _settle()

    reports = [report for report in _loop_blocks(monitor) if report.cpu_time >= work_cpu]
    assert len(reports) == 1, _describe(monitor.reports)  # type: ignore[attr-defined]
    [report] = reports
    # OS preemption can increase wall duration without consuming this thread's CPU.
    assert report.cpu_time >= work_cpu
    [span] = [
        span
        for span in _blocked_spans(span_exporter)
        if isinstance((span.attributes or {}).get(trace_types.ATTR_BLOCKING_CPU_TIME), int | float)
        and (span.attributes or {})[trace_types.ATTR_BLOCKING_CPU_TIME] >= work_cpu
    ]
    # the watchdog usually cannot run at all and the report says so; it can also get the GIL
    # in the instant the call returns and sample the frame still on that line. Either names
    # the cause; what must not happen is a silent or host-attributed report.
    stack = str((span.attributes or {})[trace_types.ATTR_BLOCKING_STACK])
    sum_call = f"line {sum_line}, in test_gil_holding_native_call_is_reported_end_to_end"
    assert "held the GIL" in stack or sum_call in stack, stack


def test_thresholds_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_WARN_THRESHOLD_MS, raising=False)
    monkeypatch.delenv(ENV_ERROR_THRESHOLD_MS, raising=False)
    t = LoopMonitorThresholds.from_env()
    assert t == LoopMonitorThresholds(warn=0.1, error=0.5)

    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "100")
    monkeypatch.setenv(ENV_ERROR_THRESHOLD_MS, "1000")
    assert LoopMonitorThresholds.from_env() == LoopMonitorThresholds(warn=0.1, error=1.0)

    # zero disables
    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "0")
    assert LoopMonitorThresholds.from_env() is None

    # garbage falls back to the default rather than disabling
    monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, "fast")
    assert LoopMonitorThresholds.from_env() == LoopMonitorThresholds(warn=0.1, error=1.0)

    # so do NaN and infinity: NaN compares false against every lag and would report each tick
    for bad in ("NaN", "inf", "-inf"):
        monkeypatch.setenv(ENV_WARN_THRESHOLD_MS, bad)
        assert LoopMonitorThresholds.from_env() == LoopMonitorThresholds(warn=0.1, error=1.0)

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


@pytest.mark.skipif(not hasattr(asyncio.Task, "get_context"), reason="Task.get_context is 3.12+")
async def test_stall_nests_under_the_blocked_tasks_span(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    """A stall inside an rpc handler (or a tool, or the user hook) is that span's problem: the
    watchdog sample carries the blocked task's current span and the stall nests under it."""

    async def handler() -> None:
        with tracer.start_as_current_span("rpc_handler"):
            _block_loop_synchronously(0.08)

    await asyncio.create_task(handler(), name="rpc_handler_task")
    await _settle()

    [handler_span] = [s for s in span_exporter.get_finished_spans() if s.name == "rpc_handler"]
    [stall] = [
        span
        for span in _blocked_spans(span_exporter)
        if (span.attributes or {}).get(trace_types.ATTR_BLOCKING_TASK) == "rpc_handler_task"
    ]
    assert stall.parent is not None
    assert stall.parent.span_id == handler_span.context.span_id
    assert (stall.attributes or {})[trace_types.ATTR_BLOCKING_TASK] == "rpc_handler_task"


def test_a_warn_threshold_below_the_tick_floor_is_raised_to_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The monitor never ticks faster than 20 ms, so it cannot measure a smaller threshold;
    rather than fail to start (the tick would exceed the threshold), the threshold is raised
    and said so."""
    loop = asyncio.new_event_loop()
    try:
        with caplog.at_level("WARNING", logger="livekit.agents"):
            m = loop_monitor.start_monitoring(
                loop, thresholds=LoopMonitorThresholds(warn=0.005, error=0.01), emit_spans=False
            )
        assert m is not None
        assert m.warn_threshold == pytest.approx(0.02) and m.error_threshold == pytest.approx(0.02)
        assert any("20ms sampling floor" in r.getMessage() for r in caplog.records)
    finally:
        loop_monitor.stop_monitoring(loop)
        loop.close()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["warn_threshold", "error_threshold", "tick_interval"])
async def test_nonfinite_monitor_timing_is_rejected_before_start(field: str, value: float) -> None:
    options = {"warn_threshold": 0.1, "error_threshold": 0.5, "tick_interval": 0.02}
    options[field] = value
    with pytest.raises(ValueError, match="finite"):
        EventLoopMonitor(asyncio.get_running_loop(), **options)


def test_stopped_monitor_releases_its_host_callback_owner() -> None:
    import gc
    import weakref

    loop = asyncio.new_event_loop()
    try:
        monitor = EventLoopMonitor(loop)
        observed = weakref.ref(monitor)
        monitor.stop()
        del monitor
        gc.collect()
        assert observed() is None
    finally:
        loop.close()


def test_start_failure_releases_registered_gc_callback() -> None:
    import gc
    from collections.abc import Callable
    from typing import cast

    class RefusingLoop:
        def call_soon_threadsafe(self, callback: Callable[[], None]) -> None:
            callback()

        def call_later(self, delay: float, callback: object) -> None:
            raise RuntimeError("timer unavailable")

    before = tuple(gc.callbacks)
    monitor = EventLoopMonitor(cast(asyncio.AbstractEventLoop, RefusingLoop()))
    try:
        with pytest.raises(RuntimeError, match="timer unavailable"):
            monitor.start()
        assert tuple(gc.callbacks) == before
    finally:
        monitor.stop()
        for callback in tuple(gc.callbacks):
            if callback not in before:
                gc.callbacks.remove(callback)


def test_suppressed_reports_keep_metrics_and_summary_in_the_saved_job_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.job import get_job_context

    recorded: list[tuple[float, object]] = []
    summary: list[float] = []
    observed: list[BlockedReport] = []
    monkeypatch.setattr(
        loop_monitor,
        "record_event_loop_blocked",
        lambda duration, *, severity: recorded.append((duration, get_job_context(required=False))),
    )
    session = SimpleNamespace(
        _record_loop_stall=lambda duration, *, timestamp_ns: summary.append(duration),
    )
    job = _fake_job_context(session)
    report = BlockedReport(
        duration=0.1,
        started_at=100.0,
        warn_threshold=WARN,
        severity="warning",
        gc_time=0.0,
        cpu_time=0.1,
        watchdog_gap=0.0,
        process_descheduled=False,
        task_name=None,
        stacks=[],
    )
    loop = asyncio.new_event_loop()
    monitor = EventLoopMonitor(loop)
    monitor._on_report = observed.append
    monitor.set_report_context(_job_report_context(job))
    try:
        for _ in range(40):
            monitor._report(LoopMonitorEmission(report=report, span=None, log=None, reported=False))
        assert recorded == [(0.1, job)] * 40
        assert summary == [0.1] * 40
        assert observed == []
    finally:
        monitor.stop()
        loop.close()


async def test_sampling_and_report_formatting_read_no_source(
    tmp_path: Path,
    monitor: EventLoopMonitor,
) -> None:
    readers: list[int] = []
    source = "def hold():\n    import time\n    time.sleep(0.08)\n"

    class RecordingLoader:
        def get_source(self, name: str) -> str:
            readers.append(threading.get_ident())
            return source

    module = ModuleType("loader_source")
    module.__loader__ = RecordingLoader()  # type: ignore[assignment]
    path = tmp_path / "missing" / "loader_source.py"
    exec(compile(source, str(path), "exec"), module.__dict__)
    linecache.clearcache()
    module.hold()
    await _settle()
    assert readers == []
    assert str(path) not in linecache.cache
    reports = _loop_blocks(monitor)
    assert reports
    assert any(
        f'File "{path}", line 3, in hold' in stack for report in reports for stack in report.stacks
    )


async def test_sampling_releases_completed_task_frames_without_cyclic_collection(
    monitor: EventLoopMonitor,
) -> None:
    class Payload:
        pass

    async def blocked_task() -> weakref.ReferenceType[Payload]:
        payload = Payload()
        observed = weakref.ref(payload)
        _block_loop_synchronously(0.08)
        return observed

    enabled = gc.isenabled()
    gc.disable()
    try:
        observed = await asyncio.create_task(blocked_task())
        await _settle()
        assert _loop_blocks(monitor)
        assert observed() is None, "the sampler retained the completed task's frame"
    finally:
        if enabled:
            gc.enable()


async def test_import_stall_names_the_module_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    monitor: EventLoopMonitor,
) -> None:
    import importlib

    path = tmp_path / "slow_monitor_import.py"
    path.write_text("import time\ntime.sleep(0.08)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        importlib.import_module("slow_monitor_import")
        await _settle()
        reports = _loop_blocks(monitor)
        assert any(report.importing == "slow_monitor_import" for report in reports)
        assert any(str(path) in stack for report in reports for stack in report.stacks)
    finally:
        sys.modules.pop("slow_monitor_import", None)
