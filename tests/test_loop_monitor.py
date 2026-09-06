"""Event loop blocking detector (``telemetry.loop_monitor``).

The monitor runs a heartbeat on the loop under test and a watchdog thread that samples the loop
thread's stack, so these tests use real wall-clock time with short thresholds. Durations are
asserted with generous tolerances to stay stable on loaded CI machines."""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Iterator
from types import SimpleNamespace

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


def _fake_job_context(session: object | None = None) -> SimpleNamespace:
    """The JobContext surface the monitor and the session-span queue touch."""
    from livekit.agents.job import JobContext

    ctx = SimpleNamespace(
        _primary_agent_session=session,
        _pending_session_spans=[],
        # read by the span processors main installs (PII stripping, job attribution)
        _redaction_enabled=False,
        _telemetry_state=None,
        job=SimpleNamespace(id="AJ_test", room=SimpleNamespace(sid="RM_test")),
    )
    ctx._defer_session_span = lambda rec: JobContext._defer_session_span(ctx, rec)  # type: ignore[arg-type]
    ctx._flush_pending_session_spans = lambda parent: JobContext._flush_pending_session_spans(
        ctx,  # type: ignore[arg-type]
        parent,
    )
    return ctx


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


async def _settle() -> None:
    # give the late heartbeat a chance to run and report
    await asyncio.sleep(TICK * 4)


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
    assert "_block_loop_synchronously" in stack
    assert "time.sleep" in stack
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


async def test_stall_before_the_session_is_held_then_emitted_under_it(
    span_exporter: InMemorySpanExporter, monitor: EventLoopMonitor
) -> None:
    """The trace view is organised around agent_session; a stall in the entrypoint before
    session.start() must still end up under it, at its original time."""
    job_ctx = _fake_job_context(session=None)
    monitor.set_report_context(_job_report_context(job_ctx))

    _block_loop_synchronously(0.07)
    await _settle()

    # recorded and held: no span yet, no root to attach it to
    assert _blocked_spans(span_exporter) == []
    assert len(job_ctx._pending_session_spans) == 1
    held = job_ctx._pending_session_spans[0]
    assert held.name == SPAN_NAME and held.end_ns is not None

    # the session starts later and adopts what happened before it
    with tracer.start_as_current_span("agent_session") as root:
        flushed = job_ctx._flush_pending_session_spans(trace.set_span_in_context(root))
    assert flushed == [held]
    [span] = _blocked_spans(span_exporter)
    assert span.parent is not None and span.parent.span_id == root.get_span_context().span_id
    # back-dated to when it happened, which is before the session existed
    assert span.start_time is not None and span.end_time is not None
    assert span.end_time <= root.start_time  # type: ignore[operator]
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_BLOCKING_SEVERITY] == "warning"
    assert "time.sleep" in attrs[trace_types.ATTR_BLOCKING_STACK]


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
    from livekit.agents import Agent

    from .fake_session import FakeActions, create_session, run_session

    actions = FakeActions()
    actions.add_user_speech(0.5, 1.0, "Hi", stt_delay=0.1)
    actions.add_llm("Hello", ttft=0.05, duration=0.1)
    actions.add_tts(0.2, ttfb=0.05, duration=0.1)
    session = create_session(actions, speed_factor=4.0)

    job_ctx = _fake_job_context(session=session)
    monitor.set_report_context(_job_report_context(job_ctx))

    blocked = False

    def _block_once(ev: object) -> None:
        nonlocal blocked
        if not blocked:
            blocked = True
            _block_loop_synchronously(0.07)  # synchronous work inside a session callback

    session.on("agent_state_changed", _block_once)
    await run_session(session, Agent(instructions="test"), drain_delay=0.5)

    [root] = [s for s in span_exporter.get_finished_spans() if s.name == "agent_session"]
    stalls = _blocked_spans(span_exporter)
    assert stalls, "the stall during the session was not emitted"
    for stall in stalls:
        assert stall.parent is not None and stall.parent.span_id == root.context.span_id

    # the session span carries the summary so the session list can flag it
    attrs = root.attributes or {}
    assert attrs[trace_types.ATTR_BLOCKING_COUNT] == len(stalls)
    assert attrs[trace_types.ATTR_BLOCKING_MAX_DURATION] >= 0.05
    assert (
        attrs[trace_types.ATTR_BLOCKING_TOTAL_DURATION]
        >= attrs[trace_types.ATTR_BLOCKING_MAX_DURATION]
    )
    events = [e for e in root.events if e.name == SPAN_NAME]
    assert len(events) == len(stalls)


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
    """Heavy but cooperative work: many callbacks each well under the threshold, timers,
    executor round trips, and tasks yielding to each other, for ~0.6 s of wall time."""
    loop = asyncio.get_running_loop()
    fired: list[int] = []

    async def worker(n: int) -> None:
        for _ in range(60):
            time.sleep(0.002)  # a small synchronous slice, well under WARN
            await asyncio.sleep(0)
        fired.append(n)

    handles = [loop.call_later(i * 0.01, fired.append, 1000 + i) for i in range(40)]
    started = time.monotonic()
    await asyncio.gather(*(worker(n) for n in range(4)))
    workers_took = time.monotonic() - started
    await loop.run_in_executor(None, time.sleep, 0.05)
    await asyncio.sleep(0.45)
    for h in handles:
        h.cancel()
    await _settle()

    assert len([f for f in fired if f < 1000]) == 4
    # the workers hold the loop for ~0.5 s of 2 ms slices; a host that starved this process
    # enough to stretch that past double is not a monitor false positive, and the
    # watchdog-gap tag only catches starvation concentrated in one stall
    if workers_took > 1.0:
        pytest.skip(f"host starved the test process: workers took {workers_took:.2f}s")
    assert _loop_blocks(monitor) == []


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
    assert isinstance(stack, str) and "_run" in stack and "self._callback" in stack


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

    reports = monitor.reports[before:]  # type: ignore[attr-defined]
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


def test_host_descheduling_is_not_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """When the watchdog stalled along with the loop, the host did not run the process; that
    is not a programming issue and must not show up as a warning or error."""
    loop = asyncio.new_event_loop()
    try:
        m = EventLoopMonitor(loop, warn_threshold=WARN, error_threshold=ERROR, tick_interval=TICK)
        report = BlockedReport(
            duration=0.8,
            started_at=time.time() - 0.8,
            warn_threshold=WARN,
            severity="error",
            gc_time=0.0,
            cpu_time=0.001,
            watchdog_gap=0.75,
            process_descheduled=True,
            task_name=None,
            stacks=[],
        )
        with caplog.at_level("DEBUG", logger="livekit.agents"):
            m._emit_log(report)
    finally:
        loop.close()

    records = [r for r in caplog.records if "event loop" in r.getMessage()]
    assert records, "the stall still leaves a debug trail"
    assert all(r.levelname == "DEBUG" for r in records)
    assert all("not scheduled" in r.getMessage() for r in records)


def test_host_descheduling_span_is_not_an_error() -> None:
    """A stall the host caused is reported at warning severity whatever its length, so the
    span keeps UNSET status; the ERROR status is reserved for code that blocked the loop."""
    loop = asyncio.new_event_loop()
    try:
        m = EventLoopMonitor(loop, warn_threshold=WARN, error_threshold=ERROR, tick_interval=TICK)
        # the loop thread and the watchdog both woke ~1s late: nothing ran in between
        descheduled = m._build_report(1.0, gc_time=0.0, cpu_time=0.0, watchdog_gap=0.9, samples=[])
        blocked = m._build_report(1.0, gc_time=0.0, cpu_time=0.9, watchdog_gap=0.0, samples=[])
    finally:
        loop.close()

    assert descheduled.process_descheduled and descheduled.severity == "warning"
    assert not blocked.process_descheduled and blocked.severity == "error"


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
