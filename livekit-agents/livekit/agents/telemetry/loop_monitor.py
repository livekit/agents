"""Detect synchronous code blocking an asyncio event loop and surface it as telemetry.

A voice pipeline runs VAD, playout, and every provider stream on one event loop, so any
synchronous work on it (a blocking HTTP client inside a tool, a large numpy operation, a
``time.sleep``) is audible as latency or jitter. asyncio only reports slow callbacks in debug
mode, which is far too expensive for production and only logs.

This module observes a single loop without patching asyncio:

* a **heartbeat** scheduled with ``loop.call_later`` every ``tick_interval`` records the time.
  When the loop is blocked, the tick fires late by the length of the block (to within one
  interval);
* a **watchdog thread** wakes on the same interval and compares now against the last tick. Once
  the gap crosses the warn threshold it samples the loop thread's stack through
  ``sys._current_frames()`` (the same call ``faulthandler`` relies on), so the report can say
  *where* the loop was stuck, not just for how long;
* when the late tick finally runs, the block has ended: the tick measures the lag and, if it
  crossed the threshold, emits a back-dated ``event_loop_blocked`` span carrying the sampled
  stack, the task that was running, GC time that fell inside the block, and the loop thread's
  CPU time. It also logs a warning and records a histogram measurement.

Spans are parented to the primary agent session's root span when a job is running so they land
on the session timeline next to the turn they delayed. Output is rate limited so a
pathologically blocked loop cannot flood the exporter.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import gc
import os
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context, trace

from ..log import logger
from . import otel_metrics, trace_types
from .traces import tracer

DEFAULT_WARN_THRESHOLD = 0.05
"""Blocks at or above this many seconds are reported as warnings."""
DEFAULT_ERROR_THRESHOLD = 0.5
"""Blocks at or above this many seconds are reported as errors (span status ERROR)."""
DEFAULT_TICK_INTERVAL = 0.01
"""Heartbeat period. Also the measurement resolution."""

ENV_WARN_THRESHOLD_MS = "LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS"
ENV_ERROR_THRESHOLD_MS = "LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS"

MAX_SPANS_PER_MINUTE = 30
MAX_LOGS_PER_MINUTE = 5
MAX_STACK_FRAMES = 20
# a second stack sample once a block has lasted this many warn thresholds
_LATE_SAMPLE_FACTOR = 10

SPAN_NAME = "event_loop_blocked"


@dataclass
class LoopMonitorThresholds:
    warn: float
    error: float

    @classmethod
    def from_env(cls) -> LoopMonitorThresholds | None:
        """Thresholds from the environment, or ``None`` when monitoring is disabled.

        ``LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS=0`` disables the monitor. Invalid values fall
        back to the defaults with a warning rather than disabling silently.
        """
        warn = _env_seconds(ENV_WARN_THRESHOLD_MS, DEFAULT_WARN_THRESHOLD)
        error = _env_seconds(ENV_ERROR_THRESHOLD_MS, DEFAULT_ERROR_THRESHOLD)
        if warn <= 0:
            return None
        if error < warn:
            logger.warning(
                "%s (%.0fms) is below %s (%.0fms); using the warn threshold for both",
                ENV_ERROR_THRESHOLD_MS,
                error * 1000,
                ENV_WARN_THRESHOLD_MS,
                warn * 1000,
            )
            error = warn
        return cls(warn=warn, error=error)


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value_ms = float(raw)
    except ValueError:
        logger.warning(
            "invalid %s=%r, expected milliseconds; using %.0fms", name, raw, default * 1000
        )
        return default
    if value_ms < 0:
        logger.warning("invalid %s=%r, must be >= 0; using %.0fms", name, raw, default * 1000)
        return default
    return value_ms / 1000.0


class _RateLimiter:
    """Allow at most ``limit`` events per rolling 60 s window; count what was refused."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._events: deque[float] = deque()
        self.suppressed = 0

    def allow(self, now: float) -> bool:
        window_start = now - 60.0
        while self._events and self._events[0] < window_start:
            self._events.popleft()
        if len(self._events) >= self._limit:
            self.suppressed += 1
            return False
        self._events.append(now)
        return True

    def take_suppressed(self) -> int:
        n, self.suppressed = self.suppressed, 0
        return n


@dataclass
class _StackSample:
    lag: float
    task_name: str | None
    frames: list[traceback.FrameSummary]


@dataclass
class _Incident:
    """Watchdog-side view of one block, keyed by the heartbeat sequence it interrupted."""

    tick_seq: int
    samples: list[_StackSample] = field(default_factory=list)
    late_sampled: bool = False


class EventLoopMonitor:
    """Watch one event loop for synchronous blocking. See the module docstring."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        warn_threshold: float = DEFAULT_WARN_THRESHOLD,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        tick_interval: float = DEFAULT_TICK_INTERVAL,
        name: str = "event-loop",
    ) -> None:
        if warn_threshold <= 0:
            raise ValueError("warn_threshold must be > 0")
        if error_threshold < warn_threshold:
            raise ValueError("error_threshold must be >= warn_threshold")
        if tick_interval <= 0 or tick_interval > warn_threshold:
            raise ValueError("tick_interval must be > 0 and <= warn_threshold")

        self._loop = loop
        self._warn = warn_threshold
        self._error = error_threshold
        self._tick = tick_interval
        self._name = name

        # written by the loop thread, read by the watchdog
        self._last_tick_at: float = 0.0
        self._tick_seq: int = 0
        self._loop_thread_ident: int | None = None

        # written by the watchdog, read by the loop thread under _lock
        self._lock = threading.Lock()
        self._incident: _Incident | None = None

        self._gc_started_at: float | None = None
        self._gc_time: float = 0.0
        self._last_thread_cpu: float = 0.0

        self._timer: asyncio.TimerHandle | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._closed = False

        self._span_limiter = _RateLimiter(MAX_SPANS_PER_MINUTE)
        self._log_limiter = _RateLimiter(MAX_LOGS_PER_MINUTE)
        self._report_context: contextvars.Context | None = None

        # tests and integrations may observe reports without going through OTel
        self._on_report: Callable[[BlockedReport], None] | None = None

    @property
    def warn_threshold(self) -> float:
        return self._warn

    @property
    def error_threshold(self) -> float:
        return self._error

    def set_report_context(self, ctx: contextvars.Context | None) -> None:
        """Run report emission inside ``ctx``.

        The heartbeat inherits the context of whoever started the monitor, which predates the
        job. Handing it a copy of the job's context (taken inside the ``job_entrypoint`` span)
        lets the emitted spans resolve the job for attribution and parent themselves to the
        session's root span.
        """
        self._report_context = ctx

    def start(self) -> None:
        """Arm the heartbeat. Safe to call from any thread, before or after the loop runs."""
        if self._started or self._closed:
            return
        self._started = True
        self._loop.call_soon_threadsafe(self._arm)

    def stop(self) -> None:
        """Stop the heartbeat and the watchdog. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        with contextlib.suppress(ValueError):
            gc.callbacks.remove(self._on_gc)
        if self._watchdog is not None and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=1.0)

    # -- loop thread --

    def _arm(self) -> None:
        if self._closed:
            return
        self._loop_thread_ident = threading.get_ident()
        self._last_tick_at = time.monotonic()
        self._last_thread_cpu = time.thread_time()
        gc.callbacks.append(self._on_gc)
        self._timer = self._loop.call_later(self._tick, self._on_tick)
        self._watchdog = threading.Thread(
            target=self._watchdog_main, name=f"livekit-loop-monitor-{self._name}", daemon=True
        )
        self._watchdog.start()

    def _on_tick(self) -> None:
        if self._closed:
            return
        now = time.monotonic()
        expected_at = self._last_tick_at + self._tick
        lag = now - expected_at
        thread_cpu = time.thread_time()
        cpu_time = thread_cpu - self._last_thread_cpu
        gc_time, self._gc_time = self._gc_time, 0.0

        blocked_seq = self._tick_seq
        self._tick_seq += 1
        self._last_tick_at = now
        self._last_thread_cpu = thread_cpu
        self._timer = self._loop.call_later(self._tick, self._on_tick)

        if lag < self._warn:
            return

        with self._lock:
            incident = self._incident
            self._incident = None
        samples = (
            incident.samples if incident is not None and incident.tick_seq == blocked_seq else []
        )

        self._report(
            BlockedReport(
                duration=lag,
                # the block started no earlier than the last on-time tick
                started_at=time.time() - lag,
                warn_threshold=self._warn,
                severity="error" if lag >= self._error else "warning",
                gc_time=min(gc_time, lag),
                cpu_time=cpu_time,
                task_name=next((s.task_name for s in samples if s.task_name), None),
                stacks=[_format_frames(s.frames) for s in samples if s.frames],
            )
        )

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        if threading.get_ident() != self._loop_thread_ident:
            return
        if phase == "start":
            self._gc_started_at = time.monotonic()
        elif phase == "stop" and self._gc_started_at is not None:
            self._gc_time += time.monotonic() - self._gc_started_at
            self._gc_started_at = None

    def _report(self, report: BlockedReport) -> None:
        now = time.monotonic()
        emit_span = self._span_limiter.allow(now)
        emit_log = self._log_limiter.allow(now)
        if not emit_span and not emit_log:
            return

        def _emit() -> None:
            if emit_span:
                self._emit_span(report, suppressed=self._span_limiter.take_suppressed())
            if emit_log:
                self._emit_log(report)
            if self._on_report is not None:
                self._on_report(report)

        try:
            if self._report_context is not None:
                self._report_context.run(_emit)
            else:
                _emit()
        except Exception:
            logger.exception("failed to report a blocked event loop")

    def _emit_span(self, report: BlockedReport, *, suppressed: int) -> None:
        start_ns = int(report.started_at * 1_000_000_000)
        end_ns = start_ns + int(report.duration * 1_000_000_000)
        attributes: dict[str, Any] = {
            trace_types.ATTR_BLOCKING_DURATION: report.duration,
            trace_types.ATTR_BLOCKING_THRESHOLD: report.warn_threshold,
            trace_types.ATTR_BLOCKING_SEVERITY: report.severity,
            trace_types.ATTR_BLOCKING_GC_TIME: report.gc_time,
            trace_types.ATTR_BLOCKING_CPU_TIME: report.cpu_time,
        }
        if report.task_name:
            attributes[trace_types.ATTR_BLOCKING_TASK] = report.task_name
        if report.stacks:
            attributes[trace_types.ATTR_BLOCKING_STACK] = "\n---\n".join(report.stacks)
        if suppressed:
            attributes[trace_types.ATTR_BLOCKING_SUPPRESSED] = suppressed

        span = tracer.start_span(
            SPAN_NAME,
            context=_resolve_parent_context(),
            start_time=start_ns,
            attributes=attributes,
        )
        if report.severity == "error":
            span.set_status(
                trace.Status(
                    trace.StatusCode.ERROR,
                    f"event loop blocked for {report.duration * 1000:.0f}ms",
                )
            )
        span.end(end_time=end_ns)
        otel_metrics.record_event_loop_blocked(report.duration, severity=report.severity)

    def _emit_log(self, report: BlockedReport) -> None:
        location = _innermost_location(report.stacks[-1]) if report.stacks else None
        extra: dict[str, Any] = {
            "duration": round(report.duration, 4),
            "threshold": report.warn_threshold,
            "gc_time": round(report.gc_time, 4),
            "cpu_time": round(report.cpu_time, 4),
        }
        if report.task_name:
            extra["task"] = report.task_name
        if report.stacks:
            extra["stack"] = report.stacks[-1]
        where = f" at {location}" if location else ""
        logger.warning(
            "event loop blocked for %.0fms%s; synchronous work on the agent loop delays "
            "audio and turn handling, move it to a thread or an async client",
            report.duration * 1000,
            where,
            extra=extra,
        )

    # -- watchdog thread --

    def _watchdog_main(self) -> None:
        while not self._stop_event.wait(self._tick):
            try:
                self._watchdog_check()
            except Exception:
                logger.exception("event loop watchdog failed")

    def _watchdog_check(self) -> None:
        # snapshot both together: the tick updates seq then time, so a torn read can only
        # make the lag look smaller for one iteration
        seq = self._tick_seq
        lag = time.monotonic() - (self._last_tick_at + self._tick)
        if lag < self._warn:
            return

        with self._lock:
            incident = self._incident
            if incident is None or incident.tick_seq != seq:
                incident = self._incident = _Incident(tick_seq=seq)
            want_first = not incident.samples
            want_late = not incident.late_sampled and lag >= self._warn * _LATE_SAMPLE_FACTOR
        if not (want_first or want_late):
            return

        sample = self._sample_loop_thread(lag)
        with self._lock:
            if self._incident is incident:
                incident.samples.append(sample)
                if want_late:
                    incident.late_sampled = True

    def _sample_loop_thread(self, lag: float) -> _StackSample:
        task_name: str | None = None
        with contextlib.suppress(Exception):
            task = asyncio.current_task(loop=self._loop)
            if task is not None:
                task_name = task.get_name()

        frames: list[traceback.FrameSummary] = []
        ident = self._loop_thread_ident
        if ident is not None:
            frame = sys._current_frames().get(ident)
            if frame is not None:
                frames = list(traceback.extract_stack(frame))
        return _StackSample(lag=lag, task_name=task_name, frames=frames)


@dataclass
class BlockedReport:
    duration: float
    started_at: float
    warn_threshold: float
    severity: str
    gc_time: float
    cpu_time: float
    task_name: str | None
    stacks: list[str]


_ASYNCIO_DIR = os.path.dirname(asyncio.__file__) + os.sep


def _format_frames(frames: list[traceback.FrameSummary]) -> str:
    # drop the event loop machinery at the base of the stack; it is the same in every sample
    trimmed = [f for f in frames if not (f.filename or "").startswith(_ASYNCIO_DIR)]
    if not trimmed:
        trimmed = frames
    trimmed = trimmed[-MAX_STACK_FRAMES:]
    return "".join(traceback.format_list(trimmed)).rstrip()


def _innermost_location(stack: str) -> str | None:
    lines = [ln for ln in stack.splitlines() if ln.lstrip().startswith("File ")]
    if not lines:
        return None
    return lines[-1].strip().removeprefix("File ")


def _resolve_parent_context() -> otel_context.Context | None:
    """The primary agent session's root span context when a job is running, else current."""
    from ..job import get_job_context

    job_ctx = get_job_context(required=False)
    if job_ctx is None:
        return None
    session = job_ctx._primary_agent_session
    if session is not None and session._root_span_context is not None:
        return session._root_span_context
    return None


# -- per-loop registry --

_monitors: dict[asyncio.AbstractEventLoop, EventLoopMonitor] = {}
_registry_lock = threading.Lock()


def start_monitoring(
    loop: asyncio.AbstractEventLoop,
    *,
    thresholds: LoopMonitorThresholds | None = None,
    name: str = "event-loop",
) -> EventLoopMonitor | None:
    """Start monitoring ``loop`` with thresholds from ``thresholds`` or the environment.

    Returns the monitor, or ``None`` when monitoring is disabled or already running for
    that loop.
    """
    if thresholds is None:
        thresholds = LoopMonitorThresholds.from_env()
    if thresholds is None:
        return None

    with _registry_lock:
        if loop in _monitors:
            return None
        monitor = EventLoopMonitor(
            loop,
            warn_threshold=thresholds.warn,
            error_threshold=thresholds.error,
            tick_interval=min(DEFAULT_TICK_INTERVAL, thresholds.warn),
            name=name,
        )
        _monitors[loop] = monitor
    monitor.start()
    return monitor


def stop_monitoring(loop: asyncio.AbstractEventLoop) -> None:
    with _registry_lock:
        monitor = _monitors.pop(loop, None)
    if monitor is not None:
        monitor.stop()


def get_monitor(loop: asyncio.AbstractEventLoop) -> EventLoopMonitor | None:
    with _registry_lock:
        return _monitors.get(loop)
