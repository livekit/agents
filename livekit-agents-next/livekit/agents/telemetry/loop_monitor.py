"""Detect synchronous code blocking an asyncio event loop and surface it as telemetry.

A voice pipeline runs VAD, playout, and every provider stream on one event loop, so any
synchronous work on it (a blocking HTTP client inside a tool, a large numpy operation, a
``time.sleep``) is audible as latency or jitter. asyncio only reports slow callbacks in debug
mode, which is far too expensive for production and only logs.

The watchdog is the core's (``_core.LoopMonitor``); this module turns its reports into the
span, log and metric. It observes a single loop without patching asyncio:

* a **heartbeat** scheduled with ``loop.call_later`` every ``tick_interval`` records the time.
  When the loop is blocked, the tick fires late by the length of the block (to within one
  interval);
* a **watchdog thread** wakes on the same interval and compares now against the last tick. Once
  the gap is within one interval of the warn threshold it samples the loop thread's stack through
  ``sys._current_frames()`` (the same call ``faulthandler`` relies on), so the report can say
  *where* the loop was stuck, not just for how long;
* when the late tick finally runs, the block has ended: the tick measures the lag and, if it
  crossed the threshold, emits a back-dated ``event_loop_blocked`` span carrying the sampled
  stack, the task that was running, GC time that fell inside the block, and the loop thread's
  CPU time. It also logs a warning and records a histogram measurement.

Spans nest under the span the blocked task was in when sampled, else under the primary agent
session's root span (``telemetry.session_context``), else under ``job_entrypoint``; the session
span also carries a per-stall event plus a count / total / max summary. The worker process has
no job to attach spans to and produces only the log and the metric. Output is rate limited so
a pathologically blocked loop cannot flood the exporter.

What a report means
-------------------
A report says the loop could not run its timers for ``lk.blocking.duration`` seconds. That is
exactly what delays audio frames and turn handling, whatever the cause:

* one synchronous call (``time.sleep``, a blocking HTTP client, heavy numpy): the stack sample
  lands inside it and names the line;
* one loop iteration with many ready callbacks that add up (a burst of frames each taking a
  few ms): still a stall of that length; the sample lands in whichever callback was running,
  so read it as "the loop was busy here", not as the single culprit;
* the process itself was not running (host CPU contention, container CPU quota, a suspended
  laptop): the watchdog thread woke late too and ``lk.blocking.cpu_time`` is near zero. Reported
  at warning severity with no log, since there is nothing in the code to fix;
* a garbage-collection pass on a large heap: reported separately as ``lk.blocking.gc_time``
  so it is not blamed on the surrounding code.

What is never reported: work handed to ``run_in_executor`` / ``asyncio.to_thread``, or an idle
loop waiting in ``select``; both leave the heartbeat on time.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry import context as otel_context, trace

from .. import _core
from ..log import logger
from . import otel_metrics, session_context, trace_types
from .traces import tracer

DEFAULT_WARN_THRESHOLD = 0.1
"""Blocks at or above this many seconds are reported as warnings."""
DEFAULT_ERROR_THRESHOLD = 0.5
"""Blocks at or above this many seconds are reported as errors (span status ERROR)."""
DEFAULT_TICK_INTERVAL = 0.02
"""Heartbeat period at the default warn threshold. Also the measurement resolution: a block
is measured to within one tick, and detection near the threshold needs several ticks per
threshold (see ``_tick_interval_for``)."""
_TICKS_PER_WARN_THRESHOLD = 5
# the monitor never ticks faster than this: two wake-ups per tick (heartbeat and watchdog) in
# every job process add up. A warn threshold below it is raised to it.
_MIN_TICK_INTERVAL = 0.02
_MAX_TICK_INTERVAL = 0.05

ENV_WARN_THRESHOLD_MS = "LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS"
ENV_ERROR_THRESHOLD_MS = "LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS"

MAX_SPANS_PER_MINUTE = 30
MAX_LOGS_PER_MINUTE = 5

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
    if not math.isfinite(value_ms) or value_ms < 0:
        logger.warning(
            "invalid %s=%r, must be finite and >= 0; using %.0fms", name, raw, default * 1000
        )
        return default
    return value_ms / 1000.0


@dataclass
class BlockedReport:
    duration: float
    started_at: float
    warn_threshold: float
    severity: str
    gc_time: float
    cpu_time: float
    watchdog_gap: float
    process_descheduled: bool
    task_name: str | None
    stacks: list[str]
    importing: str | None = None
    """The module a lazy import was loading when the loop thread was sampled."""
    parent_span_context: trace.SpanContext | None = None
    """The span current in the blocked task when sampled; the stall span nests under it."""
    location: str | None = None
    """The innermost frame of the last sample that is not the import machinery."""


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
        emit_spans: bool = True,
    ) -> None:
        self._emit_spans = emit_spans  # False on the worker loop: no job to attach spans to
        self._core = _core.LoopMonitor(
            loop,
            warn_threshold=warn_threshold,
            error_threshold=error_threshold,
            tick_interval=tick_interval,
            name=name,
            emit_spans=emit_spans,
            report=self._report,
            span_of=_current_span_in,
        )
        self._report_context: contextvars.Context | None = None
        # tests and integrations may observe reports without going through OTel
        self._on_report: Callable[[BlockedReport], None] | None = None

    @property
    def warn_threshold(self) -> float:
        return self._core.warn_threshold  # type: ignore[no-any-return]

    @property
    def error_threshold(self) -> float:
        return self._core.error_threshold  # type: ignore[no-any-return]

    def set_report_context(self, ctx: contextvars.Context | None) -> None:
        """Emit reports inside ``ctx``, a copy of the job's context taken inside ``job_entrypoint``:
        the heartbeat itself inherits the context of whoever started the monitor, which predates
        the job."""
        self._report_context = ctx

    def start(self) -> None:
        """Arm the heartbeat. Safe to call from any thread, before or after the loop runs."""
        self._core.start()

    def stop(self) -> None:
        """Stop the heartbeat and the watchdog. Idempotent."""
        self._core.stop()

    def _report(self, raw: dict[str, Any]) -> None:
        emit_span, emit_log = raw.pop("emit_span"), raw.pop("emit_log")
        suppressed = raw.pop("suppressed")
        report = BlockedReport(**raw)

        def _emit() -> None:
            # the metric and the session summary count every stall, including the ones the
            # span and log limiters drop
            try:
                otel_metrics.record_event_loop_blocked(report.duration, severity=report.severity)
            except Exception:
                logger.exception("failed to record the blocked event loop metric")
            if (session := session_context.primary_session()) is not None and (
                session_context.session_root_context() is not None
            ):
                session._record_loop_stall(
                    report.duration,
                    timestamp_ns=int((report.started_at + report.duration) * 1_000_000_000),
                )
            if not emit_span and not emit_log:
                return
            if emit_span and self._emit_spans:
                self._emit_span(report, suppressed=suppressed)
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
        if report.importing:
            attributes[trace_types.ATTR_BLOCKING_IMPORT] = report.importing
        if suppressed:
            attributes[trace_types.ATTR_BLOCKING_SUPPRESSED] = suppressed

        from ..job import get_job_context

        if get_job_context(required=False) is None:
            return  # no job, no trace to belong to: a root span here would be a stray trace
        # under the span the blocked task was in, else the session root, else the report
        # context (job_entrypoint); the heartbeat's own context predates all of them
        if report.parent_span_context is not None:
            parent: otel_context.Context | None = trace.set_span_in_context(
                trace.NonRecordingSpan(report.parent_span_context)
            )
        else:
            parent = session_context.session_root_context()
        span = tracer.start_span(
            SPAN_NAME, context=parent, start_time=start_ns, attributes=attributes
        )
        if report.severity == "error":
            span.set_status(
                trace.Status(
                    trace.StatusCode.ERROR,
                    f"event loop blocked for {report.duration * 1000:.0f}ms",
                )
            )
        span.end(end_time=end_ns)

    def _emit_log(self, report: BlockedReport) -> None:
        location = report.location
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
        if report.importing:
            extra["import"] = report.importing
        where = f" at {location}" if location else ""
        if report.importing:
            # a lazy import on the loop: name the module, so it can be preloaded
            logger.warning(
                "event loop blocked for %.0fms importing %s%s; import it at process "
                "warm-up (prewarm) instead of on first use",
                report.duration * 1000,
                report.importing,
                where,
                extra=extra,
            )
            return
        logger.warning(
            "event loop blocked for %.0fms%s; synchronous work on the agent loop delays "
            "audio and turn handling, move it to a thread or an async client",
            report.duration * 1000,
            where,
            extra=extra,
        )


def _current_span_in(task_context: contextvars.Context) -> trace.SpanContext | None:
    """The span current in another task's context, read without entering it.

    OpenTelemetry keeps its own ``Context`` in a context variable; a task's
    ``contextvars.Context`` is a mapping of those variables, so the OTel context is one of its
    values. Entering the context is not an option (the blocked task is inside it right now)."""
    for value in task_context.values():
        if isinstance(value, otel_context.Context):
            current = trace.get_current_span(value).get_span_context()
            return current if current.is_valid else None
    return None


# -- per-loop registry --

_monitors: dict[asyncio.AbstractEventLoop, EventLoopMonitor] = {}
_registry_lock = threading.Lock()


def _tick_interval_for(warn_threshold: float) -> float:
    """A fifth of the warn threshold, within bounds. A block is measured to within one tick,
    so fewer ticks per threshold means blocks just over it go unreported; more ticks cost
    wake-ups (two per tick, heartbeat and watchdog) for no gain."""
    return min(
        max(warn_threshold / _TICKS_PER_WARN_THRESHOLD, _MIN_TICK_INTERVAL), _MAX_TICK_INTERVAL
    )


def start_monitoring(
    loop: asyncio.AbstractEventLoop,
    *,
    thresholds: LoopMonitorThresholds | None = None,
    name: str = "event-loop",
    emit_spans: bool = True,
) -> EventLoopMonitor | None:
    """Start monitoring ``loop`` with thresholds from ``thresholds`` or the environment.

    Returns the monitor, or ``None`` when monitoring is disabled or already running for
    that loop.
    """
    if thresholds is None:
        thresholds = LoopMonitorThresholds.from_env()
    if thresholds is None:
        return None
    if thresholds.warn < _MIN_TICK_INTERVAL:
        logger.warning(
            "loop monitor warn threshold %.0fms raised to %.0fms, the smallest it measures",
            thresholds.warn * 1000,
            _MIN_TICK_INTERVAL * 1000,
        )
        thresholds = LoopMonitorThresholds(
            warn=_MIN_TICK_INTERVAL, error=max(thresholds.error, _MIN_TICK_INTERVAL)
        )

    with _registry_lock:
        if loop in _monitors:
            return None
        monitor = EventLoopMonitor(
            loop,
            warn_threshold=thresholds.warn,
            error_threshold=thresholds.error,
            tick_interval=_tick_interval_for(thresholds.warn),
            name=name,
            emit_spans=emit_spans,
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
