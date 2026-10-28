"""Detect synchronous code blocking an asyncio event loop and surface it as telemetry.

A voice pipeline runs VAD, playout, and every provider stream on one event loop, so any
synchronous work on it (a blocking HTTP client inside a tool, a large numpy operation, a
``time.sleep``) is audible as latency or jitter. asyncio only reports slow callbacks in debug
mode, which is far too expensive for production and only logs.

The watchdog is the core's (``_core.LoopMonitor``), and so are the thresholds (from the
environment), the report's span attributes and log line, and their rate limits; this module
emits them and records the metric. It observes a single loop without patching asyncio:

* a **heartbeat** scheduled with ``loop.call_later`` every tick interval records the time.
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
session's root span (``telemetry.traces.session_root_context``), else under ``job_entrypoint``;
the session span also carries a per-stall event plus a count / total / max summary. The worker
process has no job to attach spans to and produces only the log and the metric. Output is rate
limited so a pathologically blocked loop cannot flood the exporter.

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
import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from opentelemetry import context as otel_context, trace
from opentelemetry.util.types import AttributeValue

from .. import _core
from .._generated.telemetry import BlockedReport, LoopMonitorEmission
from ..log import logger
from .metrics import record_event_loop_blocked
from .traces import primary_session, session_root_context, tracer

DEFAULT_WARN_THRESHOLD = 0.1
"""Blocks at or above this many seconds are reported as warnings."""
DEFAULT_ERROR_THRESHOLD = 0.5
"""Blocks at or above this many seconds are reported as errors (span status ERROR)."""

SPAN_NAME = "event_loop_blocked"


@dataclass
class LoopMonitorThresholds:
    warn: float
    error: float

    @classmethod
    def from_env(cls) -> LoopMonitorThresholds | None:
        """Thresholds from ``LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS`` and ``_ERROR_MS``, or ``None``
        when monitoring is disabled (a warn threshold of 0). Invalid values fall back to the
        defaults with a warning rather than disabling silently."""
        thresholds = _core._core_loop_monitor_thresholds(
            os.environ.get("LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS"),
            os.environ.get("LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS"),
        )
        return cls(*thresholds) if thresholds is not None else None


class EventLoopMonitor:
    """Watch one event loop for synchronous blocking. See the module docstring."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        warn_threshold: float = DEFAULT_WARN_THRESHOLD,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        tick_interval: float | None = None,
        name: str = "event-loop",
        emit_spans: bool = True,
    ) -> None:
        """``tick_interval`` defaults to a fifth of the warn threshold, within bounds; a warn
        threshold below the smallest tick is then raised to it. ``emit_spans=False`` on the
        worker loop: no job to attach spans to."""
        owner = weakref.ref(self)

        def report(emission: LoopMonitorEmission) -> None:
            if (monitor := owner()) is not None:
                monitor._report(emission)

        self._core = _core.LoopMonitor(
            loop,
            warn_threshold=warn_threshold,
            error_threshold=error_threshold,
            tick_interval=tick_interval,
            name=name,
            emit_spans=emit_spans,
            report=report,
            span_of=_current_span_in,
        )
        self._report_context: contextvars.Context | None = None
        # tests and integrations may observe reports without going through OTel
        self._on_report: Callable[[BlockedReport], None] | None = None

    @property
    def warn_threshold(self) -> float:
        return self._core.warn_threshold

    @property
    def error_threshold(self) -> float:
        return self._core.error_threshold

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

    def _report(self, emission: LoopMonitorEmission) -> None:
        report = emission.report

        def _emit() -> None:
            # the metric counts every stall, including the ones the limiters drop
            try:
                record_event_loop_blocked(report.duration, severity=report.severity)
            except Exception:
                logger.exception("failed to record the blocked event loop metric")
            # the session summary counts every stall too
            if (session := primary_session()) is not None:
                session._record_loop_stall(
                    report.duration,
                    timestamp_ns=int((report.started_at + report.duration) * 1_000_000_000),
                )
            if emission.span is not None:
                _emit_span(
                    report,
                    cast(dict[str, AttributeValue], emission.span.attributes),
                    emission.span.error,
                )
            if emission.log is not None:
                logger.warning(emission.log.message, extra=emission.log.extra)
            if emission.reported and self._on_report is not None:
                self._on_report(report)

        try:
            if self._report_context is not None:
                self._report_context.run(_emit)
            else:
                _emit()
        except Exception:
            logger.exception("failed to report a blocked event loop")


def _emit_span(
    report: BlockedReport, attributes: dict[str, AttributeValue], error: str | None
) -> None:
    from ..job import get_job_context

    if get_job_context(required=False) is None:
        return  # no job, no trace to belong to: a root span here would be a stray trace
    # under the span the blocked task was in, else the session root, else the report context
    # (job_entrypoint); the heartbeat's own context predates all of them
    if report.parent_span_context is not None:
        parent: otel_context.Context | None = trace.set_span_in_context(
            trace.NonRecordingSpan(cast(trace.SpanContext, report.parent_span_context))
        )
    else:
        parent = session_root_context()
    start_ns = int(report.started_at * 1_000_000_000)
    span = tracer.start_span(SPAN_NAME, context=parent, start_time=start_ns, attributes=attributes)
    if error is not None:
        span.set_status(trace.Status(trace.StatusCode.ERROR, error))
    span.end(end_time=start_ns + int(report.duration * 1_000_000_000))


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
    if (thresholds := thresholds or LoopMonitorThresholds.from_env()) is None:
        return None
    with _registry_lock:
        if loop in _monitors:
            return None
        monitor = _monitors[loop] = EventLoopMonitor(
            loop,
            warn_threshold=thresholds.warn,
            error_threshold=thresholds.error,
            name=name,
            emit_spans=emit_spans,
        )
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
