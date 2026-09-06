"""Spans that belong to the agent session but happen before it exists.

The cloud trace view is organised around ``agent_session``: anything not under that span is
effectively invisible. Some work the session should show happens before ``AgentSession.start()``
creates the span, most commonly the room connection and participant wait when the entrypoint
calls ``ctx.connect()`` first, and any event loop stall during that window. Those are recorded
here as :class:`RecordedSpan` objects and emitted as back-dated children of ``agent_session``
the moment the session starts. Their original start and end times are kept, so they land where
they happened on the timeline, ahead of ``session_start``.

Once a session exists, :func:`session_span` is a plain span in the current context. Outside a
job there is no session to wait for, so it is a plain span there too.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context, trace
from opentelemetry.util.types import Attributes

from . import utils as trace_utils
from .traces import tracer

if TYPE_CHECKING:
    from ..job import JobContext
    from ..voice.agent_session import AgentSession


@dataclass
class _RecordedEvent:
    name: str
    attributes: Attributes
    timestamp_ns: int


@dataclass
class RecordedSpan:
    """A span captured now and created later. Mirrors the parts of the ``Span`` API the
    recording code uses, so a call site can treat it and a real span alike."""

    name: str
    start_ns: int
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[_RecordedEvent] = field(default_factory=list)
    status: trace.Status | None = None
    exception: Exception | None = None
    end_ns: int | None = None

    @property
    def duration(self) -> float:
        end_ns = self.end_ns if self.end_ns is not None else time.time_ns()
        return max(end_ns - self.start_ns, 0) / 1_000_000_000

    def is_recording(self) -> bool:
        return self.end_ns is None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def add_event(
        self, name: str, attributes: Attributes = None, timestamp: int | None = None
    ) -> None:
        self.events.append(_RecordedEvent(name, attributes, timestamp or time.time_ns()))

    def set_status(self, status: trace.Status) -> None:
        self.status = status

    def record_exception(self, exception: Exception) -> None:
        # rendered at emit time through trace_utils.record_exception so redaction applies
        self.exception = exception

    def end(self, end_time: int | None = None) -> None:
        if self.end_ns is None:
            self.end_ns = end_time or time.time_ns()

    def emit(self, parent: otel_context.Context | None) -> trace.Span:
        """Create the real span under ``parent`` with the recorded timing and contents."""
        self.end()
        span = tracer.start_span(
            self.name,
            context=parent,
            kind=self.kind,
            start_time=self.start_ns,
            attributes=self.attributes,
        )
        for ev in self.events:
            span.add_event(ev.name, ev.attributes, timestamp=ev.timestamp_ns)
        if self.exception is not None:
            trace_utils.record_exception(span, self.exception)
        elif self.status is not None:
            span.set_status(self.status)
        span.end(end_time=self.end_ns)
        return span


def primary_session() -> AgentSession | None:
    from ..job import get_job_context

    job_ctx = get_job_context(required=False)
    return job_ctx._primary_agent_session if job_ctx is not None else None


def session_root_context() -> otel_context.Context | None:
    """The primary agent session's root span context, if a session is running."""
    session = primary_session()
    if session is not None and session._root_span_context is not None:
        return session._root_span_context
    return None


def defer_to_session(recorded: RecordedSpan) -> bool:
    """Queue ``recorded`` for emission under ``agent_session`` when the session starts.

    Returns False when there is no job to hold it (the worker process): the caller keeps
    whatever log it has, since there will never be a session to attach the span to."""
    from ..job import get_job_context

    job_ctx: JobContext | None = get_job_context(required=False)
    if job_ctx is None:
        return False
    job_ctx._defer_session_span(recorded)
    return True


@contextmanager
def session_span(
    name: str, *, attributes: dict[str, Any] | None = None
) -> Iterator[trace.Span | RecordedSpan]:
    """A span that is guaranteed to show up under ``agent_session``.

    With a session running (or outside a job) this is an ordinary span in the current
    context. Before the session exists it records the span and defers it, so work done ahead
    of ``AgentSession.start()`` still appears as the session's opening act."""
    from ..job import get_job_context

    if session_root_context() is not None or get_job_context(required=False) is None:
        with tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span
        return

    recorded = RecordedSpan(name, start_ns=time.time_ns(), attributes=dict(attributes or {}))
    try:
        yield recorded
    except Exception as e:
        recorded.record_exception(e)
        raise
    finally:
        recorded.end()
        defer_to_session(recorded)
