"""Trace-context helpers for the primary agent session.

The job's trace is rooted at ``job_entrypoint`` and ``agent_session`` is a child of it, so
work done before or after the session lands under the job in the ambient context and needs
no special handling. What does need help is code running on a task whose context predates
the session, the event loop monitor's heartbeat above all: it resolves the running session
through the job so its spans nest under ``agent_session`` while one exists.
:func:`session_span` nests startup work under ``session_start`` while the session is starting.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context, trace

from .traces import tracer

if TYPE_CHECKING:
    from ..job import JobContext
    from ..voice.agent_session import AgentSession


def _current_job(job_ctx: JobContext | None) -> JobContext | None:
    if job_ctx is not None:
        return job_ctx
    from ..job import get_job_context

    return get_job_context(required=False)


def primary_session(job_ctx: JobContext | None = None) -> AgentSession | None:
    job_ctx = _current_job(job_ctx)
    return job_ctx._primary_agent_session if job_ctx is not None else None


def session_root_context(job_ctx: JobContext | None = None) -> otel_context.Context | None:
    """The primary agent session's root span context, if a session is running."""
    session = primary_session(job_ctx)
    if session is not None and session._root_span_context is not None:
        return session._root_span_context
    return None


@contextmanager
def session_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    job_ctx: JobContext | None = None,
) -> Iterator[trace.Span]:
    """A span for work that belongs to the session's story wherever it runs: under
    ``agent_session`` once the session exists, under ``job_entrypoint`` before it. Both are
    the ambient context at those points, so this is an ordinary current span. ``job_ctx`` is
    accepted from callers that have it at hand (the ``JobContext`` methods themselves)."""
    del job_ctx  # the ambient context already carries the right parent
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span
