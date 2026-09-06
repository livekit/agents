"""Trace-context helpers for the primary agent session.

The job's trace is rooted at ``job_entrypoint`` and ``agent_session`` is a child of it, so
work done before or after the session (the room connect in the entrypoint, an event loop
stall while models load, the job's shutdown) lands under the job in the ambient context and
needs no special handling. What does need help is code running on a task whose context
predates the session, the event loop monitor's heartbeat above all: it resolves the running
session through the job (:func:`session_root_context`) so its spans nest under
``agent_session`` while one exists. :func:`session_span` nests startup work under
``session_start`` while the session is starting.
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
    """A span for work that belongs to the session's story wherever it runs.

    While the primary session is starting it nests under ``session_start`` with the rest of
    the startup work; otherwise the ambient context is right already: ``agent_session`` once
    the session exists, ``job_entrypoint`` before it. Never made current: ``room.connect()``
    spawns the room's event tasks, and a current span here would become the parent of every
    span they later emit (see ``detached_span``). ``job_ctx`` lets the ``JobContext`` methods
    resolve the session regardless of which task or thread they run on."""
    session = primary_session(_current_job(job_ctx))
    parent = session._session_start_context if session is not None else None
    with tracer.detached_span(name, context=parent, attributes=attributes) as span:
        yield span
