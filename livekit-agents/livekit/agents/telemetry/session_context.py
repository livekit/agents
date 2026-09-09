"""Trace-context helpers for the primary agent session.

The job's trace is rooted at ``job_entrypoint`` and ``agent_session`` is a child of it, so
work done before or after the session lands under the job in the ambient context and needs
no special handling. What does need help is code running on a task whose context predates
the session, the event loop monitor's heartbeat above all: it resolves the running session
through the job so its spans nest under ``agent_session`` while one exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import context as otel_context

if TYPE_CHECKING:
    from ..voice.agent_session import AgentSession


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
