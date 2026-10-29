"""Invocation-local OpenTelemetry context and framework tracer selection."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, cast

from opentelemetry import context as otel_context, trace, trace as trace_api
from opentelemetry.trace import Link, Span, SpanKind, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AttributeValue

from . import trace_types, utils as telemetry_utils

if TYPE_CHECKING:
    from ..job import JobContext
    from ..voice.agent_session import VoiceSession


def _with_conversation_id(
    attributes: Mapping[str, AttributeValue] | None,
) -> Mapping[str, AttributeValue] | None:
    captured = {
        "lk.telemetry.capture_content": capture_content_enabled(),
        **(attributes or {}),
    }
    if (conversation_id := _conversation_id()) is None:
        return captured
    return {trace_types.ATTR_GEN_AI_CONVERSATION_ID: conversation_id, **captured}


class _DynamicTracer(Tracer):
    def __init__(self, instrumenting_module_name: str) -> None:
        self._instrumenting_module_name = instrumenting_module_name
        self._explicit_provider = False
        self._tracer_provider: trace_api.TracerProvider = trace_api.get_tracer_provider()
        self._tracer = trace_api.get_tracer(instrumenting_module_name)

    def set_provider(
        self, tracer_provider: trace_api.TracerProvider, *, explicit: bool = True
    ) -> None:
        self._explicit_provider = explicit
        self._tracer_provider = tracer_provider
        self._tracer = trace_api.get_tracer(
            self._instrumenting_module_name,
            tracer_provider=self._tracer_provider,
        )

    def start_span(
        self,
        name: str,
        context: otel_context.Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        return self._tracer.start_span(
            name,
            context=context,
            kind=kind,
            attributes=_with_conversation_id(attributes),
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )

    @_agnosticcontextmanager
    def use_span(
        self,
        span: Span,
        end_on_exit: bool = False,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Iterator[Span]:
        with trace_api.use_span(
            span, end_on_exit=end_on_exit, record_exception=False, set_status_on_exception=False
        ):
            try:
                yield span
            except Exception as exc:
                telemetry_utils.record_exception(
                    span, exc, record_event=record_exception, set_status=set_status_on_exception
                )
                raise

    @_agnosticcontextmanager
    def detached_span(
        self,
        name: str,
        *,
        context: otel_context.Context | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[Span]:
        """A span that is never made current.

        For code that spawns long-lived tasks (connecting the room, publishing a track,
        connecting MCP servers): a *current* span is inherited by every task created under
        it and becomes the accidental parent of unrelated spans those tasks emit for the rest
        of the session. The parent is ``context`` when given, else the ambient context; the
        exception, if any, is recorded redaction-aware and the span is ended."""
        span = self.start_span(name, context=context, attributes=attributes)
        try:
            yield span
        except Exception as e:
            telemetry_utils.record_exception(span, e)
            raise
        finally:
            span.end()

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: otel_context.Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        span = self.start_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=False,
            set_status_on_exception=False,
        )
        with self.use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ):
            yield span


tracer: _DynamicTracer = _DynamicTracer("livekit-agents")


_TRUTHY = ("1", "true", "yes", "on")


# the env var name the GenAI conventions standardise for this opt-in
_capture_content: bool = (
    os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").strip().lower()
    in _TRUTHY
)


def set_capture_content(enabled: bool) -> None:
    """Whether spans carry message content (judge explanations); off by default unless
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is set."""
    global _capture_content
    _capture_content = enabled


def capture_content_enabled() -> bool:
    return _capture_content


def _conversation_id() -> str | None:
    """The convention forbids fabricating one (no UUIDs, trace ids or content hashes),
    so this is the room sid LiveKit already stamps on every record."""
    from ..job import get_job_context

    ctx = get_job_context(required=False)
    if ctx is None:
        return None
    return ctx.job.room.sid or None


def set_error_type(span: trace.Span, error: BaseException | str) -> None:
    """A low-cardinality identifier, never the error message: that is free-form and may
    carry user data."""
    if not span.is_recording():
        return
    if isinstance(error, str):
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, error)
        return

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, str(status_code))
        return
    span.set_attribute(trace_types.ATTR_ERROR_TYPE, type(error).__qualname__)


def _current_job(job_ctx: JobContext | None) -> JobContext | None:
    if job_ctx is not None:
        return job_ctx
    from ..job import get_job_context

    return get_job_context(required=False)


def primary_session(job_ctx: JobContext | None = None) -> VoiceSession | None:
    job_ctx = _current_job(job_ctx)
    return job_ctx._primary_agent_session if job_ctx is not None else None


def session_root_context(job_ctx: JobContext | None = None) -> otel_context.Context | None:
    """The primary agent session's root span context, if a session is running."""
    session = primary_session(job_ctx)
    if session is not None and session._root_span_context is not None:
        return cast(otel_context.Context, session._root_span_context)
    return None
