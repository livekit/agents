"""Invocation-local OpenTelemetry context and framework tracer selection."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as otel_context, trace, trace as trace_api
from opentelemetry.trace import Span, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AttributeValue

from . import trace_types, utils as telemetry_utils

if TYPE_CHECKING:
    from ..job import JobContext
    from ..voice.agent_session import VoiceSession

_USE_SPAN_SIGNATURE = inspect.signature(trace_api.use_span)


_START_SPAN_SIGNATURE = inspect.signature(Tracer.start_span)


_START_AS_CURRENT_SPAN_SIGNATURE = inspect.signature(Tracer.start_as_current_span)


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

    def start_span(self, *args: Any, **kwargs: Any) -> Span:
        bound = _START_SPAN_SIGNATURE.bind(self._tracer, *args, **kwargs)
        bound.arguments["attributes"] = _with_conversation_id(bound.arguments.get("attributes"))
        return self._tracer.start_span(*bound.args[1:], **bound.kwargs)

    @_agnosticcontextmanager
    def use_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        bound = _USE_SPAN_SIGNATURE.bind(*args, **kwargs)
        record_exception = bound.arguments.get("record_exception", True)
        set_status_on_exception = bound.arguments.get("set_status_on_exception", True)
        bound.arguments.update(record_exception=False, set_status_on_exception=False)
        with trace_api.use_span(*bound.args, **bound.kwargs) as span:
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
        attributes: dict[str, Any] | None = None,
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
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        bound = _START_AS_CURRENT_SPAN_SIGNATURE.bind(self._tracer, *args, **kwargs)
        bound.arguments["attributes"] = _with_conversation_id(bound.arguments.get("attributes"))
        record_exception = bound.arguments.get("record_exception", True)
        set_status_on_exception = bound.arguments.get("set_status_on_exception", True)
        bound.arguments.update(record_exception=False, set_status_on_exception=False)
        with self._tracer.start_as_current_span(*bound.args[1:], **bound.kwargs) as span:
            try:
                yield span
            except Exception as exc:
                telemetry_utils.record_exception(
                    span, exc, record_event=record_exception, set_status=set_status_on_exception
                )
                raise


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
