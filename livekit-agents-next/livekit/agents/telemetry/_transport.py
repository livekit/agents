"""OpenTelemetry SDK capture and invocation-local activation of native-validated carriers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import context as otel_context, trace

from .. import _core


def capture_traceparent() -> str | None:
    parent = trace.get_current_span().get_span_context()
    if not parent.is_valid:
        return None
    return _core._core_format_trace_parent(
        parent.trace_id, parent.span_id, parent.trace_flags.sampled
    )


@contextmanager
def use_traceparent(traceparent: str | None) -> Iterator[None]:
    parent = _core._core_trace_parent(traceparent)
    span: trace.Span = trace.INVALID_SPAN
    if parent is not None:
        trace_id, span_id, sampled = parent
        span = trace.NonRecordingSpan(
            trace.SpanContext(
                trace_id,
                span_id,
                is_remote=True,
                trace_flags=trace.TraceFlags(1 if sampled else 0),
            )
        )
    token = otel_context.attach(trace.set_span_in_context(span))
    try:
        yield
    finally:
        otel_context.detach(token)
