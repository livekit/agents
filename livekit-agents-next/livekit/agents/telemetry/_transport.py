"""OpenTelemetry SDK capture and invocation-local activation of native-validated carriers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import context as otel_context, trace

from .. import _core
from .._generated.telemetry_transport import TraceCarrier


def capture_trace_carrier() -> TraceCarrier | None:
    parent = trace.get_current_span().get_span_context()
    if not parent.is_valid:
        return None
    return TraceCarrier(
        traceparent=(
            f"00-{parent.trace_id:032x}-{parent.span_id:016x}-{int(parent.trace_flags):02x}"
        )
    )


@contextmanager
def use_trace_carrier(carrier: TraceCarrier | None) -> Iterator[None]:
    parent = _core._core_trace_parent(carrier.traceparent if carrier is not None else None)
    span: trace.Span = trace.INVALID_SPAN
    if parent is not None:
        trace_id, span_id, sampled = parent
        span = trace.NonRecordingSpan(
            trace.SpanContext(
                int(trace_id, 16),
                int(span_id, 16),
                is_remote=True,
                trace_flags=trace.TraceFlags(1 if sampled else 0),
            )
        )
    token = otel_context.attach(trace.set_span_in_context(span))
    try:
        yield
    finally:
        otel_context.detach(token)
