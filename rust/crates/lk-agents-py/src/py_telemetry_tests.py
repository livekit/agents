"""The OpenTelemetry API backed by the Rust core, and the bridge from an application SDK."""

from typing import Any

import pytest
from opentelemetry import context as otel_context, trace

from livekit.agents import _core, telemetry

pytestmark = pytest.mark.unit


def test_current_span_is_an_api_span_and_parents_application_spans() -> None:
    tracer = telemetry._TracerProvider().get_tracer("clinic")
    with tracer.start_as_current_span("clinic.lookup", attributes={"clinic.id": "c1"}) as outer:
        current = trace.get_current_span()
        assert current is outer and isinstance(current, trace.Span)
        inner = tracer.start_span("clinic.inner")
        assert inner.get_span_context().trace_id == outer.get_span_context().trace_id
        assert telemetry._current() == outer.get_span_context()
        inner.end()
    assert not outer.is_recording()
    assert not telemetry._current().is_valid


def test_start_as_current_span_decorates_and_records_failures() -> None:
    tracer = telemetry._TracerProvider().get_tracer("clinic")
    seen: list[trace.Span] = []

    @tracer.start_as_current_span("clinic.decorated")
    def decorated() -> None:
        seen.append(trace.get_current_span())
        raise ValueError("no slot")

    with pytest.raises(ValueError):
        decorated()
    assert isinstance(seen[0], telemetry._Span) and not seen[0].is_recording()


def test_an_activated_framework_span_parents_application_spans() -> None:
    tracer = telemetry._TracerProvider().get_tracer("clinic")
    token = otel_context.attach(otel_context.Context())
    try:
        sampled = trace.TraceFlags(trace.TraceFlags.SAMPLED)
        telemetry._activate(trace.SpanContext(0xABC, 0x42, is_remote=False, trace_flags=sampled))
        span = tracer.start_span("clinic.child")
        assert span.get_span_context().trace_id == 0xABC
        span.end()
    finally:
        otel_context.detach(token)


def test_the_bridge_forwards_an_sdk_span_with_its_ids() -> None:
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    provider = sdk.TracerProvider()
    forwarded: list[tuple[str, int]] = []
    bridge = telemetry._bridge()
    real = _core._core_telemetry_export

    def export(span: Any, **kwargs: Any) -> None:
        forwarded.append((span.name, span.get_span_context().span_id))
        real(span, **kwargs)

    _core._core_telemetry_export = export  # type: ignore[attr-defined]
    try:
        provider.add_span_processor(bridge)
        with provider.get_tracer("app").start_as_current_span("app.work") as span:
            span_id = span.get_span_context().span_id
    finally:
        _core._core_telemetry_export = real  # type: ignore[attr-defined]
    assert forwarded == [("app.work", span_id)]


def test_diagnostics_count() -> None:
    d = telemetry.diagnostics()
    assert (
        min(
            d.dropped,
            d.failed,
            d.conflicts,
            d.incomplete_cleanup,
            d.swept_spans,
            d.abandoned_sessions,
        )
        >= 0
    )
