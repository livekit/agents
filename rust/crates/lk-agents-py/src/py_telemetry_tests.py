"""The OpenTelemetry API backed by the Rust core, and the bridge from an application SDK."""

import pytest
from opentelemetry import context as otel_context, trace

from livekit.agents import _core, telemetry

pytestmark = pytest.mark.unit


def test_current_span_is_an_api_span_and_parents_application_spans() -> None:
    tracer = telemetry.TracerProvider().get_tracer("clinic")
    with tracer.start_as_current_span("clinic.lookup", attributes={"clinic.id": "c1"}) as outer:
        current = trace.get_current_span()
        assert current is outer and isinstance(current, trace.Span)
        inner = tracer.start_span("clinic.inner")
        assert inner.get_span_context().trace_id == outer.get_span_context().trace_id
        assert telemetry._current()[:2] == (
            outer.get_span_context().trace_id,
            outer.get_span_context().span_id,
        )
        inner.end()
    assert not outer.is_recording()
    assert telemetry._current() == (0, 0, False)


def test_start_as_current_span_decorates_and_records_failures() -> None:
    tracer = telemetry.TracerProvider().get_tracer("clinic")
    seen: list[trace.Span] = []

    @tracer.start_as_current_span("clinic.decorated")
    def decorated() -> None:
        seen.append(trace.get_current_span())
        raise ValueError("no slot")

    with pytest.raises(ValueError):
        decorated()
    assert isinstance(seen[0], telemetry.Span) and not seen[0].is_recording()


def test_an_activated_framework_span_parents_application_spans() -> None:
    tracer = telemetry.TracerProvider().get_tracer("clinic")
    token = otel_context.attach(otel_context.Context())
    try:
        telemetry._activate(0xABC, 0x42, True)
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
    real = _core.TelemetrySpan

    def record(name: str, scope: str, **kwargs: object) -> object:
        span = real(name, scope, **kwargs)
        forwarded.append((name, span.context[1]))
        return span

    _core.TelemetrySpan = record  # type: ignore[attr-defined]
    try:
        provider.add_span_processor(bridge)
        with provider.get_tracer("app").start_as_current_span("app.work") as span:
            span_id = span.get_span_context().span_id
    finally:
        _core.TelemetrySpan = real  # type: ignore[attr-defined]
    assert forwarded == [("app.work", span_id)]


def test_configure_refuses_an_unknown_protocol_and_diagnostics_count() -> None:
    with pytest.raises(ValueError, match="http/json"):
        telemetry.configure(endpoint="http://127.0.0.1:1", protocol="grpc")
    assert set(telemetry.diagnostics()) == {
        "dropped",
        "failed",
        "conflicts",
        "incomplete_cleanup",
        "swept_spans",
        "abandoned_sessions",
    }
