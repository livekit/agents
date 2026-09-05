"""RPC tracing (``telemetry.rpc``).

The interceptor is exercised directly with fake ``next`` continuations, so these tests run
against any ``livekit-rtc`` version. With an SDK that has ``RpcInterceptor`` support the same
interceptor is what ``install`` registers on the local participant; without it, ``install`` is
a no-op, which is also covered."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import rtc
from livekit.agents.telemetry import rpc as rpc_tracing, set_tracer_provider, trace_types, tracer

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _call(**overrides: object) -> SimpleNamespace:
    # shaped like rtc.RpcCallInfo (livekit>=1.1.18)
    base: dict[str, object] = {
        "destination_identity": "avatar-1",
        "method": "playback.start",
        "payload": '{"id": 7}',
        "response_timeout": 5.0,
        "max_round_trip_latency": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _invocation(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "request_id": "req-42",
        "caller_identity": "client-9",
        "payload": '{"q": "x"}',
        "response_timeout": 10.0,
        "method": "agent.lookup",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def test_outgoing_call_span_nests_under_the_caller(
    span_exporter: InMemorySpanExporter,
) -> None:
    interceptor = rpc_tracing.TracingRpcInterceptor()

    async def next_(call: object) -> str:
        return "ok!"

    with tracer.start_as_current_span("function_tool") as parent:
        result = await interceptor.intercept_outgoing(_call(), next_)

    assert result == "ok!"
    [span] = _spans(span_exporter, "rpc_call")
    assert span.kind == trace.SpanKind.CLIENT
    assert span.parent is not None and span.parent.span_id == parent.get_span_context().span_id
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_RPC_SYSTEM] == "livekit"
    assert attrs[trace_types.ATTR_RPC_METHOD] == "playback.start"
    assert attrs[trace_types.ATTR_RPC_DESTINATION_IDENTITY] == "avatar-1"
    assert attrs[trace_types.ATTR_RPC_PAYLOAD] == '{"id": 7}'
    assert attrs[trace_types.ATTR_RPC_PAYLOAD_SIZE] == 9
    assert attrs[trace_types.ATTR_RPC_RESPONSE_SIZE] == 3
    assert attrs[trace_types.ATTR_RPC_RESPONSE_TIMEOUT] == 5.0
    assert span.status.status_code == trace.StatusCode.UNSET


async def test_outgoing_error_records_code_and_status(span_exporter: InMemorySpanExporter) -> None:
    interceptor = rpc_tracing.TracingRpcInterceptor()

    async def next_(call: object) -> str:
        raise rtc.RpcError._built_in(rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND)

    with pytest.raises(rtc.RpcError):
        await interceptor.intercept_outgoing(_call(), next_)

    [span] = _spans(span_exporter, "rpc_call")
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_RPC_ERROR_CODE] == int(rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND)
    assert span.status.status_code == trace.StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


async def test_payload_is_truncated_and_size_kept(span_exporter: InMemorySpanExporter) -> None:
    interceptor = rpc_tracing.TracingRpcInterceptor()
    payload = "x" * (rpc_tracing.MAX_PAYLOAD_ATTR_LEN + 500)

    async def next_(call: object) -> str:
        return ""

    await interceptor.intercept_outgoing(_call(payload=payload, response_timeout=None), next_)

    [span] = _spans(span_exporter, "rpc_call")
    attrs = span.attributes or {}
    assert len(attrs[trace_types.ATTR_RPC_PAYLOAD]) == rpc_tracing.MAX_PAYLOAD_ATTR_LEN
    assert attrs[trace_types.ATTR_RPC_PAYLOAD_SIZE] == len(payload)
    assert trace_types.ATTR_RPC_RESPONSE_TIMEOUT not in attrs
    assert attrs[trace_types.ATTR_RPC_RESPONSE_SIZE] == 0


async def test_incoming_handler_span(span_exporter: InMemorySpanExporter) -> None:
    interceptor = rpc_tracing.TracingRpcInterceptor()

    async def next_(invocation: object) -> str | None:
        return "found"

    result = await interceptor.intercept_incoming(_invocation(), next_)

    assert result == "found"
    [span] = _spans(span_exporter, "rpc_handler")
    assert span.kind == trace.SpanKind.SERVER
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_RPC_METHOD] == "agent.lookup"
    assert attrs[trace_types.ATTR_RPC_REQUEST_ID] == "req-42"
    assert attrs[trace_types.ATTR_RPC_CALLER_IDENTITY] == "client-9"
    assert attrs[trace_types.ATTR_RPC_HANDLER_REGISTERED] is True
    assert attrs[trace_types.ATTR_RPC_RESPONSE_SIZE] == 5


async def test_incoming_unregistered_method_is_flagged(span_exporter: InMemorySpanExporter) -> None:
    interceptor = rpc_tracing.TracingRpcInterceptor()

    async def next_(invocation: object) -> str | None:
        raise rtc.RpcError._built_in(rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD)

    with pytest.raises(rtc.RpcError):
        await interceptor.intercept_incoming(_invocation(method="nope"), next_)

    [span] = _spans(span_exporter, "rpc_handler")
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_RPC_HANDLER_REGISTERED] is False
    assert attrs[trace_types.ATTR_RPC_ERROR_CODE] == int(rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD)
    assert span.status.status_code == trace.StatusCode.ERROR


async def test_incoming_handler_exception_is_recorded(span_exporter: InMemorySpanExporter) -> None:
    interceptor = rpc_tracing.TracingRpcInterceptor()

    async def next_(invocation: object) -> str | None:
        raise ValueError("bad request body")

    with pytest.raises(ValueError):
        await interceptor.intercept_incoming(_invocation(), next_)

    [span] = _spans(span_exporter, "rpc_handler")
    assert span.status.status_code == trace.StatusCode.ERROR
    assert (span.attributes or {})[trace_types.ATTR_RPC_HANDLER_REGISTERED] is True


def test_install_registers_once_or_degrades() -> None:
    supported = MagicMock()
    assert rpc_tracing.install(supported) is True
    assert rpc_tracing.install(supported) is True
    # the same singleton interceptor each time: the SDK dedups by identity
    (first,), (second,) = (c.args for c in supported.add_rpc_interceptor.call_args_list)
    assert first is second is rpc_tracing._interceptor

    unsupported = SimpleNamespace(identity="agent")  # no add_rpc_interceptor: older SDK
    assert rpc_tracing.install(unsupported) is False


def test_interceptor_is_an_sdk_interceptor_when_the_sdk_has_them() -> None:
    base = getattr(rtc, "RpcInterceptor", None)
    if base is None:
        pytest.skip("installed livekit-rtc predates RpcInterceptor")
    assert isinstance(rpc_tracing._interceptor, base)
