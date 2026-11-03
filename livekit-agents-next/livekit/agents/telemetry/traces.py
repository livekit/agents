"""Session report option serialization and RTC RPC instrumentation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from typing import Protocol, runtime_checkable

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue
from pydantic import JsonValue

from livekit import rtc

from .. import _core
from ..log import logger
from . import trace_types
from ._context import (
    capture_content_enabled as capture_content_enabled,
    session_root_context as session_root_context,
    set_capture_content as set_capture_content,
    set_error_type as set_error_type,
    tracer as tracer,
)
from ._providers import (
    _cloud_log_handler as _cloud_log_handler,
    _discard_cloud_tracer as _discard_cloud_tracer,
    _prepare_cloud_tracer as _prepare_cloud_tracer,
    _setup_cloud_tracer as _setup_cloud_tracer,
    _shutdown_telemetry as _shutdown_telemetry,
    set_tracer_provider as set_tracer_provider,
)
from ._redaction import (
    RedactingLogExporter as RedactingLogExporter,
    RedactingSpanExporter as RedactingSpanExporter,
    filter_attributes as filter_attributes,
    redact as redact,
)

# Public, non-callable attributes worth showing when a model-like object (turn detector,
# interruption detector, ...) appears in the session options. Read in this order; missing,
# NOT_GIVEN and None values are skipped. Kept to a whitelist so a plugin's credentials or
# internals never end up in the report.
_OPTION_PRIMITIVES = (str, bool, int, float)


@runtime_checkable
class DescribesOptions(Protocol):
    """An object that can appear in ``VoiceSession`` options (a turn detector, a model) and
    wants the session report to show its configuration.

    Return the options worth reporting, keyed by name; values can be primitives, mappings
    or sequences of them. Leave secrets and endpoints out: the report is uploaded. Objects
    without this method are reported by class name alone."""

    def describe_options(self) -> Mapping[str, object]: ...


def _describe_option_object(obj: object, *, depth: int) -> str:
    """Ask a host SDK object for its reportable options on admitted preparation work."""
    cls = type(obj)
    name = f"{cls.__module__}.{cls.__name__}"
    if not isinstance(obj, DescribesOptions):
        return name
    options = _serialize_option_value(obj.describe_options(), depth=depth + 1)
    if not isinstance(options, Mapping):
        raise TypeError("describe_options() must return a mapping")
    parts = [
        f"{key}={value if value is None or isinstance(value, _OPTION_PRIMITIVES) else json.dumps(value, sort_keys=True)}"
        for key, value in options.items()
        if value is not None
    ]
    return f"{name}({', '.join(parts)})"


def _serialize_option_value(value: object, *, depth: int = 0) -> JsonValue:
    if depth > _core.TELEMETRY_MAX_DEPTH:
        raise ValueError("session report options exceed the native input budget")
    if value is None or isinstance(value, _OPTION_PRIMITIVES):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("session report option keys must be strings")
            result[key] = _serialize_option_value(item, depth=depth + 1)
        return result
    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes)):
        items = [_serialize_option_value(item, depth=depth + 1) for item in value]
        return sorted(items, key=str) if isinstance(value, Set) else items
    return _describe_option_object(value, depth=depth)


MAX_PAYLOAD_ATTR_LEN = 1024


"""Request and response payloads longer than this many characters are truncated in span
attributes."""


_RpcInterceptorBase: type = getattr(rtc, "RpcInterceptor", object)


_warned_unsupported = False


def _truncate(payload: str) -> str:
    return payload[:MAX_PAYLOAD_ATTR_LEN]


def _payload_attributes(payload: str) -> dict[str, AttributeValue]:
    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_RPC_PAYLOAD_SIZE: len(payload.encode("utf-8"))
    }
    if payload:
        attrs[trace_types.ATTR_RPC_PAYLOAD] = _truncate(payload)
    return attrs


def _response_attributes(response: str | None) -> dict[str, AttributeValue]:
    response = response or ""
    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_RPC_RESPONSE_SIZE: len(response.encode("utf-8"))
    }
    if response:
        attrs[trace_types.ATTR_RPC_RESPONSE] = _truncate(response)
    return attrs


_CANCEL_DESCRIPTIONS = {
    "RESPONSE_TIMEOUT": "response timeout",
    "RECIPIENT_DISCONNECTED": "caller disconnected",
    "APPLICATION_ERROR": "handler cancelled",
}


def _cancellation_outcome(
    invocation: rtc.RpcInvocationData,
) -> tuple[rtc.RpcError.ErrorCode | None, str]:
    """The RPC error code the caller receives for a cancelled handler chain, and a status text.

    ``cancel_reason`` is set by the SDK before it cancels the chain; ``None`` there means the
    cancellation came from inside the chain, which the SDK answers as ``APPLICATION_ERROR``.
    An SDK without the field gives no code."""
    if not hasattr(invocation, "cancel_reason"):
        return None, "cancelled"
    code = invocation.cancel_reason
    if code is None:
        code = rtc.RpcError.ErrorCode.APPLICATION_ERROR
    return code, _CANCEL_DESCRIPTIONS.get(getattr(code, "name", ""), "cancelled")


class TracingRpcInterceptor(_RpcInterceptorBase):  # type: ignore[misc]
    """An ``rtc.RpcInterceptor`` emitting ``rpc_call`` / ``rpc_handler`` spans."""

    async def intercept_outgoing(
        self, call: rtc.RpcCallInfo, next: Callable[[rtc.RpcCallInfo], Awaitable[str]]
    ) -> str:
        attributes: dict[str, AttributeValue] = {
            trace_types.ATTR_RPC_METHOD: call.method,
            trace_types.ATTR_RPC_DESTINATION_IDENTITY: call.destination_identity,
            **_payload_attributes(call.payload),
        }
        if call.response_timeout is not None:
            attributes[trace_types.ATTR_RPC_RESPONSE_TIMEOUT] = call.response_timeout

        with tracer.start_as_current_span(
            "rpc_call", kind=trace.SpanKind.CLIENT, attributes=attributes
        ) as span:
            try:
                response = await next(call)
            except rtc.RpcError as e:
                span.set_attribute(trace_types.ATTR_RPC_ERROR_CODE, int(e.code))
                raise
            span.set_attributes(_response_attributes(response))
            return response

    async def intercept_incoming(
        self,
        invocation: rtc.RpcInvocationData,
        next: Callable[[rtc.RpcInvocationData], Awaitable[str | None]],
    ) -> str | None:
        attributes: dict[str, AttributeValue] = {
            trace_types.ATTR_RPC_METHOD: invocation.method,
            trace_types.ATTR_RPC_REQUEST_ID: invocation.request_id,
            trace_types.ATTR_RPC_CALLER_IDENTITY: invocation.caller_identity,
            trace_types.ATTR_RPC_RESPONSE_TIMEOUT: invocation.response_timeout,
            trace_types.ATTR_RPC_HANDLER_REGISTERED: True,
            **_payload_attributes(invocation.payload),
        }
        with tracer.start_as_current_span(
            "rpc_handler",
            context=session_root_context(),
            kind=trace.SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            try:
                response = await next(invocation)
            except rtc.RpcError as e:
                span.set_attribute(trace_types.ATTR_RPC_ERROR_CODE, int(e.code))
                if e.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD:
                    # a client called a method this agent never registered
                    span.set_attribute(trace_types.ATTR_RPC_HANDLER_REGISTERED, False)
                raise
            except asyncio.CancelledError:
                # the SDK maps a cancellation to an RpcError only after this interceptor has
                # unwound, and a CancelledError is not an Exception, so the span would end
                # UNSET; invocation.cancel_reason says what the caller gets (livekit>=1.1.19)
                code, description = _cancellation_outcome(invocation)
                if code is not None:
                    span.set_attribute(trace_types.ATTR_RPC_ERROR_CODE, int(code))
                span.set_status(trace.Status(trace.StatusCode.ERROR, description))
                raise
            span.set_attributes(_response_attributes(response))
            return response


_interceptor = TracingRpcInterceptor()


def install_rpc_tracing(local_participant: rtc.LocalParticipant) -> bool:
    """Trace RPCs on ``local_participant``. Idempotent. Returns False when the installed
    ``livekit-rtc`` has no interceptor support."""
    global _warned_unsupported
    add = getattr(local_participant, "add_rpc_interceptor", None)
    if add is None:
        if not _warned_unsupported:
            _warned_unsupported = True
            logger.debug(
                "livekit-rtc has no RpcInterceptor support; RPC calls will not be traced "
                "(requires livekit>=1.1.18)"
            )
        return False
    add(_interceptor)
    return True
