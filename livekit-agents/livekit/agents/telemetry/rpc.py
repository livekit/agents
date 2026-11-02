"""Trace RPCs the agent performs and handles.

The room SDK exposes an ``RpcInterceptor`` hook (``livekit-rtc`` >= 1.1.18) that wraps every
call made through ``LocalParticipant.perform_rpc`` and every invocation dispatched to a
registered handler. This module installs one interceptor per local participant that turns
each call into a span following the OpenTelemetry RPC semantic conventions:

* ``rpc_call`` (``SpanKind.CLIENT``) for outgoing calls, under whatever span is current where
  the call is made (an RPC issued from a tool nests under ``function_tool``);
* ``rpc_handler`` (``SpanKind.SERVER``) for incoming invocations, under the primary agent
  session's root span.

Payloads are recorded truncated under ``lk.pii`` keys. Participant identities are application
identifiers, not end-user data, and are recorded as is. On an SDK without the hook,
``install`` is a no-op.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from opentelemetry import trace

from livekit import rtc

from ..log import logger
from . import session_context, trace_types
from .traces import tracer

MAX_PAYLOAD_ATTR_LEN = 1024
"""Request and response payloads longer than this many characters are truncated in span
attributes."""

_RpcInterceptorBase: type = getattr(rtc, "RpcInterceptor", object)
_warned_unsupported = False


def _truncate(payload: str) -> str:
    return payload[:MAX_PAYLOAD_ATTR_LEN]


def _payload_attributes(payload: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {trace_types.ATTR_RPC_PAYLOAD_SIZE: len(payload.encode("utf-8"))}
    if payload:
        attrs[trace_types.ATTR_RPC_PAYLOAD] = _truncate(payload)
    return attrs


def _response_attributes(response: str | None) -> dict[str, Any]:
    response = response or ""
    attrs: dict[str, Any] = {trace_types.ATTR_RPC_RESPONSE_SIZE: len(response.encode("utf-8"))}
    if response:
        attrs[trace_types.ATTR_RPC_RESPONSE] = _truncate(response)
    return attrs


_CANCEL_DESCRIPTIONS = {
    "RESPONSE_TIMEOUT": "response timeout",
    "RECIPIENT_DISCONNECTED": "caller disconnected",
    "APPLICATION_ERROR": "handler cancelled",
}


def _cancellation_outcome(invocation: Any) -> tuple[Any, str]:
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

    async def intercept_outgoing(self, call: Any, next: Callable[[Any], Awaitable[str]]) -> str:
        attributes: dict[str, Any] = {
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
        self, invocation: Any, next: Callable[[Any], Awaitable[str | None]]
    ) -> str | None:
        attributes: dict[str, Any] = {
            trace_types.ATTR_RPC_METHOD: getattr(invocation, "method", ""),
            trace_types.ATTR_RPC_REQUEST_ID: invocation.request_id,
            trace_types.ATTR_RPC_CALLER_IDENTITY: invocation.caller_identity,
            trace_types.ATTR_RPC_RESPONSE_TIMEOUT: invocation.response_timeout,
            trace_types.ATTR_RPC_HANDLER_REGISTERED: True,
            **_payload_attributes(invocation.payload),
        }
        with tracer.start_as_current_span(
            "rpc_handler",
            context=session_context.session_root_context(),
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


def install(local_participant: Any) -> bool:
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
