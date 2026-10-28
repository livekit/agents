"""Session report SDK projection and RTC RPC instrumentation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import aiohttp
from google.protobuf.json_format import MessageToDict
from opentelemetry import context as otel_context, trace
from opentelemetry._logs import get_logger_provider
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.util.types import AnyValue, AttributeValue

from livekit import rtc

from .. import _core
from .._proto import encode_chat_item
from ..log import logger
from ..types import ATTRIBUTE_SIMULATION_ENABLED
from . import trace_types, utils as telemetry_utils

if TYPE_CHECKING:
    from .._core import Tagger
    from .._generated.telemetry import ProjectedReportEvent
    from ..job import SessionReport
    from ..llm import ChatItem

from .._generated.telemetry import LogRecordWire
from . import _destination as destination
from ._context import (
    capture_content_enabled as capture_content_enabled,
    primary_session as primary_session,
    session_root_context as session_root_context,
    set_capture_content as set_capture_content,
    set_error_type as set_error_type,
    tracer as tracer,
)
from ._providers import (
    _cloud,
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
    is_pii_attribute as is_pii_attribute,
    redact as redact,
)

_SESSION_OPTION_KEY_ALIASES = {
    "keyterms": "lk.pii.keyterms",
}


# Option keys never written to the report: prompt text authored by the customer
# (``stt_context_options.keyterm_detection.instructions``) can embed anything about their
# business or users, and the report has no use for it.
_SESSION_OPTION_OMITTED_KEYS = frozenset({"instructions"})


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

    def describe_options(self) -> Mapping[str, Any]: ...


def _describe_option_object(obj: object, *, depth: int) -> str:
    """Ask a host SDK object for its reportable options on admitted preparation work."""
    cls = type(obj)
    name = f"{cls.__module__}.{cls.__name__}"
    describe = getattr(obj, "describe_options", None)
    if not callable(describe):
        return name
    options, _ = _serialize_option_value(describe(), depth=depth + 1)
    if not isinstance(options, Mapping):
        raise TypeError("describe_options() must return a mapping")
    parts = [
        f"{key}={value if value is None or isinstance(value, _OPTION_PRIMITIVES) else json.dumps(value, sort_keys=True)}"
        for key, value in options.items()
        if value is not None
    ]
    return f"{name}({', '.join(parts)})"


def _serialize_option_value(value: Any, *, depth: int = 0) -> tuple[Any, int]:
    from ._export import attribute_bytes, policy

    native = policy()

    def checked(charge: int) -> int:
        if charge > native.max_record_bytes:
            raise ValueError("session report options exceed the native input budget")
        return charge

    if value is None or isinstance(value, _OPTION_PRIMITIVES):
        return value, checked(attribute_bytes(value, depth=depth))
    if isinstance(value, Mapping):
        charge = checked(native.value_bytes("mapping", len(value), depth, []))
        children = 0
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("session report option keys must be strings")
            if key not in _SESSION_OPTION_OMITTED_KEYS:
                name = _SESSION_OPTION_KEY_ALIASES.get(key, key)
                key_charge = checked(attribute_bytes(name, depth=depth + 1))
                serialized, item_charge = _serialize_option_value(item, depth=depth + 1)
                children += key_charge + item_charge
                charge = checked(native.value_bytes("mapping", len(value), depth, [children]))
                result[name] = serialized
        return result, charge
    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes)):
        charge = checked(native.value_bytes("sequence", len(value), depth, []))
        children = 0
        items: list[Any] = []
        for item in value:
            serialized, item_charge = _serialize_option_value(item, depth=depth + 1)
            children += item_charge
            charge = checked(native.value_bytes("sequence", len(value), depth, [children]))
            items.append(serialized)
        return (sorted(items, key=str) if isinstance(value, Set) else items), charge
    serialized = _describe_option_object(value, depth=depth)
    return serialized, checked(attribute_bytes(serialized, depth=depth))


def _serialize_session_options(options: dict[str, Any]) -> dict[str, Any]:
    serialized, _ = _serialize_option_value(options)
    assert isinstance(serialized, dict)
    return serialized


def _chat_item_span_attribute(item: ChatItem) -> dict[str, Any]:
    return MessageToDict(encode_chat_item(item), preserving_proto_field_name=True)


def _append_transcript_item(
    builder: _core._core_PreparedTranscript, item: ChatItem, item_count: int
) -> _core.ChatItem:
    from .._core_view import to_core_item
    from ..llm.chat_context import AudioContent, ImageContent
    from ._export import attribute_bytes, policy

    include_image, include_audio = builder.content_policy()
    fields = item.__dict__
    content = None
    if item.type == "message":
        builder.check_input(item_count, policy().value_bytes("sequence", len(item.content), 0, []))
        content = [
            part
            for part in item.content
            if isinstance(part, str)
            or (include_image and isinstance(part, ImageContent))
            or (include_audio and isinstance(part, AudioContent))
        ]
        fields = {**fields, "content": content}
    builder.check_input(item_count, attribute_bytes(fields))
    native_item = to_core_item(item, content=content)
    builder.append(native_item)
    return native_item


async def _upload_session_report(
    *,
    agent_name: str,
    observability_url: str,
    report: SessionReport,
    tagger: Tagger,
    http_session: aiohttp.ClientSession,
    metadata: dict[str, AttributeValue] | None = None,
) -> None:
    """Project host objects on admitted SDK work; the native runtime uploads the recording."""
    from .._core_view import from_core_item
    from ._export import attribute_bytes, call_bounded, policy

    if attribute_bytes(metadata) > policy().max_record_bytes:
        raise ValueError("session report metadata exceeds the native input budget")
    if attribute_bytes(agent_name) > policy().max_record_bytes:
        raise ValueError("session report summary exceeds the native input budget")
    metadata = dict(metadata or {})
    emission_context = otel_context.get_current()
    credentials = (os.environ.get("LIVEKIT_API_KEY", ""), os.environ.get("LIVEKIT_API_SECRET", ""))
    credential_bytes = attribute_bytes(credentials)
    if credential_bytes > policy().max_record_bytes:
        raise ValueError("session report credentials exceed the native input budget")
    uploader = await call_bounded(
        f"session-report-client/{id(report):x}",
        partial(_core._core_SessionReportUpload, *credentials),
        credential_bytes,
        policy().upload_timeout_seconds,
    )
    options = report._recording_options
    capture_content = report._capture_content
    redacted = options.redaction or telemetry_utils.redaction_enabled(metadata)
    projection = _core._core_SessionReportProjection(capture_content, options, redacted)
    transcript_allowed, audio_allowed = projection.transcript, projection.audio
    prepared: list[str] = []

    builder: _core._core_PreparedTranscript | None = None
    batch_size = 1
    provider = _cloud.logger_provider or get_logger_provider()
    loggers: dict[str, Any] = {}
    recording_options = _core.RecordingOptions(
        audio=audio_allowed,
        traces=options.traces,
        logs=options.logs,
        transcript=transcript_allowed,
        redaction=redacted,
    )

    def emit(record: LogRecordWire) -> None:
        if (otel_logger := loggers.get(record.logger)) is None:
            otel_logger = loggers[record.logger] = provider.get_logger(
                name=record.logger,
                attributes={
                    "room_id": snapshot.room_id,
                    "job_id": snapshot.job_id,
                    "room": snapshot.room,
                    **metadata,
                },
            )
        otel_logger.emit(
            body=record.body,
            timestamp=record.timestamp_nanos,
            context=emission_context,
            attributes={**metadata, **cast(Mapping[str, AnyValue], record.attributes)},
            severity_number=SeverityNumber(record.severity_number),
            severity_text=record.severity_text,
        )

    def start() -> None:
        nonlocal builder, batch_size
        if transcript_allowed:
            builder = _core._core_PreparedTranscript()
            builder.check_input(len(report.chat_history.items), 0)
            batch_size = builder.input_batch_size
        records = _core.session_records(
            destination.upload_gate,
            {
                "agent_name": agent_name,
                "job_id": snapshot.job_id,
                "room_id": snapshot.room_id,
                "room": snapshot.room,
                "sdk_version": snapshot.sdk_version,
                "started_at": report.started_at,
                "timestamp": snapshot.timestamp,
                "options": snapshot.options,
                "omitted_events": snapshot.omitted_events,
                "recording_options": recording_options,
                "model_usage": snapshot.usage or [],
                "chat_history": [],
            },
            tagger,
            lambda item: _chat_item_span_attribute(from_core_item(item)),
            capture_content=capture_content,
        )
        for record in records:
            emit(record)

    def emit_event(event: ProjectedReportEvent) -> None:
        record = projection.event_record(
            event,
            (snapshot.job_id, snapshot.room_id, snapshot.room),
            snapshot.timestamp,
            destination.upload_gate.disabled,
        )
        if record is None:
            worker.record_drop("dropped", 1)
        else:
            emit(record)

    def append_batch(items: Sequence[ChatItem]) -> None:
        assert builder is not None
        for item in items:
            native_item = _append_transcript_item(builder, item, len(report.chat_history.items))
            record = _core._core_session_chat_record(
                destination.upload_gate,
                recording_options,
                native_item,
                _chat_item_span_attribute(item),
                capture_content,
                (snapshot.job_id, snapshot.room_id, snapshot.room),
            )
            if record is not None:
                emit(record)

    def finish() -> None:
        nonlocal builder
        if builder is not None:
            prepared.append(builder.finish())
            builder = None

    worker = _core._core_TelemetryWorker()
    key = f"session-report/{report.job_id}"
    if not worker.reserve(key, policy().report_retained_bytes):
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + policy().upload_timeout_seconds

    async def run_preparation(callback: Callable[[], None]) -> None:
        await call_bounded(
            key, callback, policy().max_record_bytes, max(0.0, deadline - loop.time())
        )

    try:
        snapshot, events = await report._snapshot(projection, deadline=deadline)
        await run_preparation(start)
        for event in events:
            await run_preparation(partial(emit_event, event))
            await asyncio.sleep(0)
        if transcript_allowed:
            for index in range(0, len(report.chat_history.items), batch_size):
                items = report.chat_history.items[index : index + batch_size]
                await run_preparation(partial(append_batch, items))
                await asyncio.sleep(0)
        await run_preparation(finish)
        transcript = prepared.pop() if transcript_allowed else None
    finally:
        prepared.clear()
        worker.release(key)
    enabled = await uploader.upload(
        observability_url,
        report.room_id,
        report.job_id,
        bool(metadata.get(ATTRIBUTE_SIMULATION_ENABLED, False)),
        redacted,
        int((report.audio_recording_started_at or 0) * 1000),
        transcript,
        str(report.audio_recording_path) if audio_allowed and report.audio_recording_path else None,
        destination.upload_gate.disabled,
    )
    if not enabled:
        destination.upload_gate.disable()


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


def install_rpc_tracing(local_participant: Any) -> bool:
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
