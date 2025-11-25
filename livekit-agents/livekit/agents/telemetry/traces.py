from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import aiofiles
import aiohttp
from google.protobuf.json_format import MessageToDict
from opentelemetry import context as otel_context, trace
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import (
    LogData,
    LoggerProvider,
    LoggingHandler,
    LogRecord,
    LogRecordProcessor,
)
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, TraceFlags, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AttributeValue

from livekit import api
from livekit.protocol import agent_pb, metrics as proto_metrics

from ..log import logger

if TYPE_CHECKING:
    from ..llm import ChatItem
    from ..voice.report import SessionReport


class _DynamicTracer(Tracer):
    def __init__(self, instrumenting_module_name: str) -> None:
        self._instrumenting_module_name = instrumenting_module_name
        self._tracer_provider = trace.get_tracer_provider()
        self._tracer = trace.get_tracer(instrumenting_module_name)

    def set_provider(self, tracer_provider: TracerProvider) -> None:
        self._tracer_provider = tracer_provider
        self._tracer = trace.get_tracer(
            self._instrumenting_module_name,
            tracer_provider=self._tracer_provider,
        )

    def start_span(self, *args: Any, **kwargs: Any) -> Span:
        return self._tracer.start_span(*args, **kwargs)

    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        with self._tracer.start_as_current_span(*args, **kwargs) as span:
            yield span


tracer: _DynamicTracer = _DynamicTracer("livekit-agents")


class _MetadataSpanProcessor(SpanProcessor):
    def __init__(self, metadata: dict[str, AttributeValue]) -> None:
        self._metadata = metadata

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        span.set_attributes(self._metadata)


class _MetadataLogProcessor(LogRecordProcessor):
    def __init__(self, metadata: dict[str, AttributeValue]) -> None:
        self._metadata = metadata

    def emit(self, log_data: LogData) -> None:
        if log_data.log_record.attributes:
            log_data.log_record.attributes.update(self._metadata)  # type: ignore
        else:
            log_data.log_record.attributes = self._metadata

        log_data.log_record.attributes.update(  # type: ignore
            {"logger.name": log_data.instrumentation_scope.name}
        )

    def on_emit(self, log_data: LogData) -> None:
        if log_data.log_record.attributes:
            log_data.log_record.attributes.update(self._metadata)  # type: ignore
        else:
            log_data.log_record.attributes = self._metadata

        log_data.log_record.attributes.update(  # type: ignore
            {"logger.name": log_data.instrumentation_scope.name}
        )

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def set_tracer_provider(
    tracer_provider: TracerProvider, *, metadata: dict[str, AttributeValue] | None = None
) -> None:
    """Set the tracer provider for the livekit-agents.

    Args:
        tracer_provider (TracerProvider): The tracer provider to set.
        metadata (dict[str, AttributeValue] | None, optional): Metadata to set on all spans. Defaults to None.
    """
    if metadata:
        tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))

    tracer.set_provider(tracer_provider)


def _setup_cloud_tracer(*, room_id: str, job_id: str, cloud_hostname: str) -> None:
    access_token = (
        api.AccessToken()
        .with_observability_grants(api.ObservabilityGrants(write=True))
        .with_ttl(timedelta(hours=6))
    )

    otlp_compression = Compression.Gzip
    headers = {
        "Authorization": f"Bearer {access_token.to_jwt()}",
    }
    metadata: dict[str, AttributeValue] = {"room_id": room_id, "job_id": job_id}

    resource = Resource.create(
        {
            SERVICE_NAME: "livekit-agents",
            "room_id": room_id,
            "job_id": job_id,
        }
    )

    if not isinstance(tracer._tracer_provider, TracerProvider):
        tracer_provider = TracerProvider(resource=resource)
        set_tracer_provider(tracer_provider)
    else:
        # attach the processor to the existing tracer provider
        tracer_provider = tracer._tracer_provider
        tracer_provider.resource.merge(resource)

    span_exporter = OTLPSpanExporter(
        endpoint=f"https://{cloud_hostname}/observability/traces/otlp/v0",
        headers=headers,
        compression=otlp_compression,
    )

    tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    logger_provider = get_logger_provider()
    if not isinstance(logger_provider, LoggerProvider):
        logger_provider = LoggerProvider()
        set_logger_provider(logger_provider)

    log_exporter = OTLPLogExporter(
        endpoint=f"https://{cloud_hostname}/observability/logs/otlp/v0",
        headers=headers,
        compression=otlp_compression,
    )
    logger_provider.add_log_record_processor(_MetadataLogProcessor(metadata))
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    root = logging.getLogger()
    root.addHandler(handler)


def _to_proto_chat_item(item: ChatItem) -> dict:  # agent_pb.agent_session.ChatContext.ChatItem:
    item_pb = agent_pb.agent_session.ChatContext.ChatItem()

    if item.type == "message":
        msg = item_pb.message
        msg.id = item.id

        role_map = {
            "developer": agent_pb.agent_session.DEVELOPER,
            "system": agent_pb.agent_session.SYSTEM,
            "user": agent_pb.agent_session.USER,
            "assistant": agent_pb.agent_session.ASSISTANT,
        }
        msg.role = role_map[item.role]

        for content in item.content:
            if isinstance(content, str):
                content_pb = msg.content.add()
                content_pb.text = content

        msg.interrupted = item.interrupted

        if item.transcript_confidence is not None:
            msg.transcript_confidence = item.transcript_confidence

        for key, value in item.extra.items():
            msg.extra[key] = str(value)

        metrics = item.metrics
        if "started_speaking_at" in metrics:
            msg.metrics.started_speaking_at.FromMilliseconds(
                int(metrics["started_speaking_at"] * 1000)
            )
        if "stopped_speaking_at" in metrics:
            msg.metrics.stopped_speaking_at.FromMilliseconds(
                int(metrics["stopped_speaking_at"] * 1000)
            )
        if "transcription_delay" in metrics:
            msg.metrics.transcription_delay = metrics["transcription_delay"]
        if "end_of_turn_delay" in metrics:
            msg.metrics.end_of_turn_delay = metrics["end_of_turn_delay"]
        if "on_user_turn_completed_delay" in metrics:
            msg.metrics.on_user_turn_completed_delay = metrics["on_user_turn_completed_delay"]
        if "llm_node_ttft" in metrics:
            msg.metrics.llm_node_ttft = metrics["llm_node_ttft"]
        if "tts_node_ttfb" in metrics:
            msg.metrics.tts_node_ttfb = metrics["tts_node_ttfb"]
        if "e2e_latency" in metrics:
            msg.metrics.e2e_latency = metrics["e2e_latency"]
        msg.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "function_call":
        fc = item_pb.function_call
        fc.id = item.id
        fc.call_id = item.call_id
        fc.arguments = item.arguments
        fc.name = item.name
        fc.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "function_call_output":
        fco = item_pb.function_call_output
        fco.id = item.id
        fco.name = item.name
        fco.call_id = item.call_id
        fco.output = item.output
        fco.is_error = item.is_error
        fco.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "agent_handoff":
        ah = item_pb.agent_handoff
        ah.id = item.id
        if item.old_agent_id is not None:
            ah.old_agent_id = item.old_agent_id
        ah.new_agent_id = item.new_agent_id
        ah.created_at.FromMilliseconds(int(item.created_at * 1000))

    item_dict = MessageToDict(item_pb)

    # patch `arguments` & `output` to make them indexable attributes
    try:
        if item.type == "function_call":
            item_dict["arguments"] = json.loads(item_dict["arguments"])
        elif item.type == "function_call_output":
            item_dict["output"] = json.loads(item_dict["output"])
    except Exception:
        pass  # ignore

    return item_dict


def _to_rfc3339(value: int | float | datetime) -> str:
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    else:
        raise TypeError(f"Unsupported type for RFC3339 conversion: {type(value)!r}")

    dt = dt.replace(microsecond=(dt.microsecond // 1000) * 1000)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def _upload_session_report(
    *,
    agent_name: str,
    cloud_hostname: str,
    report: SessionReport,
    http_session: aiohttp.ClientSession,
) -> None:
    chat_logger = get_logger_provider().get_logger(
        name="chat_history",
        attributes={
            "room_id": report.room_id,
            "job_id": report.job_id,
            "room": report.room,
        },
    )

    def _log(
        body: str,
        timestamp: int,
        attributes: dict,
        severity: SeverityNumber = SeverityNumber.UNSPECIFIED,
        severity_text: str = "unspecified",
    ) -> None:
        chat_logger.emit(
            LogRecord(
                body=body,
                timestamp=timestamp,
                attributes=attributes,
                trace_id=0,
                span_id=0,
                severity_number=severity,
                severity_text=severity_text,
                trace_flags=TraceFlags.get_default(),
            )
        )

    _log(
        body="session report",
        timestamp=int((report.started_at or report.timestamp or 0) * 1e9),
        attributes={
            "session.options": vars(report.options),
            "session.report_timestamp": report.timestamp,
            "agent_name": agent_name,
        },
    )

    for item in report.chat_history.items:
        item_log = _to_proto_chat_item(item)
        severity: SeverityNumber = SeverityNumber.UNSPECIFIED
        severity_text: str = "unspecified"

        if item.type == "function_call_output" and item.is_error:
            severity = SeverityNumber.ERROR
            severity_text = "error"

        _log(
            body="chat item",
            timestamp=int(item.created_at * 1e9),
            attributes={"chat.item": item_log},
            severity=severity,
            severity_text=severity_text,
        )

    # emit recording
    access_token = (
        api.AccessToken()
        .with_observability_grants(api.ObservabilityGrants(write=True))
        .with_ttl(timedelta(hours=6))
    )
    jwt = access_token.to_jwt()

    header_msg = proto_metrics.MetricsRecordingHeader(
        room_id=report.room_id,
        duration=int((report.duration or 0) * 1000),
    )
    header_msg.start_time.FromMilliseconds(int((report.audio_recording_started_at or 0) * 1000))
    header_bytes = header_msg.SerializeToString()

    mp = aiohttp.MultipartWriter("form-data")

    part = mp.append(header_bytes)
    part.set_content_disposition("form-data", name="header", filename="header.binpb")
    part.headers["Content-Type"] = "application/protobuf"
    part.headers["Content-Length"] = str(len(header_bytes))

    chat_history_json = json.dumps(report.chat_history.to_dict(exclude_timestamp=False))
    part = mp.append(chat_history_json)
    part.set_content_disposition("form-data", name="chat_history", filename="chat_history.json")
    part.headers["Content-Type"] = "application/json"
    part.headers["Content-Length"] = str(len(chat_history_json))

    if report.audio_recording_path and report.audio_recording_started_at:
        try:
            async with aiofiles.open(report.audio_recording_path, "rb") as f:
                audio_bytes = await f.read()
        except Exception:
            audio_bytes = b""

        if audio_bytes:
            part = mp.append(audio_bytes)
            part.set_content_disposition("form-data", name="audio", filename="recording.ogg")
            part.headers["Content-Type"] = "audio/ogg"
            part.headers["Content-Length"] = str(len(audio_bytes))
            part.headers["Created-At"] = _to_rfc3339(report.audio_recording_started_at)

    url = f"https://{cloud_hostname}/observability/recordings/v0"
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": mp.content_type,
    }

    logger.debug("uploading session report to LiveKit Cloud")
    async with http_session.post(url, data=mp, headers=headers) as resp:
        resp.raise_for_status()

    logger.debug("finished uploading")


def _shutdown_telemetry() -> None:
    if isinstance(tracer_provider := tracer._tracer_provider, TracerProvider):
        logger.debug("shutting down telemetry tracer provider")
        tracer_provider.force_flush()
        tracer_provider.shutdown()

    if isinstance(logger_provider := get_logger_provider(), LoggerProvider):
        # force_flush will cause deadlock when new logs from OTLPLogExporter are emitted
        # logger_provider.force_flush()
        logger.debug("shutting down telemetry logger provider")
        logger_provider.shutdown()  # type: ignore
