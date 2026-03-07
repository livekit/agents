from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import aiofiles
import aiohttp
import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry import context as otel_context, trace, trace as trace_api
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
    LogRecordProcessor,
    ReadWriteLogRecord,
)
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import Attributes, AttributeValue

from livekit import api
from livekit.protocol import metrics as proto_metrics

from ..log import logger
from . import trace_types

if TYPE_CHECKING:
    from ..llm import ChatContext
    from ..observability import Tagger
    from ..voice.report import SessionReport


class _DynamicTracer(Tracer):
    def __init__(self, instrumenting_module_name: str) -> None:
        self._instrumenting_module_name = instrumenting_module_name
        self._tracer_provider: trace_api.TracerProvider = trace.get_tracer_provider()
        self._tracer = trace.get_tracer(instrumenting_module_name)

    def set_provider(self, tracer_provider: trace_api.TracerProvider) -> None:
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

    def on_emit(self, log_data: ReadWriteLogRecord) -> None:
        if log_data.log_record.attributes:
            log_data.log_record.attributes.update(self._metadata)  # type: ignore
        else:
            log_data.log_record.attributes = self._metadata

        if log_data.instrumentation_scope:
            log_data.log_record.attributes.update(  # type: ignore
                {"logger.name": log_data.instrumentation_scope.name}
            )

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _BufferingHandler(logging.Handler):
    """Buffers log records in memory for later replay through OTLP."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(record)


def set_tracer_provider(
    tracer_provider: trace_api.TracerProvider, *, metadata: dict[str, AttributeValue] | None = None
) -> None:
    """Set the tracer provider for the livekit-agents.

    Args:
        tracer_provider (TracerProvider): The tracer provider to set.
        metadata (dict[str, AttributeValue] | None, optional): Metadata to set on all spans. Defaults to None.
    """
    if metadata and isinstance(tracer_provider, trace_sdk.TracerProvider):
        tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))

    tracer.set_provider(tracer_provider)


def _setup_cloud_tracer(
    *,
    room_id: str,
    job_id: str,
    cloud_hostname: str,
    enable_traces: bool = True,
    enable_logs: bool = True,
) -> None:
    token_ttl = timedelta(hours=6)
    refresh_margin = timedelta(minutes=5)

    class _AuthRefreshingSession(requests.Session):
        def __init__(self, header_provider: _AuthHeaderProvider) -> None:
            super().__init__()
            self._header_provider = header_provider

        def request(self, *args: Any, **kwargs: Any) -> requests.Response:
            self.headers.update(self._header_provider())
            return super().request(*args, **kwargs)

    class _AuthHeaderProvider:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._auth_header = ""
            self._expires_at = datetime.min.replace(tzinfo=timezone.utc)
            self._refresh()

        def _refresh(self) -> None:
            access_token = (
                api.AccessToken()
                .with_observability_grants(api.ObservabilityGrants(write=True))
                .with_ttl(token_ttl)
            )
            self._auth_header = f"Bearer {access_token.to_jwt()}"
            self._expires_at = datetime.now(timezone.utc) + token_ttl

        def __call__(self) -> dict[str, str]:
            now = datetime.now(timezone.utc)
            if now >= self._expires_at - refresh_margin:
                with self._lock:
                    if now >= self._expires_at - refresh_margin:
                        self._refresh()
            return {"Authorization": self._auth_header}

    header_provider = _AuthHeaderProvider()
    session = _AuthRefreshingSession(header_provider)
    otlp_compression = Compression.Gzip
    metadata: dict[str, AttributeValue] = {"room_id": room_id, "job_id": job_id}

    resource = Resource.create(
        {
            SERVICE_NAME: "livekit-agents",
            "room_id": room_id,
            "job_id": job_id,
        }
    )

    if enable_traces:
        # Check if a tracer provider is not set and set one up
        # below shows how the ProxyTracerProvider is returned when none have been setup
        # https://github.com/open-telemetry/opentelemetry-python/blob/0018c0030bac9bdce4487fe5fcb3ec6a542ec904/opentelemetry-api/src/opentelemetry/trace/__init__.py#L555
        tracer_provider: trace_api.TracerProvider
        if isinstance(
            tracer._tracer_provider,
            (trace_api.ProxyTracerProvider, trace_api.NoOpTracerProvider),
        ):
            tracer_provider = trace_sdk.TracerProvider(resource=resource)
            set_tracer_provider(tracer_provider)
        else:
            # attach the processor to the existing tracer provider
            tracer_provider = tracer._tracer_provider
            if isinstance(tracer_provider, trace_sdk.TracerProvider):
                tracer_provider.resource.merge(resource)

        span_exporter = OTLPSpanExporter(
            endpoint=f"https://{cloud_hostname}/observability/traces/otlp/v0",
            compression=otlp_compression,
            session=session,
        )

        if isinstance(tracer_provider, trace_sdk.TracerProvider):
            tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    # Set up the logger provider — it's the entrypoint for session reports,
    # evaluations, and chat history, not just Python log export.
    logger_provider = get_logger_provider()
    if not isinstance(logger_provider, LoggerProvider):
        logger_provider = LoggerProvider()
        set_logger_provider(logger_provider)

    if enable_logs:
        log_exporter = OTLPLogExporter(
            endpoint=f"https://{cloud_hostname}/observability/logs/otlp/v0",
            compression=otlp_compression,
            session=session,
        )
        logger_provider.add_log_record_processor(_MetadataLogProcessor(metadata))
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

        handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

        root = logging.getLogger()
        root.addHandler(handler)


def _chat_ctx_to_otel_events(chat_ctx: ChatContext) -> list[tuple[str, Attributes]]:
    role_to_event = {
        "system": trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE,
        "user": trace_types.EVENT_GEN_AI_USER_MESSAGE,
        "assistant": trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
    }

    events: list[tuple[str, Attributes]] = []
    for item in chat_ctx.items:
        if item.type == "message" and (event_name := role_to_event.get(item.role)):
            # only support text content for now
            events.append((event_name, {"content": item.text_content or ""}))
        elif item.type == "function_call":
            events.append(
                (
                    trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
                    {
                        "role": "assistant",
                        "tool_calls": [
                            json.dumps(
                                {
                                    "function": {"name": item.name, "arguments": item.arguments},
                                    "id": item.call_id,
                                    "type": "function",
                                }
                            )
                        ],
                    },
                )
            )
        elif item.type == "function_call_output":
            events.append(
                (
                    trace_types.EVENT_GEN_AI_TOOL_MESSAGE,
                    {"content": item.output, "name": item.name, "id": item.call_id},
                )
            )
    return events


async def _upload_session_report(
    *,
    agent_name: str,
    cloud_hostname: str,
    report: SessionReport,
    tagger: Tagger,
    http_session: aiohttp.ClientSession,
) -> None:
    from ..llm.chat_context import chat_item_to_proto

    def _get_logger(name: str) -> Any:
        return get_logger_provider().get_logger(
            name=name,
            attributes={
                "room_id": report.room_id,
                "job_id": report.job_id,
                "room": report.room,
            },
        )

    def _log(
        otel_logger: Any,
        body: str,
        timestamp: int,
        attributes: dict,
        severity: SeverityNumber = SeverityNumber.UNSPECIFIED,
        severity_text: str = "unspecified",
    ) -> None:
        otel_logger.emit(
            body=body,
            timestamp=timestamp,
            attributes=attributes,
            severity_number=severity,
            severity_text=severity_text,
        )

    chat_logger = _get_logger("chat_history")
    recording_options = report.recording_options

    if any(recording_options.values()):
        _log(
            chat_logger,
            body="session report",
            timestamp=int((report.started_at or report.timestamp or 0) * 1e9),
            attributes={
                "session.options": vars(report.options),
                "session.report_timestamp": report.timestamp,
                "agent_name": agent_name,
            },
        )

    if recording_options["transcript"]:
        for item in report.chat_history.items:
            item_log = MessageToDict(chat_item_to_proto(item))
            severity: SeverityNumber = SeverityNumber.UNSPECIFIED
            severity_text: str = "unspecified"

            if item.type == "function_call_output" and item.is_error:
                severity = SeverityNumber.ERROR
                severity_text = "error"

            _log(
                chat_logger,
                body="chat item",
                timestamp=int(item.created_at * 1e9),
                attributes={"chat.item": item_log},
                severity=severity,
                severity_text=severity_text,
            )

    eval_logger = _get_logger("evaluations")
    if tagger.evaluations:
        for evaluation in tagger.evaluations:
            severity = SeverityNumber.UNSPECIFIED
            severity_text = "unspecified"

            if evaluation.get("verdict") == "fail":
                severity = SeverityNumber.ERROR
                severity_text = "error"

            _log(
                eval_logger,
                body="evaluation",
                timestamp=int(report.timestamp * 1e9),
                attributes={"evaluation": evaluation},
                severity=severity,
                severity_text=severity_text,
            )

    if tagger.outcome_reason:
        _log(
            eval_logger,
            body="outcome",
            timestamp=int(report.timestamp * 1e9),
            attributes={"outcome": {"reason": tagger.outcome_reason}},
        )

    has_audio = (
        recording_options["audio"]
        and report.audio_recording_path
        and report.audio_recording_started_at
    )
    if not recording_options["transcript"] and not has_audio:
        return

    # emit recording
    access_token = (
        api.AccessToken()
        .with_observability_grants(api.ObservabilityGrants(write=True))
        .with_ttl(timedelta(hours=6))
    )
    jwt = access_token.to_jwt()

    header_msg = proto_metrics.MetricsRecordingHeader(
        room_id=report.room_id,
    )
    header_msg.start_time.FromMilliseconds(int((report.audio_recording_started_at or 0) * 1000))
    header_bytes = header_msg.SerializeToString()

    mp = aiohttp.MultipartWriter("form-data")

    part = mp.append(header_bytes)
    part.set_content_disposition("form-data", name="header", filename="header.binpb")
    part.headers["Content-Type"] = "application/protobuf"
    part.headers["Content-Length"] = str(len(header_bytes))

    if recording_options["transcript"]:
        chat_history_json = json.dumps(report.chat_history.to_dict(exclude_timestamp=False))
        part = mp.append(chat_history_json)
        part.set_content_disposition("form-data", name="chat_history", filename="chat_history.json")
        part.headers["Content-Type"] = "application/json"
        part.headers["Content-Length"] = str(len(chat_history_json))

    if has_audio and report.audio_recording_path:
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
    if isinstance(tracer_provider := tracer._tracer_provider, trace_sdk.TracerProvider):
        logger.debug("shutting down telemetry tracer provider")
        tracer_provider.force_flush()
        tracer_provider.shutdown()

    if isinstance(logger_provider := get_logger_provider(), LoggerProvider):
        # remove the OTLP LoggingHandler before flushing to avoid deadlock —
        # force_flush triggers log export which emits new logs back through the handler
        root = logging.getLogger()
        for h in root.handlers[:]:
            if isinstance(h, LoggingHandler):
                root.removeHandler(h)

        logger_provider.force_flush()
        logger_provider.shutdown()  # type: ignore
