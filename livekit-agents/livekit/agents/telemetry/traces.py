from __future__ import annotations

import aiofiles
import logging
import json
import aiohttp
from collections.abc import Iterator
import datetime
from livekit import api
from livekit.protocol import metrics as proto_metrics, agent_pb
from typing import TYPE_CHECKING, Any

from urllib.parse import urlparse


from opentelemetry import context as otel_context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.trace import Span, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import Attributes, AttributeValue


from opentelemetry import context as otel_context, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
    LogRecordProcessor,
    LogData,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http import Compression

from ..utils import misc
from ..log import logger

if TYPE_CHECKING:
    from ..voice.report import SessionReport
    from ..llm import ChatContext


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


tracer: Tracer = _DynamicTracer("livekit-agents")


class _MetadataSpanProcessor(SpanProcessor):
    def __init__(self, metadata: dict[str, str]) -> None:
        self._metadata = metadata

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        span.set_attributes(self._metadata)


class _MetadataLogProcessor(LogRecordProcessor):
    def __init__(self, metadata: dict[str, str]) -> None:
        self._metadata = metadata

    def on_emit(self, log_data: LogData) -> None:
        log_data.log_record.attributes.update(self._metadata)

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


def _setup_cloud_tracer(
    *,
    room_id: str,
    job_id: str,
    cloud_hostname: str,
) -> None:
    access_token = (
        api.AccessToken()
        .with_observability_grants(api.ObservabilityGrants(write=True))
        .with_ttl(datetime.timedelta(hours=6))
    )

    otlp_compression = Compression.Gzip
    headers = (("Authorization", f"Bearer {access_token.to_jwt()}"),)
    metadata = {"room_id": room_id, "job_id": job_id}

    resource = Resource.create(
        {
            SERVICE_NAME: "livekit-agents",
            "room_id": room_id,
            "job_id": job_id,
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    set_tracer_provider(tracer_provider)

    span_exporter = OTLPSpanExporter(
        endpoint=f"https://{cloud_hostname}/observability/traces/otlp/v0",
        headers=headers,
        compression=otlp_compression,
    )

    tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

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


def _to_proto_chat_ctx(chat_ctx: ChatContext) -> agent_pb.agent_session.ChatContext:
    ctx_pb = agent_pb.agent_session.ChatContext()

    for item in chat_ctx.items:
        item_pb = ctx_pb.items.add()

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
                msg.metrics.started_speaking_at.FromSeconds(int(metrics["started_speaking_at"]))
            if "stopped_speaking_at" in metrics:
                msg.metrics.stopped_speaking_at.FromSeconds(int(metrics["stopped_speaking_at"]))
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

            msg.created_at.FromSeconds(int(item.created_at))

        elif item.type == "function_call":
            fc = item_pb.function_call
            fc.id = item.id
            fc.call_id = item.call_id
            fc.arguments = item.arguments
            fc.name = item.name
            fc.created_at.FromSeconds(int(item.created_at))

        elif item.type == "function_call_output":
            fco = item_pb.function_call_output
            fco.id = item.id
            fco.name = item.name
            fco.call_id = item.call_id
            fco.output = item.output
            fco.is_error = item.is_error
            fco.created_at.FromSeconds(int(item.created_at))

        elif item.type == "agent_handoff":
            ah = item_pb.agent_handoff
            ah.id = item.id
            if item.old_agent_id is not None:
                ah.old_agent_id = item.old_agent_id
            ah.new_agent_id = item.new_agent_id
            ah.created_at.FromSeconds(int(item.created_at))

    return ctx_pb


async def _upload_session_report(
    *,
    room_id: str,
    job_id: str,
    cloud_hostname: str,
    report: SessionReport,
    http_session: aiohttp.ClientSession,
) -> None:
    access_token = (
        api.AccessToken()
        .with_observability_grants(api.ObservabilityGrants(write=True))
        .with_ttl(datetime.timedelta(hours=6))
    )
    jwt = access_token.to_jwt()

    header_msg = proto_metrics.MetricsRecordingHeader(
        room_id=room_id, enable_user_data_training=report.enable_user_data_training
    )
    header_bytes = header_msg.SerializeToString()

    chat_history_pb = _to_proto_chat_ctx(report.chat_history)
    chat_history_bytes = chat_history_pb.SerializeToString() if chat_history_pb is not None else b""

    mp = aiohttp.MultipartWriter("form-data")

    part = mp.append(header_bytes)
    part.set_content_disposition("form-data", name="header", filename="header.binpb")
    part.headers["Content-Type"] = "application/protobuf"
    part.headers["Content-Length"] = str(len(header_bytes))

    if chat_history_bytes:
        part = mp.append(chat_history_bytes)
        part.set_content_disposition(
            "form-data", name="chat_history", filename="chat_history.binpb"
        )
        part.headers["Content-Type"] = "application/protobuf"
        part.headers["Content-Length"] = str(len(chat_history_bytes))

    if report.audio_recording_path:
        try:
            async with aiofiles.open(report.audio_recording_path, "rb") as f:
                audio_bytes = await f.read()
        except Exception:
            audio_bytes = b""
        part = mp.append(audio_bytes)
        part.set_content_disposition("form-data", name="audio", filename="recording.ogg")
        part.headers["Content-Type"] = "audio/ogg"
        part.headers["Content-Length"] = str(len(audio_bytes))

    url = f"https://{cloud_hostname}/observability/recordings/v0"
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": mp.content_type,
    }

    async with http_session.post(url, data=mp, headers=headers) as resp:
        resp.raise_for_status()

    logger.info("uploaded")
