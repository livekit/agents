# Copyright 2025 Hamming
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from opentelemetry.util.types import AttributeValue

from .log import logger

DEFAULT_BASE_URL = "https://app.hamming.ai"
WORKSPACE_KEY_HEADER = "X-Workspace-Key"


@dataclass
class HammingTelemetry:
    """Container for the configured OTel providers. Use for shutdown/flush."""

    trace_provider: TracerProvider | None = None
    logger_provider: LoggerProvider | None = None
    meter_provider: MeterProvider | None = None

    def force_flush(self, timeout_ms: int = 5000) -> None:
        """Flush all pending telemetry data."""
        if self.trace_provider:
            self.trace_provider.force_flush(timeout_millis=timeout_ms)
        if self.logger_provider:
            self.logger_provider.force_flush(timeout_millis=timeout_ms)
        if self.meter_provider:
            self.meter_provider.force_flush(timeout_millis=timeout_ms)

    def shutdown(self) -> None:
        """Shutdown all providers."""
        if self.trace_provider:
            self.trace_provider.shutdown()
        if self.logger_provider:
            self.logger_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()


def setup_hamming(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    metadata: dict[str, AttributeValue] | None = None,
    service_name: str = "livekit-voice-agent",
    enable_traces: bool = True,
    enable_logs: bool = True,
    enable_metrics: bool = True,
    metrics_export_interval_ms: int = 5000,
    log_level: int = logging.INFO,
) -> HammingTelemetry:
    """Configure OpenTelemetry to export traces, logs, and metrics to Hamming.

    Call this once in your entrypoint before AgentSession.start().

    Args:
        api_key: Hamming workspace API key. Falls back to HAMMING_API_KEY env var.
        base_url: Hamming base URL. Falls back to HAMMING_BASE_URL or https://app.hamming.ai.
        metadata: Attributes stamped on all spans (e.g., room name, session ID).
        service_name: OTel service name for resource attributes.
        enable_traces: Export traces (default True).
        enable_logs: Export logs (default True).
        enable_metrics: Export metrics (default True).
        metrics_export_interval_ms: Metrics export interval in ms (default 5000).
        log_level: Minimum log level for OTel log export (default logging.INFO).

    Returns:
        HammingTelemetry with provider references for flush/shutdown.

    Raises:
        ValueError: If no API key is provided or found in environment.
    """
    resolved_api_key = api_key or os.environ.get("HAMMING_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Hamming API key required. Pass api_key= or set HAMMING_API_KEY env var. "
            "Get your key from Settings > API Keys in your Hamming dashboard."
        )

    resolved_base_url = (
        base_url or os.environ.get("HAMMING_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")

    headers = {WORKSPACE_KEY_HEADER: resolved_api_key}
    resource = Resource.create({"service.name": service_name})
    telemetry = HammingTelemetry()

    # --- Traces ---
    if enable_traces:
        trace_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(
            endpoint=f"{resolved_base_url}/api/ingest/otlp/v1/traces",
            headers=headers,
        )
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))

        from livekit.agents.telemetry import set_tracer_provider

        set_tracer_provider(trace_provider, metadata=metadata)

        telemetry.trace_provider = trace_provider
        logger.info("Hamming trace export configured: %s", resolved_base_url)

    # --- Logs ---
    if enable_logs:
        log_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(
            endpoint=f"{resolved_base_url}/api/ingest/otlp/v1/logs",
            headers=headers,
        )
        log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(log_provider)

        log_handler = LoggingHandler(level=log_level, logger_provider=log_provider)
        logging.getLogger().addHandler(log_handler)

        telemetry.logger_provider = log_provider
        logger.info("Hamming log export configured: %s", resolved_base_url)

    # --- Metrics ---
    if enable_metrics:
        metric_exporter = OTLPMetricExporter(
            endpoint=f"{resolved_base_url}/api/ingest/otlp/v1/metrics",
            headers=headers,
        )
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=metrics_export_interval_ms,
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)

        telemetry.meter_provider = meter_provider
        logger.info("Hamming metric export configured: %s", resolved_base_url)

    return telemetry
