# Copyright 2025 Hamming, Inc.
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
from dataclasses import dataclass
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
_OTLP_BASE_PATH = "/api/ingest/otlp/v1"

_telemetry: HammingTelemetry | None = None


@dataclass
class HammingTelemetry:
    """Container for the configured OTel providers. Use for shutdown/flush."""

    trace_provider: TracerProvider | None = None
    logger_provider: LoggerProvider | None = None
    meter_provider: MeterProvider | None = None

    def _remove_log_handlers(self) -> None:
        """Remove OTel LoggingHandler from root logger to prevent deadlock.

        During flush/shutdown, the BatchLogRecordProcessor may emit logs that
        feed back through the LoggingHandler into the provider mid-flush.
        Follows the pattern from livekit-agents/livekit/agents/telemetry/traces.py.
        """
        root = logging.getLogger()
        for h in root.handlers[:]:
            if isinstance(h, LoggingHandler):
                root.removeHandler(h)

    def force_flush(self, timeout_ms: int = 5000) -> None:
        """Flush all pending telemetry data.

        Attempts to flush all configured providers. If any provider fails,
        the remaining providers are still flushed before the first error
        is re-raised.
        """
        if self.logger_provider:
            self._remove_log_handlers()

        errors: list[Exception] = []
        for provider in (self.trace_provider, self.logger_provider, self.meter_provider):
            if provider:
                try:
                    provider.force_flush(timeout_millis=timeout_ms)
                except Exception as e:
                    logger.warning("Flush failed for %s: %s", type(provider).__name__, e)
                    errors.append(e)
        if errors:
            raise errors[0]

    def shutdown(self) -> None:
        """Shutdown all providers.

        Attempts to shut down all configured providers. If any provider fails,
        the remaining providers are still shut down before the first error
        is re-raised.
        """
        if self.logger_provider:
            self._remove_log_handlers()

        errors: list[Exception] = []
        for provider in (self.trace_provider, self.logger_provider, self.meter_provider):
            if provider:
                try:
                    provider.shutdown()
                except Exception as e:
                    logger.warning("Shutdown failed for %s: %s", type(provider).__name__, e)
                    errors.append(e)
        if errors:
            raise errors[0]

    def __repr__(self) -> str:
        signals = []
        if self.trace_provider:
            signals.append("traces")
        if self.logger_provider:
            signals.append("logs")
        if self.meter_provider:
            signals.append("metrics")
        return f"HammingTelemetry(signals=[{', '.join(signals)}])"


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
    global _telemetry
    if _telemetry is not None:
        logger.warning(
            "setup_hamming() already called; returning existing instance. "
            "New parameters are ignored."
        )
        return _telemetry

    resolved_api_key = api_key or os.environ.get("HAMMING_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Hamming API key required. Pass api_key= or set HAMMING_API_KEY env var. "
            "Get your key from Settings > API Keys in your Hamming dashboard."
        )

    resolved_base_url = (base_url or os.environ.get("HAMMING_BASE_URL") or DEFAULT_BASE_URL).rstrip(
        "/"
    )

    headers = {WORKSPACE_KEY_HEADER: resolved_api_key}
    resource = Resource.create({"service.name": service_name})
    telemetry = HammingTelemetry()

    # --- Traces ---
    if enable_traces:
        trace_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(
            endpoint=f"{resolved_base_url}{_OTLP_BASE_PATH}/traces",
            headers=headers,
        )
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))

        # Set global OTel tracer so user spans also go to Hamming
        trace.set_tracer_provider(trace_provider)

        # Set LiveKit-specific tracer with metadata for span correlation
        from livekit.agents.telemetry import set_tracer_provider

        set_tracer_provider(trace_provider, metadata=metadata)

        telemetry.trace_provider = trace_provider
        logger.info("Hamming trace export configured: %s", resolved_base_url)

    # --- Logs ---
    if enable_logs:
        log_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(
            endpoint=f"{resolved_base_url}{_OTLP_BASE_PATH}/logs",
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
            endpoint=f"{resolved_base_url}{_OTLP_BASE_PATH}/metrics",
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

    _telemetry = telemetry
    return telemetry
