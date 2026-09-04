from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import random
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import aiofiles
import aiohttp
import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry import context as otel_context, metrics as metrics_api, trace as trace_api
from opentelemetry._logs import LogRecord as OTelLogRecord, get_logger_provider, set_logger_provider
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import NoOpMeterProvider

# _ProxyMeterProvider is what get_meter_provider() returns before anyone calls
# set_meter_provider; there is no public alias for it (opentelemetry-api is
# pinned <1.45, where this import is stable).
from opentelemetry.metrics._internal import _ProxyMeterProvider
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
    LogRecordProcessor,
    ReadableLogRecord,
    ReadWriteLogRecord,
)
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.metrics import (
    Counter as SdkCounter,
    Histogram as SdkHistogram,
    MeterProvider as SdkMeterProvider,
    ObservableCounter as SdkObservableCounter,
    ObservableGauge as SdkObservableGauge,
    ObservableUpDownCounter as SdkObservableUpDownCounter,
    UpDownCounter as SdkUpDownCounter,
)
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AttributeValue

from livekit import api
from livekit.protocol import metrics as proto_metrics

from .._proto import encode_chat_item
from ..log import TRACE_LEVEL, logger
from ..types import (
    ATTRIBUTE_REDACTION_ENABLED,
    ATTRIBUTE_SIMULATION_ENABLED,
    recording_enabled,
)
from . import pii, trace_types, utils as telemetry_utils

if TYPE_CHECKING:
    from ..llm import ChatItem
    from ..observability import Tagger
    from ..voice.agent_session import AgentSessionOptions
    from ..voice.report import SessionReport


_SESSION_OPTION_KEY_ALIASES = {
    "keyterms": "lk.pii.keyterms",
}


def _serialize_session_options(options: AgentSessionOptions) -> dict[str, Any]:
    def _serialize(value: dict[str, Any]) -> dict[str, Any]:
        return {
            _SESSION_OPTION_KEY_ALIASES.get(key, key): (
                _serialize(nested_value) if isinstance(nested_value, dict) else nested_value
            )
            for key, nested_value in value.items()
        }

    return _serialize(vars(options))


class _DynamicTracer(Tracer):
    def __init__(self, instrumenting_module_name: str) -> None:
        self._instrumenting_module_name = instrumenting_module_name
        self._tracer_provider: trace_api.TracerProvider = trace_api.get_tracer_provider()
        self._tracer = trace_api.get_tracer(instrumenting_module_name)

    def set_provider(self, tracer_provider: trace_api.TracerProvider) -> None:
        self._tracer_provider = tracer_provider
        self._tracer = trace_api.get_tracer(
            self._instrumenting_module_name,
            tracer_provider=self._tracer_provider,
        )

    def start_span(self, *args: Any, **kwargs: Any) -> Span:
        return self._tracer.start_span(*args, **kwargs)

    @_agnosticcontextmanager
    def use_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        if telemetry_utils.redaction_enabled():
            kwargs = {
                **kwargs,
                "record_exception": False,
                "set_status_on_exception": False,
            }
        with trace_api.use_span(*args, **kwargs) as span:
            yield span

    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        if telemetry_utils.redaction_enabled():
            kwargs = {
                **kwargs,
                "record_exception": False,
                "set_status_on_exception": False,
            }
        with self._tracer.start_as_current_span(*args, **kwargs) as span:
            yield span


tracer: _DynamicTracer = _DynamicTracer("livekit-agents")


class _UploadGate:
    """Process-wide gate that stops observability uploads once LiveKit Cloud reports data
    recording is disabled for the project. Reset per session from JobContext.init_recording().
    """

    # substrings identifying the 401 "data recording is disabled by owner" rejection. Other
    # 401s ("missing project id", "operation requires observability write grant") share the
    # same status/grpc code, so we match the message text rather than the code.
    _DISABLED_MARKERS = ("data recording is disabled", "disabled by owner")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._disabled = False

    def reset(self) -> None:
        with self._lock:
            self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    def disable(self) -> None:
        with self._lock:
            if self._disabled:
                return
            self._disabled = True
        logger.warning(
            "LiveKit Cloud data recording is disabled for this project; "
            "skipping telemetry and recording uploads for this session"
        )

    @staticmethod
    def is_disabled_response(status_code: int, body: bytes) -> bool:
        """Return True if an upload response means recording is disabled by the project owner."""
        if status_code != 401:
            return False
        text = body.decode("utf-8", "ignore").lower()
        return any(marker in text for marker in _UploadGate._DISABLED_MARKERS)


_upload_gate = _UploadGate()


class _AuthRefreshingSession(requests.Session):
    """requests.Session shared by the OTLP exporters. Injects a fresh auth header on every
    request and, once the project reports recording is disabled, stops uploading and reports
    success so the exporters don't keep logging errors."""

    def __init__(self, header_provider: Callable[[], dict[str, str]]) -> None:
        super().__init__()
        self._header_provider = header_provider

    @staticmethod
    def _make_ok_response() -> requests.Response:
        """A synthetic 200 response so OTLP exporters treat the export as successful."""
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b""
        return resp

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        if _upload_gate.disabled:
            return self._make_ok_response()

        self.headers.update(self._header_provider())
        resp = super().request(*args, **kwargs)
        if _upload_gate.is_disabled_response(resp.status_code, resp.content):
            _upload_gate.disable()
            return self._make_ok_response()
        return resp


@dataclass(frozen=True)
class _JobTelemetry:
    """A job's cloud-telemetry registration, built by
    ``_CloudTelemetry.configure`` and kept on the ``JobContext``. Spans, logs,
    and metric measurements created by the job resolve their attribution and
    upload gating through this object, which stays correct when jobs run
    concurrently on the shared providers (THREAD executor)."""

    # the job's identity and session metadata, stamped on every span/log/metric
    attributes: dict[str, AttributeValue]
    traces_enabled: bool
    logs_enabled: bool


def _job_stamp_attributes() -> dict[str, AttributeValue] | None:
    """The attributes stamped on telemetry created by the job on this context,
    or None outside any job context.

    Stamping is pure attribution and applies regardless of the job's recording
    options — every exporter on a shared provider (an integrator's included)
    sees the same attributes. Whether LiveKit Cloud uploads a record is decided
    separately, by the exportable-jobs registry (``_job_export_state``)."""

    from ..job import get_job_context  # local import: job.py imports this module

    ctx = get_job_context(required=False)
    if ctx is None:
        return None
    if (state := ctx._telemetry_state) is not None:
        return state.attributes
    # recording not (or not yet) initialized: still attribute the telemetry
    return {"room_id": ctx.job.room.sid, "job_id": ctx.job.id}


class _MetadataSpanProcessor(SpanProcessor):
    """Stamps per-job metadata on every span, resolved from the originating
    job's context. The process-wide slot remains as a fallback for spans created
    outside a job context (worker-level telemetry) while a job is running."""

    def __init__(self, metadata: dict[str, AttributeValue] | None = None) -> None:
        self._metadata = dict(metadata) if metadata else {}

    def set_metadata(self, metadata: dict[str, AttributeValue]) -> None:
        # rebind rather than mutate: on_start may read it from another thread
        self._metadata = dict(metadata)

    def clear_metadata(self) -> None:
        self._metadata = {}

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        if (attributes := _job_stamp_attributes()) is not None:
            span.set_attributes(attributes)
            return
        if self._metadata:
            span.set_attributes(self._metadata)


class _MetadataLogProcessor(LogRecordProcessor):
    """Log counterpart of :class:`_MetadataSpanProcessor` — same per-job
    resolution with the process-wide slot as fallback."""

    def __init__(self, metadata: dict[str, AttributeValue] | None = None) -> None:
        self._metadata = dict(metadata) if metadata else {}

    def set_metadata(self, metadata: dict[str, AttributeValue]) -> None:
        self._metadata = dict(metadata)

    def clear_metadata(self) -> None:
        self._metadata = {}

    def on_emit(self, log_data: ReadWriteLogRecord) -> None:
        stamped: dict[str, AttributeValue]
        if (attributes := _job_stamp_attributes()) is not None:
            stamped = dict(attributes)
        else:
            stamped = dict(self._metadata)

        if log_data.log_record.attributes:
            log_data.log_record.attributes.update(stamped)  # type: ignore
        elif stamped:
            log_data.log_record.attributes = stamped

        if log_data.instrumentation_scope:
            if log_data.log_record.attributes is None:
                log_data.log_record.attributes = {}
            log_data.log_record.attributes.update(  # type: ignore
                {"logger.name": log_data.instrumentation_scope.name}
            )

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _job_export_state(
    export_jobs: Mapping[str, _JobTelemetry], attributes: Mapping[str, Any] | None
) -> _JobTelemetry | None:
    """Look up a record's originating job in the exportable-jobs registry.

    Upload to LiveKit Cloud is explicit policy, decoupled from attribution: a
    record is uploaded only while its stamped job_id (globally unique) is
    registered — jobs register in ``configure()`` and are removed after their
    final flush in ``release()`` — and only for the signals that job enabled.
    Records of a job that disabled recording, and records emitted between jobs,
    keep their attributes on every destination but never reach Cloud."""
    if not attributes:
        return None
    job_id = attributes.get("job_id")
    if not isinstance(job_id, str):
        return None
    return export_jobs.get(job_id)


class _GatedSpanExporter(SpanExporter):
    """Wraps the OTLP span exporter so only spans of registered, trace-enabled
    jobs are uploaded (see ``_job_export_state``)."""

    def __init__(self, inner: SpanExporter, export_jobs: Mapping[str, _JobTelemetry]) -> None:
        self._inner = inner
        self._export_jobs = export_jobs

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        exportable = [
            # PII filtered for third-party exporters is put back here: what LiveKit Cloud
            # may receive is the project's setting, applied at its collector. restore_pii
            # is a no-op once that setting mandates redaction.
            pii.restore_pii(s)
            for s in spans
            if (state := _job_export_state(self._export_jobs, s.attributes)) is not None
            and state.traces_enabled
        ]
        if not exportable:
            return SpanExportResult.SUCCESS
        return self._inner.export(exportable)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()


class _GatedLogExporter(LogRecordExporter):
    """Log counterpart of :class:`_GatedSpanExporter`."""

    def __init__(self, inner: LogRecordExporter, export_jobs: Mapping[str, _JobTelemetry]) -> None:
        self._inner = inner
        self._export_jobs = export_jobs

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        exportable = [
            r
            for r in batch
            if (state := _job_export_state(self._export_jobs, r.log_record.attributes)) is not None
            and state.logs_enabled
        ]
        if not exportable:
            return LogRecordExportResult.SUCCESS
        return self._inner.export(exportable)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        # LogRecordExporter.shutdown is untyped in the OTel SDK
        self._inner.shutdown()  # type: ignore[no-untyped-call]


class _BufferingHandler(logging.Handler):
    """Buffers log records in memory for later replay through OTLP."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(record)


class _TraceLevelLoggingHandler(LoggingHandler):
    """Custom LoggingHandler that properly maps TRACE_LEVEL to OTel TRACE severity.

    The default OTel LoggingHandler maps any log level < 10 to UNSPECIFIED,
    but we want TRACE_LEVEL (5) to map to TRACE for proper severity in exports.
    """

    def _translate(self, record: logging.LogRecord) -> OTelLogRecord:
        log_record = super()._translate(record)
        if telemetry_utils.redaction_enabled() and log_record.attributes:
            attributes = dict(log_record.attributes)
            if trace_types.ATTR_EXCEPTION_MESSAGE in attributes:
                attributes[trace_types.ATTR_EXCEPTION_MESSAGE] = (
                    telemetry_utils.REDACTED_EXCEPTION_MESSAGE
                )
            attributes.pop(trace_types.ATTR_EXCEPTION_TRACE, None)
            # callers pass user data through `extra={"lk.pii.<name>": ...}` precisely
            # because a log body cannot be redacted; drop those before export
            log_record.attributes = pii.filter_attributes(attributes)

        # OTel's std_to_otel returns UNSPECIFIED for levels < 10
        # Map our TRACE_LEVEL to OTel's TRACE
        if record.levelno == TRACE_LEVEL:
            log_record.severity_number = SeverityNumber.TRACE
        return log_record


def _prepend_span_processor(provider: trace_sdk.TracerProvider, processor: SpanProcessor) -> None:
    """Attach ``processor`` ahead of every processor already on ``provider``.

    ``on_end`` is dispatched in registration order over a single shared span snapshot,
    so a processor that rewrites the snapshot only protects the exporters registered
    after it. Redaction has to come first, including ahead of exporters the integrator
    attached before handing us their provider.
    """
    provider.add_span_processor(processor)

    multi = getattr(provider, "_active_span_processor", None)
    processors = getattr(multi, "_span_processors", None)
    if not isinstance(processors, tuple) or processor not in processors:
        # a provider shape we don't recognise: the plain append above still covers
        # every exporter the framework attaches itself
        return

    lock = getattr(multi, "_lock", None)
    reordered = (processor, *(p for p in processors if p is not processor))
    if lock is not None:
        with lock:
            multi._span_processors = reordered  # type: ignore[union-attr]
    else:
        multi._span_processors = reordered  # type: ignore[union-attr]


_pii_redaction_installed: weakref.WeakSet[trace_sdk.TracerProvider] = weakref.WeakSet()


def _prepend_log_processor(provider: LoggerProvider, processor: LogRecordProcessor) -> None:
    """Log counterpart of :func:`_prepend_span_processor` — same dispatch-order reasoning."""
    provider.add_log_record_processor(processor)

    multi = getattr(provider, "_multi_log_record_processor", None)
    processors = getattr(multi, "_log_record_processors", None)
    if not isinstance(processors, tuple) or processor not in processors:
        return

    lock = getattr(multi, "_lock", None)
    reordered = (processor, *(p for p in processors if p is not processor))
    if lock is not None:
        with lock:
            multi._log_record_processors = reordered  # type: ignore[union-attr]
    else:
        multi._log_record_processors = reordered  # type: ignore[union-attr]


def _install_pii_redaction(
    tracer_provider: trace_api.TracerProvider, *, allow_pii: bool | None = None
) -> None:
    """Install in-process PII stripping on an SDK provider, at most once per provider."""
    if not isinstance(tracer_provider, trace_sdk.TracerProvider):
        # processors can only be attached to an SDK provider; a no-op/proxy provider
        # exports nothing, so there is nothing to strip
        return
    if tracer_provider in _pii_redaction_installed:
        return
    _pii_redaction_installed.add(tracer_provider)
    if allow_pii is None:
        allow_pii = telemetry_utils.allow_pii_from_env()
    _prepend_span_processor(
        tracer_provider,
        # PII flows to every exporter unless withheld: the GenAI conventions are only
        # useful to a backend that can render the conversation
        pii._PIIFilteringSpanProcessor(allow_pii=allow_pii if allow_pii is not None else True),
    )


def set_tracer_provider(
    tracer_provider: trace_api.TracerProvider,
    *,
    metadata: dict[str, AttributeValue] | None = None,
    allow_pii: bool | None = None,
) -> None:
    """Set the tracer provider for the livekit-agents.

    Args:
        tracer_provider (TracerProvider): The tracer provider to set.
        metadata (dict[str, AttributeValue] | None, optional): Metadata to set on all spans. Defaults to None.
        allow_pii (bool | None, optional): Whether the exporters on this provider *other
            than LiveKit Cloud's* may receive conversational content, tool payloads and
            other user data. What LiveKit Cloud receives is the project's PII setting in
            the dashboard, which this cannot widen or narrow. Defaults to
            ``True`` (or ``LIVEKIT_TELEMETRY_ALLOW_PII``, when set), since a GenAI
            backend can only render the conversation if it receives it. Pass ``False``
            to strip PII in-process before every exporter but LiveKit Cloud's, leaving
            them the non-content attributes. Ignored when the project mandates redaction
            — that setting is not weakened from here.
    """
    if metadata and isinstance(tracer_provider, trace_sdk.TracerProvider):
        tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))

    _install_pii_redaction(tracer_provider, allow_pii=allow_pii)
    tracer.set_provider(tracer_provider)


_TOKEN_TTL = timedelta(hours=6)
_TOKEN_REFRESH_MARGIN = timedelta(minutes=5)


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
            .with_ttl(_TOKEN_TTL)
        )
        self._auth_header = f"Bearer {access_token.to_jwt()}"
        self._expires_at = datetime.now(timezone.utc) + _TOKEN_TTL

    def __call__(self) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        if now >= self._expires_at - _TOKEN_REFRESH_MARGIN:
            with self._lock:
                if now >= self._expires_at - _TOKEN_REFRESH_MARGIN:
                    self._refresh()
        return {"Authorization": self._auth_header}


_TELEMETRY_SHUTDOWN_TIMEOUT = 10.0


def _run_bounded(action: str, targets: list[tuple[str, Callable[[], Any]]], timeout: float) -> None:
    """Run each target on its own daemon thread with a hard wall-clock bound.

    ``provider.shutdown()`` internally joins its exporter worker with a 30s
    default timeout per provider (and ``force_flush`` ignores its timeout arg
    in the current SDK — see #4623). Across tracer/logger/meter that's up to
    ~90s, enough to stall the caller's event loop past the supervisor's 60s
    ping/pong deadline when the OTLP endpoint is rate-limiting or unreachable.

    Each target runs in its *own* daemon thread, in parallel. That matters for
    two reasons:
      1) Main-thread wait is bounded by ``max`` of the targets, not the ``sum``.
      2) ``BatchProcessor.shutdown()`` sets ``_shutdown = True`` as its first
         action; running in parallel guarantees that flag gets set on every
         processor within milliseconds, even if one hangs in
         ``worker_thread.join``. Any later re-entry (e.g. Python's
         ``logging.shutdown()`` may spawn a *non-daemon* thread via
         ``LoggingHandler.flush`` → ``force_flush`` — see opentelemetry-python
         PR #4636) then short-circuits instead of hanging process exit.

    Any unfinished work stays on the daemon threads and is discarded at
    process exit.

    Upstream context:
    - https://github.com/open-telemetry/opentelemetry-python/issues/4623
      (TracerProvider.shutdown() has no configurable timeout — still open)
    """
    if not targets:
        return

    def _run_one(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("telemetry %s failed (%s)", action, name)

    threads = [
        threading.Thread(
            target=_run_one,
            args=(name, fn),
            name=f"livekit-telemetry-{action}-{name}",
            daemon=True,
        )
        for name, fn in targets
    ]
    for t in threads:
        t.start()

    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))

    if any(t.is_alive() for t in threads):
        logger.warning("telemetry %s exceeded %.1fs; continuing", action, timeout)


class _CloudTelemetry:
    """Process-lifetime LiveKit Cloud telemetry infrastructure.

    ``configure()`` and ``release()`` run once per job (from
    ``JobContext.init_recording`` and ``JobContext._on_cleanup``), but worker
    processes outlive jobs: the THREAD executor runs every job of the worker in
    one shared process — possibly several concurrently — and an integrator's
    OTel providers (e.g. the Langfuse setup in the tracing docs, Logfire,
    dd-trace) are configured once at process start. State is therefore split by
    lifetime:

    * **process** — the OTLP exporters, the framework's batch and metadata
      processors, any provider the framework itself creates, and the
      root-logger handler instance. Created lazily on first use and shut down
      exactly once, at process exit (bounded, via atexit). A provider supplied
      by the integrator is *adopted*: the framework attaches its own processors
      to it and never shuts it down.
    * **job** — attribution and upload policy, kept independent of each other.
      Every span/log/metric is stamped with its originating job's attributes,
      resolved through the job contextvar, regardless of the job's recording
      options — every destination on a shared provider sees consistent
      attribution. Whether LiveKit Cloud uploads a record is decided by the
      exportable-jobs registry (``_export_jobs``): the gated exporters upload a
      record only while its stamped job_id is registered with that signal
      enabled. ``release()`` flushes the batch processors (bounded) while the
      job is still registered, then unregisters it.

    The ``Resource`` on a framework-created tracer/logger provider is built by
    the first configuring job; the meter provider's resource carries only
    process-stable identity, with per-job attribution on each measurement.
    Telemetry emitted outside any job context is stamped from a fallback slot
    holding the most recently configured job's attributes, cleared once no job
    remains registered.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # the exportable-jobs registry: job_id (globally unique) -> the job's
        # telemetry state, registered by configure() and removed after the final
        # flush in release(). The gated exporters hold a reference to this exact
        # dict, so it is mutated in place, never rebound.
        self._export_jobs: dict[str, _JobTelemetry] = {}
        self._atexit_registered = False
        self._session: _AuthRefreshingSession | None = None
        self._observability_url: str | None = None

        # everything the framework created, shut down once at process exit:
        # providers it built (constructed with shutdown_on_exit=False) and every
        # batch processor it attached — appended at each creation site. A
        # provider supplied by the integrator is never in this list.
        self._exit_targets: list[tuple[str, Callable[[], Any]]] = []

        # traces
        self._trace_provider_attached: trace_sdk.TracerProvider | None = None
        self._span_metadata_processor: _MetadataSpanProcessor | None = None
        self._span_batch_processor: BatchSpanProcessor | None = None

        # logs
        self._logger_provider: LoggerProvider | None = None
        self._log_metadata_processor: _MetadataLogProcessor | None = None
        self._log_batch_processor: BatchLogRecordProcessor | None = None
        self._log_handler: _TraceLevelLoggingHandler | None = None

        # metrics
        self._owned_meter_provider: SdkMeterProvider | None = None
        self._metrics_unavailable = False

    @property
    def logger_provider(self) -> LoggerProvider | None:
        """The logger provider the framework exports through (ours or adopted)."""
        return self._logger_provider

    @property
    def log_handler(self) -> _TraceLevelLoggingHandler | None:
        """The framework's own root-logger OTLP handler, if logs were configured."""
        return self._log_handler

    def configure(
        self,
        *,
        room_id: str,
        job_id: str,
        agent_name: str = "",
        observability_url: str,
        enable_traces: bool = True,
        enable_logs: bool = True,
        metadata: dict[str, AttributeValue] | None = None,
    ) -> _JobTelemetry:
        """Set up (or reuse) the pipelines for a job. Returns the job's
        telemetry state, for the caller to keep on the JobContext — per-record
        stamping and upload gating resolve it from there."""
        base_metadata: dict[str, AttributeValue] = {"room_id": room_id, "job_id": job_id}
        if agent_name:
            # identifies the agent for LiveKit Cloud agent insights (explicit dispatch
            # only; the default dispatch has no agent name). Included in both the
            # resource (traces) and the session metadata (spans + logs).
            base_metadata[trace_types.ATTR_AGENT_NAME] = agent_name
        # cloud agent id and deployment provided by LiveKit Cloud via env vars.
        # Included in both the resource and the session metadata like agent_name;
        # omitted when unset.
        if cloud_agent_id := os.environ.get("LIVEKIT_AGENT_ID"):
            base_metadata[trace_types.ATTR_CLOUD_AGENT_ID] = cloud_agent_id
        if deployment_id := os.environ.get("LIVEKIT_AGENT_DEPLOYMENT"):
            base_metadata[trace_types.ATTR_DEPLOYMENT_ID] = deployment_id
        session_metadata = dict(base_metadata)
        if metadata:
            session_metadata.update(metadata)

        with self._lock:
            if self._observability_url is None:
                self._observability_url = observability_url
            elif observability_url != self._observability_url:
                logger.warning(
                    "observability endpoint changed across jobs; keeping %s",
                    self._observability_url,
                )
            url = self._observability_url

            if self._session is None:
                self._session = _AuthRefreshingSession(_AuthHeaderProvider())

            resource = Resource.create({SERVICE_NAME: "livekit-agents", **base_metadata})

            if enable_traces:
                self._ensure_trace_pipeline(resource, url)
                if self._span_metadata_processor is not None:
                    self._span_metadata_processor.set_metadata(session_metadata)

            # Always set up the logger provider — it's needed for session reports,
            # evaluations, and chat history, not just Python log export.
            self._ensure_logger_provider()

            if enable_logs:
                self._ensure_log_pipeline(url)
                if self._log_metadata_processor is not None:
                    self._log_metadata_processor.set_metadata(session_metadata)
                if self._log_handler is not None:
                    root = logging.getLogger()
                    if self._log_handler not in root.handlers:
                        root.addHandler(self._log_handler)

            # the meter provider outlives the job (the OTel metrics global is
            # set-once), so its resource carries only process-stable identity;
            # per-job room_id/job_id ride on each measurement instead
            # (otel_metrics._job_attrs)
            process_metadata = {
                k: v for k, v in base_metadata.items() if k not in ("room_id", "job_id")
            }
            meter_resource = Resource.create({SERVICE_NAME: "livekit-agents", **process_metadata})
            self._ensure_meter_provider(meter_resource, url)

            if not self._atexit_registered:
                self._atexit_registered = True
                atexit.register(self.shutdown_at_exit)

            state = _JobTelemetry(
                attributes=session_metadata,
                traces_enabled=enable_traces,
                logs_enabled=enable_logs,
            )
            self._export_jobs[job_id] = state

        return state

    def _ensure_trace_pipeline(self, resource: Resource, url: str) -> None:
        # Check if a tracer provider is not set and set one up
        # below shows how the ProxyTracerProvider is returned when none have been setup
        # https://github.com/open-telemetry/opentelemetry-python/blob/0018c0030bac9bdce4487fe5fcb3ec6a542ec904/opentelemetry-api/src/opentelemetry/trace/__init__.py#L555
        provider: trace_api.TracerProvider
        if isinstance(
            tracer._tracer_provider,
            (trace_api.ProxyTracerProvider, trace_api.NoOpTracerProvider),
        ):
            owned = trace_sdk.TracerProvider(resource=resource, shutdown_on_exit=False)
            # shutting the provider down also shuts down every processor attached
            # to it (processor shutdown is idempotent, so overlap is harmless)
            self._exit_targets.append(("TracerProvider", owned.shutdown))
            set_tracer_provider(owned)
            provider = owned
        else:
            # the integrator's provider (or ours, from an earlier job)
            provider = tracer._tracer_provider

        if not isinstance(provider, trace_sdk.TracerProvider):
            # processors can only be attached to an SDK provider
            return

        _install_pii_redaction(provider)

        if provider is self._trace_provider_attached:
            return

        if self._trace_provider_attached is not None:
            # the tracer provider was replaced mid-process (set_tracer_provider
            # called after a job already exported); re-attach to the new one and
            # retire the old pipeline
            logger.warning("tracer provider changed; re-attaching LiveKit Cloud span exporter")
            if self._span_metadata_processor is not None:
                self._span_metadata_processor.clear_metadata()
            if self._span_batch_processor is not None:
                # shut the old pipeline down in the background: shutdown drains
                # its queue, exporting the prior jobs' remaining stamped spans,
                # then goes quiet. shutdown() is idempotent, so shutting it down
                # again at process exit is harmless.
                threading.Thread(
                    target=self._span_batch_processor.shutdown,
                    name="livekit-telemetry-retire-BatchSpanProcessor",
                    daemon=True,
                ).start()

        assert self._session is not None
        span_exporter = OTLPSpanExporter(
            endpoint=f"{url}/observability/traces/otlp/v0",
            compression=Compression.Gzip,
            session=self._session,
        )
        self._span_metadata_processor = _MetadataSpanProcessor()
        self._span_batch_processor = BatchSpanProcessor(
            _GatedSpanExporter(span_exporter, self._export_jobs)
        )
        self._exit_targets.append(("BatchSpanProcessor", self._span_batch_processor.shutdown))
        provider.add_span_processor(self._span_metadata_processor)
        provider.add_span_processor(self._span_batch_processor)
        self._trace_provider_attached = provider

    def _ensure_logger_provider(self) -> None:
        if self._logger_provider is not None:
            return
        current = get_logger_provider()
        if isinstance(current, LoggerProvider):
            # an SDK provider the integrator set up — adopt it
            self._logger_provider = current
        else:
            owned = LoggerProvider(shutdown_on_exit=False)
            self._exit_targets.append(("LoggerProvider", owned.shutdown))
            set_logger_provider(owned)
            self._logger_provider = owned

        # ahead of any exporter already on the provider, as for spans
        _prepend_log_processor(self._logger_provider, pii._PIIFilteringLogProcessor())

    def _ensure_log_pipeline(self, url: str) -> None:
        if self._log_batch_processor is not None or self._logger_provider is None:
            return
        assert self._session is not None
        log_exporter = OTLPLogExporter(
            endpoint=f"{url}/observability/logs/otlp/v0",
            compression=Compression.Gzip,
            session=self._session,
        )
        self._log_metadata_processor = _MetadataLogProcessor()
        self._log_batch_processor = BatchLogRecordProcessor(
            _GatedLogExporter(log_exporter, self._export_jobs)
        )
        self._exit_targets.append(("BatchLogRecordProcessor", self._log_batch_processor.shutdown))
        self._logger_provider.add_log_record_processor(self._log_metadata_processor)
        self._logger_provider.add_log_record_processor(self._log_batch_processor)
        self._log_handler = _TraceLevelLoggingHandler(
            level=logging.NOTSET, logger_provider=self._logger_provider
        )

    def _ensure_meter_provider(self, resource: Resource, url: str) -> None:
        if self._owned_meter_provider is not None or self._metrics_unavailable:
            return
        current = metrics_api.get_meter_provider()
        if not isinstance(current, (_ProxyMeterProvider, NoOpMeterProvider)):
            # the integrator configured their own meter provider; the metrics API
            # has no way to attach a reader to it, so Cloud metrics are skipped
            self._metrics_unavailable = True
            return
        assert self._session is not None
        metric_exporter = OTLPMetricExporter(
            endpoint=f"{url}/observability/metrics/otlp/v0",
            compression=Compression.Gzip,
            session=self._session,
            preferred_temporality={
                SdkCounter: AggregationTemporality.DELTA,
                SdkUpDownCounter: AggregationTemporality.DELTA,
                SdkHistogram: AggregationTemporality.DELTA,
                SdkObservableCounter: AggregationTemporality.DELTA,
                SdkObservableUpDownCounter: AggregationTemporality.DELTA,
                SdkObservableGauge: AggregationTemporality.DELTA,
            },
        )
        reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
        provider = SdkMeterProvider(
            resource=resource, metric_readers=[reader], shutdown_on_exit=False
        )
        metrics_api.set_meter_provider(provider)
        if metrics_api.get_meter_provider() is not provider:
            # the set-once global was already consumed (e.g. by a NoOp provider):
            # no instrument would ever reach our provider, so don't leave its
            # periodic reader running
            threading.Thread(
                target=provider.shutdown,
                name="livekit-telemetry-orphan-meter-shutdown",
                daemon=True,
            ).start()
            self._metrics_unavailable = True
            return
        self._exit_targets.append(("MeterProvider", provider.shutdown))
        self._owned_meter_provider = provider

    def release(self, job_id: str, timeout: float = _TELEMETRY_SHUTDOWN_TIMEOUT) -> None:
        """Per-job teardown: flush the framework's exporters (bounded) while the
        job is still registered, then remove it from the exportable-jobs
        registry — records stamped with its id no longer upload. When no job
        remains, the fallback stamp is cleared and the log handler detached.

        Nothing is shut down here: providers (ours or the integrator's) and the
        batch processors keep running for the next job in the process. Final
        shutdown happens once, at process exit (:meth:`shutdown_at_exit`).
        """
        with self._lock:
            if job_id not in self._export_jobs:
                return  # never configured, or already released

            flush_targets: list[tuple[str, Callable[[], Any]]] = []
            if self._span_batch_processor is not None:
                flush_targets.append(("spans", self._span_batch_processor.force_flush))
            if self._log_batch_processor is not None:
                flush_targets.append(("logs", self._log_batch_processor.force_flush))
            if self._owned_meter_provider is not None:
                flush_targets.append(("metrics", self._owned_meter_provider.force_flush))

        # flush before unregistering, so the job's remaining telemetry is exported
        _run_bounded("flush", flush_targets, timeout)

        with self._lock:
            self._export_jobs.pop(job_id, None)
            if self._export_jobs:
                # another job is still running in this process; keep exporting
                return
            if self._span_metadata_processor is not None:
                self._span_metadata_processor.clear_metadata()
            if self._log_metadata_processor is not None:
                self._log_metadata_processor.clear_metadata()
            handler = self._log_handler

        if handler is not None:
            # detach only our own handler — the integrator may have their own
            # OTel LoggingHandler on the root logger
            logging.getLogger().removeHandler(handler)

    def shutdown_at_exit(self, timeout: float = _TELEMETRY_SHUTDOWN_TIMEOUT) -> None:
        """Shut down what the framework created — once, at process exit, bounded.

        ``_exit_targets`` is appended at each creation site: providers the
        framework created (built with ``shutdown_on_exit=False``, so this is the
        only shutdown they get) and every batch processor the framework attached.
        Processor shutdown is idempotent, so a processor also covered by its
        owned provider is harmless — and a provider the integrator supplied is
        never in the list.
        """
        with self._lock:
            targets = list(self._exit_targets)
            handler = self._log_handler

        if handler is not None:
            logging.getLogger().removeHandler(handler)

        _run_bounded("shutdown", targets, timeout)


_cloud = _CloudTelemetry()


def _cloud_log_handler() -> _TraceLevelLoggingHandler | None:
    """The framework's own root-logger OTLP handler, if configured for this job."""
    return _cloud.log_handler


def _setup_cloud_tracer(
    *,
    room_id: str,
    job_id: str,
    agent_name: str = "",
    observability_url: str,
    enable_traces: bool = True,
    enable_logs: bool = True,
    metadata: dict[str, AttributeValue] | None = None,
) -> _JobTelemetry:
    _upload_gate.reset()
    return _cloud.configure(
        room_id=room_id,
        job_id=job_id,
        agent_name=agent_name,
        observability_url=observability_url,
        enable_traces=enable_traces,
        enable_logs=enable_logs,
        metadata=metadata,
    )


def _chat_item_span_attribute(item: ChatItem) -> dict:
    return MessageToDict(encode_chat_item(item), preserving_proto_field_name=True)


def _parse_retry_delay(body: bytes) -> float | None:
    """Return the delay from a protobuf ``RetryInfo`` detail, if present."""
    from google.rpc import error_details_pb2, status_pb2  # type: ignore[import-untyped]

    try:
        status = status_pb2.Status()
        status.ParseFromString(body)
        for detail in status.details:
            retry_info = error_details_pb2.RetryInfo()
            if detail.Unpack(retry_info):
                delay = retry_info.retry_delay
                return float(delay.seconds + delay.nanos / 1e9)
    except Exception:
        pass

    return None


_RECORDING_UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=900, sock_connect=30)
_RECORDING_UPLOAD_MAX_RETRIES = 3


def _recording_upload_retry_delay(attempt: int) -> float:
    return random.uniform(0.0, min(2.0**attempt, 8.0))


async def _upload_session_report(
    *,
    agent_name: str,
    observability_url: str,
    report: SessionReport,
    tagger: Tagger,
    http_session: aiohttp.ClientSession,
    metadata: dict[str, AttributeValue] | None = None,
) -> None:
    if _upload_gate.disabled:
        return
    metadata = metadata or {}

    def _get_logger(name: str) -> Any:
        # prefer the provider the framework exports through; the OTel global may
        # have been claimed by an integrator provider we could not adopt
        provider = _cloud.logger_provider or get_logger_provider()
        return provider.get_logger(
            name=name,
            attributes={
                "room_id": report.room_id,
                "job_id": report.job_id,
                "room": report.room,
                **metadata,
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
    recording_options = report.options.recording_options

    if recording_enabled(recording_options):
        _log(
            chat_logger,
            body="session report",
            timestamp=int((report.started_at or report.timestamp or 0) * 1e9),
            attributes={
                "session.options": _serialize_session_options(report.options),
                "session.report_timestamp": report.timestamp,
                "session.tags": sorted(tagger.tags) if tagger.tags else None,
                "agent_name": agent_name,
                "sdk_version": report.sdk_version,
                "usage": [
                    {k: v for k, v in u.model_dump().items() if v != 0 and v != 0.0}
                    for u in report.model_usage
                ]
                if report.model_usage
                else None,
            },
        )

    if recording_options["transcript"]:
        for item in report.chat_history.items:
            item_log = _chat_item_span_attribute(item)
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

    for tag, entry in tagger._tags.items():
        if entry.metadata:
            _log(
                eval_logger,
                body="tag",
                timestamp=int(entry.timestamp * 1e9),
                attributes={"tag": {"name": tag, "metadata": entry.metadata}},
            )

    if tagger.outcome:
        is_fail = tagger.outcome == "fail"
        outcome_data: dict[str, Any] = {"outcome": tagger.outcome}
        if tagger.outcome_reason:
            outcome_data["reason"] = tagger.outcome_reason

        _log(
            eval_logger,
            body="outcome",
            timestamp=int(report.timestamp * 1e9),
            attributes={"outcome": outcome_data},
            severity=SeverityNumber.ERROR if is_fail else SeverityNumber.UNSPECIFIED,
            severity_text="error" if is_fail else "unspecified",
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
        job_id=report.job_id,
        simulated=bool(metadata.get(ATTRIBUTE_SIMULATION_ENABLED, False)),
        redaction_enabled=bool(metadata.get(ATTRIBUTE_REDACTION_ENABLED, False)),
    )
    header_msg.start_time.FromMilliseconds(int((report.audio_recording_started_at or 0) * 1000))
    header_bytes = header_msg.SerializeToString()

    chat_history_json = ""
    if recording_options["transcript"]:
        chat_history_json = json.dumps(report.chat_history.to_dict(exclude_timestamp=False))

    audio_bytes = b""
    if has_audio and report.audio_recording_path:
        try:
            async with aiofiles.open(report.audio_recording_path, "rb") as f:
                audio_bytes = await f.read()
        except Exception as e:
            audio_bytes = b""
            logger.warning(
                "failed to read audio recording for session report upload, "
                "uploading without the audio part (path=%s)",
                report.audio_recording_path,
                exc_info=e,
            )

    url = f"{observability_url}/observability/recordings/v0"

    def _build_multipart() -> aiohttp.MultipartWriter:
        mp = aiohttp.MultipartWriter("form-data")

        part = mp.append(header_bytes)
        part.set_content_disposition("form-data", name="header", filename="header.binpb")
        part.headers["Content-Type"] = "application/protobuf"
        part.headers["Content-Length"] = str(len(header_bytes))

        if recording_options["transcript"]:
            part = mp.append(chat_history_json)
            part.set_content_disposition(
                "form-data", name="chat_history", filename="chat_history.json"
            )
            part.headers["Content-Type"] = "application/json"
            part.headers["Content-Length"] = str(len(chat_history_json))

        if audio_bytes:
            part = mp.append(audio_bytes)
            part.set_content_disposition("form-data", name="audio", filename="recording.ogg")
            part.headers["Content-Type"] = "audio/ogg"
            part.headers["Content-Length"] = str(len(audio_bytes))

        return mp

    for attempt in range(_RECORDING_UPLOAD_MAX_RETRIES + 1):
        mp = _build_multipart()
        headers = {
            "Authorization": f"Bearer {jwt}",
            "Content-Type": mp.content_type,
        }

        logger.debug("uploading session report to LiveKit Cloud")
        retry: tuple[float, str] | None = None
        try:
            async with http_session.post(
                url,
                data=mp,
                headers=headers,
                timeout=_RECORDING_UPLOAD_TIMEOUT,
            ) as resp:
                if resp.status < 400:
                    break

                body = await resp.read()
                if _upload_gate.is_disabled_response(resp.status, body):
                    _upload_gate.disable()
                    return

                retry_delay = _parse_retry_delay(body)
                if retry_delay is None or attempt == _RECORDING_UPLOAD_MAX_RETRIES:
                    resp.raise_for_status()
                else:
                    retry = (retry_delay, f"status {resp.status}")
        except aiohttp.ClientSSLError:
            raise
        except (aiohttp.ConnectionTimeoutError, aiohttp.ClientConnectorError) as e:
            if attempt == _RECORDING_UPLOAD_MAX_RETRIES:
                raise
            retry = (_recording_upload_retry_delay(attempt), type(e).__name__)

        if retry is None:
            break
        retry_delay, failure = retry
        logger.warning(
            "recording upload failed (%s, attempt %d/%d), retrying in %.1fs",
            failure,
            attempt + 1,
            _RECORDING_UPLOAD_MAX_RETRIES + 1,
            retry_delay,
        )
        await asyncio.sleep(retry_delay)

    logger.debug("finished uploading")


def _shutdown_telemetry(job_id: str, timeout: float = _TELEMETRY_SHUTDOWN_TIMEOUT) -> None:
    """Per-job telemetry teardown, with a hard wall-clock bound.

    This flushes the exporters the framework attached, so LiveKit Cloud receives
    the job's telemetry at job end, and unregisters the job from the export
    registry — it does NOT shut providers down. Worker processes are reused
    across jobs (the THREAD executor runs every job in one shared process), and
    a provider configured by the integrator at process start (Langfuse, Logfire,
    dd-trace) must keep running for later jobs; the framework's own providers
    likewise stay alive because the OTel logger/meter globals are set-once. See
    :class:`_CloudTelemetry` for the ownership model; final shutdown happens
    once at process exit.
    """
    _cloud.release(job_id, timeout)
