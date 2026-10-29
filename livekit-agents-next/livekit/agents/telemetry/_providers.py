"""Adopt application providers and own the process-lifetime Cloud SDK pipelines."""

from __future__ import annotations

import atexit
import logging
import os
import threading
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence

from opentelemetry import context as otel_context, metrics as metrics_api, trace as trace_api
from opentelemetry._logs import get_logger_provider
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics._internal import _ProxyMeterProvider
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LogRecordProcessor,
    ReadWriteLogRecord,
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
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.util.types import AttributeValue

from .. import _core
from ..types import ATTRIBUTE_REDACTION_ENABLED
from . import (
    _context as tracing,
    _destination as destination,
    trace_types,
    utils as telemetry_utils,
)
from ._destination import (
    _AuthRefreshingSession,
    _GatedLogExporter,
    _GatedSpanExporter,
    _JobTelemetry,
    _TraceLevelLoggingHandler,
)
from ._export import (
    BoundedLogRecordProcessor,
    BoundedSpanProcessor,
    GuardedMetricExporter,
    attribute_bytes,
    policy,
    record,
    run_bounded,
)


def _job_stamp_attributes() -> Mapping[str, AttributeValue] | None:
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
    """Stamp the identity captured by the current invocation, never the last configured job."""

    def __init__(self, metadata: dict[str, AttributeValue] | None = None) -> None:
        self._provider_metadata = dict(metadata or {})

    def on_start(
        self, span: trace_sdk.Span, parent_context: otel_context.Context | None = None
    ) -> None:
        attributes = {
            **self._provider_metadata,
            **(_job_stamp_attributes() or {}),
            "lk.telemetry.capture_content": (span.attributes or {}).get(
                "lk.telemetry.capture_content", tracing.capture_content_enabled()
            ),
        }
        if attributes:
            existing = span.attributes or {}
            span.set_attributes(
                {key: value for key, value in attributes.items() if key not in existing}
            )


_metadata_processors: weakref.WeakKeyDictionary[
    trace_sdk.TracerProvider, _MetadataSpanProcessor
] = weakref.WeakKeyDictionary()
_provider_setup_lock = threading.Lock()


def _install_metadata(
    provider: trace_sdk.TracerProvider, metadata: Mapping[str, AttributeValue] | None = None
) -> _MetadataSpanProcessor:
    with _provider_setup_lock:
        processor = _metadata_processors.get(provider)
        if processor is None:
            processor = _MetadataSpanProcessor(dict(metadata or {}))
            provider.add_span_processor(processor)
            _metadata_processors[provider] = processor
        elif metadata is not None:
            processor._provider_metadata = dict(metadata)
        return processor


class _MetadataLogProcessor(LogRecordProcessor):
    """Stamp log identity on the emitting invocation before asynchronous export."""

    def __init__(self, metadata: dict[str, AttributeValue] | None = None) -> None:
        self._metadata = dict(metadata or {})

    def on_emit(self, log_data: ReadWriteLogRecord) -> None:
        supplied = log_data.log_record.attributes or {}
        stamped = {
            **self._metadata,
            **(
                {}
                if "lk.telemetry.capture_content" in supplied
                else (_job_stamp_attributes() or {})
            ),
            "lk.telemetry.capture_content": (log_data.log_record.attributes or {}).get(
                "lk.telemetry.capture_content", tracing.capture_content_enabled()
            ),
        }
        if log_data.instrumentation_scope:
            stamped["logger.name"] = log_data.instrumentation_scope.name
        if stamped:
            log_data.log_record.attributes = {**stamped, **(log_data.log_record.attributes or {})}

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


_pii_redaction_installed: weakref.WeakSet[trace_api.TracerProvider] = weakref.WeakSet()


def _install_pii_redaction(
    tracer_provider: trace_api.TracerProvider, *, allow_pii: bool | None = None
) -> None:
    """Report provider-wide redaction requests; destination wrappers own filtering."""
    if allow_pii is None:
        allow_pii = telemetry_utils.allow_pii_from_env()
    if allow_pii is False and tracer_provider not in _pii_redaction_installed:
        _pii_redaction_installed.add(tracer_provider)
        warnings.warn(
            "Provider-wide PII filtering cannot modify application processors. Wrap each "
            "application exporter in RedactingSpanExporter before adding it to the provider.",
            RuntimeWarning,
            stacklevel=2,
        )


def set_tracer_provider(
    tracer_provider: trace_api.TracerProvider,
    *,
    metadata: dict[str, AttributeValue] | None = None,
    allow_pii: bool | None = None,
) -> None:
    """Select the provider used by framework spans without changing the OTel global.

    For child instrumentation with an explicit provider, call the standard host API as
    ``trace.get_tracer(name, tracer_provider=provider)``. Invocation parentage uses the
    current OTel context in either case. Application exporters keep their own policy;
    use :class:`RedactingSpanExporter` to filter one destination. ``allow_pii=False``
    reports the migration from the former provider-wide filtering mechanism.
    """
    if isinstance(tracer_provider, trace_sdk.TracerProvider):
        _install_metadata(tracer_provider, metadata)
    _install_pii_redaction(tracer_provider, allow_pii=allow_pii)
    tracing.tracer.set_provider(tracer_provider)


class _AuthHeaderProvider:
    """OTLP SDK adapter; credential construction and lifetime are runtime-owned."""

    def __init__(self) -> None:
        from ..job import _server_credentials  # local import: job.py imports this module

        self._uploader = _core._core_SessionReportUpload(*_server_credentials()[1:])

    def __call__(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._uploader.bearer()}"}


_TELEMETRY_SHUTDOWN_TIMEOUT = policy().shutdown_timeout_seconds


def _report_configuration_conflicts(messages: Sequence[str]) -> None:
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=3)


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
    Telemetry emitted outside a captured job invocation has no job attribution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # the exportable-jobs registry: job_id (globally unique) -> the job's
        # telemetry state, registered by configure() and removed after the final
        # flush in release(). The gated exporters hold a reference to this exact
        # dict, so it is mutated in place, never rebound.
        self._export_jobs: dict[str, _JobTelemetry] = {}
        self._atexit_registered = False
        self._closed = False
        self._session: _AuthRefreshingSession | None = None
        self._observability_url: str | None = None

        # everything the framework created, shut down once at process exit:
        # providers it built (constructed with shutdown_on_exit=False) and every
        # batch processor it attached — appended at each creation site. A
        # provider supplied by the integrator is never in this list.
        self._exit_targets: list[tuple[str, Callable[[], object]]] = []

        # traces
        self._trace_provider_attached: trace_sdk.TracerProvider | None = None
        self._owned_trace_provider: trace_sdk.TracerProvider | None = None
        self._span_metadata_processor: _MetadataSpanProcessor | None = None
        self._span_batch_processor: BoundedSpanProcessor | None = None
        # every span processor attached so far: a provider replaced mid-process keeps its
        # processor, since spans still open on it (the job's root) end there
        self._span_batch_processors: list[BoundedSpanProcessor] = []
        self._span_gate: _GatedSpanExporter | None = None
        self._span_pipelines: dict[
            trace_sdk.TracerProvider, tuple[BoundedSpanProcessor, _GatedSpanExporter]
        ] = {}

        # logs
        self._logger_provider: LoggerProvider | None = None
        self._log_metadata_processor: _MetadataLogProcessor | None = None
        self._log_batch_processor: BoundedLogRecordProcessor | None = None
        self._log_handler: _TraceLevelLoggingHandler | None = None

        # metrics
        self._owned_meter_provider: SdkMeterProvider | None = None
        self._meter_conflict_provider: metrics_api.MeterProvider | None = None

    @property
    def logger_provider(self) -> LoggerProvider | None:
        """The logger provider the framework exports through (ours or adopted)."""
        return self._logger_provider

    @property
    def log_handler(self) -> _TraceLevelLoggingHandler | None:
        """The framework's own root-logger OTLP handler, if logs were configured."""
        return self._log_handler

    def prepare(
        self, *, room_id: str, job_id: str, agent_name: str, observability_url: str
    ) -> None:
        """Have the trace pipeline up before the job's first span.

        Without a tracer provider every span is a non-recording stub: the job's root and
        everything before ``session.start()`` would be lost, and later spans would start
        their own traces. Creates the provider and the gated exporter but registers nothing:
        the job's spans are held until ``configure`` decides."""
        base_metadata: dict[str, AttributeValue] = {"room_id": room_id, "job_id": job_id}
        if agent_name:
            base_metadata[trace_types.ATTR_AGENT_NAME] = agent_name
        with self._lock:
            if self._observability_url is None:
                self._observability_url = observability_url
            if self._session is None:
                self._session = _AuthRefreshingSession(_AuthHeaderProvider())
            resource = Resource.create({SERVICE_NAME: "livekit-agents"})
            conflict = self._ensure_trace_pipeline(resource, self._observability_url)
            if self._span_gate is not None:
                self._span_gate.open_job(job_id)
            if not self._atexit_registered:
                self._atexit_registered = True
                atexit.register(self.shutdown_at_exit)
            selected_provider = tracing.tracer._tracer_provider

        _install_pii_redaction(selected_provider)
        _report_configuration_conflicts([conflict] if conflict else [])

    def discard(self, job_id: str) -> None:
        """A job ended without ever registering: drop whatever the gate held for it."""
        with self._lock:
            for _, gate in self._span_pipelines.values():
                gate.close_job(job_id)

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
        base_valid = attribute_bytes(base_metadata) <= policy().max_record_bytes
        if not base_valid:
            base_metadata = {}
        metadata_valid = base_valid and attribute_bytes(metadata) <= policy().max_record_bytes
        session_metadata = dict(base_metadata)
        if metadata_valid and metadata:
            session_metadata.update(metadata)
        if attribute_bytes(session_metadata) > policy().max_record_bytes:
            metadata_valid = False
            session_metadata = dict(base_metadata)
        if not metadata_valid:
            record("dropped")

        conflicts: list[str] = []
        with self._lock:
            if self._observability_url is None:
                self._observability_url = observability_url
            elif observability_url != self._observability_url:
                conflicts.append(
                    f"observability endpoint changed across jobs; keeping {self._observability_url}"
                )
            url = self._observability_url

            if self._session is None:
                self._session = _AuthRefreshingSession(_AuthHeaderProvider())

            resource = Resource.create({SERVICE_NAME: "livekit-agents", **base_metadata})

            if conflict := self._ensure_trace_pipeline(resource, url):
                conflicts.append(conflict)

            # Always set up the logger provider — it's needed for session reports,
            # evaluations, and chat history, not just Python log export.
            self._ensure_logger_provider()

            if enable_logs:
                self._ensure_log_pipeline(url)

            # The meter provider outlives the job. Keep its resource process-stable;
            # per-job identity belongs on traces, never on metric measurements.
            meter_resource = Resource.create({SERVICE_NAME: "livekit-agents"})
            if conflict := self._ensure_meter_provider(meter_resource, url):
                conflicts.append(conflict)

            if not self._atexit_registered:
                self._atexit_registered = True
                atexit.register(self.shutdown_at_exit)

            state = _JobTelemetry(
                attributes=session_metadata,
                traces_enabled=enable_traces and metadata_valid,
                logs_enabled=enable_logs and metadata_valid,
            )
            if metadata_valid:
                self._export_jobs[job_id] = state
            gates = [gate for _, gate in self._span_pipelines.values()]
            registered = job_id in self._export_jobs
            log_provider = self._logger_provider
            log_handler = self._log_handler
            selected_provider = tracing.tracer._tracer_provider
        _install_pii_redaction(selected_provider)
        _report_configuration_conflicts(conflicts)
        if enable_logs and log_provider is not None:
            if log_handler is None:
                created_handler = _TraceLevelLoggingHandler(
                    level=logging.NOTSET, logger_provider=log_provider
                )
                with self._lock:
                    if self._log_handler is None:
                        self._log_handler = created_handler
                    log_handler = self._log_handler
            logging.getLogger().addHandler(log_handler)
        for gate in gates:
            gate.job_registered(
                job_id,
                traces_enabled=enable_traces and registered,
                redacted=bool((metadata or {}).get(ATTRIBUTE_REDACTION_ENABLED, False)),
            )
        return state

    def _ensure_trace_pipeline(self, resource: Resource, url: str) -> str | None:
        provider = (
            tracing.tracer._tracer_provider
            if tracing.tracer._explicit_provider
            else trace_api.get_tracer_provider()
        )
        if isinstance(provider, trace_api.ProxyTracerProvider):
            if self._owned_trace_provider is None:
                self._owned_trace_provider = trace_sdk.TracerProvider(
                    resource=Resource.create({SERVICE_NAME: "livekit-agents"}),
                    shutdown_on_exit=False,
                )
                self._exit_targets.append(
                    (
                        f"tracer:{id(self._owned_trace_provider):x}",
                        self._owned_trace_provider.shutdown,
                    )
                )
            provider = self._owned_trace_provider
        tracing.tracer.set_provider(provider, explicit=tracing.tracer._explicit_provider)
        if not isinstance(provider, trace_sdk.TracerProvider):
            self._span_gate = None
            return None

        if provider in self._span_pipelines:
            self._span_batch_processor, self._span_gate = self._span_pipelines[provider]
            self._trace_provider_attached = provider
            return None

        if len(self._span_batch_processors) >= policy().max_providers:
            record("conflicts")
            return (
                "Cloud tracing is limited to four provider registrations per process; "
                "the newly selected application provider remains active."
            )
        assert self._session is not None
        span_exporter = OTLPSpanExporter(
            endpoint=f"{url}/observability/traces/otlp/v0",
            compression=Compression.Gzip,
            session=self._session,
        )
        self._span_gate = _GatedSpanExporter(span_exporter, self._export_jobs)
        self._span_metadata_processor = _install_metadata(provider)
        self._span_batch_processor = BoundedSpanProcessor(self._span_gate)
        self._span_batch_processors.append(self._span_batch_processor)
        self._span_gate._requeue = self._span_batch_processor.on_end
        self._span_pipelines[provider] = (self._span_batch_processor, self._span_gate)
        if provider is not self._owned_trace_provider:
            self._exit_targets.append(
                (f"spans:{id(self._span_batch_processor):x}", self._span_batch_processor.shutdown)
            )
        provider.add_span_processor(self._span_batch_processor)
        self._trace_provider_attached = provider
        return None

    def _ensure_logger_provider(self) -> None:
        if self._logger_provider is not None:
            return
        current = get_logger_provider()
        if isinstance(current, LoggerProvider):
            # an SDK provider the integrator set up — adopt it
            self._logger_provider = current
        else:
            owned = LoggerProvider(shutdown_on_exit=False)
            self._exit_targets.append((f"logger:{id(owned):x}", owned.shutdown))
            self._logger_provider = owned

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
        self._log_batch_processor = BoundedLogRecordProcessor(
            _GatedLogExporter(log_exporter, self._export_jobs)
        )
        self._exit_targets.append(
            (f"logs:{id(self._log_batch_processor):x}", self._log_batch_processor.shutdown)
        )
        self._logger_provider.add_log_record_processor(self._log_metadata_processor)
        self._logger_provider.add_log_record_processor(self._log_batch_processor)

    def _ensure_meter_provider(self, resource: Resource, url: str) -> str | None:
        from .metrics import _adopt_meter_provider

        current = metrics_api.get_meter_provider()
        if not isinstance(current, _ProxyMeterProvider):
            _adopt_meter_provider(current)
            if current is self._meter_conflict_provider:
                return None
            self._meter_conflict_provider = current
            return (
                "Cloud metrics require a reader registered at application MeterProvider setup; "
                "the existing provider and its readers were preserved."
            )
        if self._owned_meter_provider is not None:
            _adopt_meter_provider(self._owned_meter_provider)
            return None
        assert self._session is not None
        temporality: dict[type[object], AggregationTemporality] = {
            SdkCounter: AggregationTemporality.DELTA,
            SdkUpDownCounter: AggregationTemporality.DELTA,
            SdkHistogram: AggregationTemporality.DELTA,
            SdkObservableCounter: AggregationTemporality.DELTA,
            SdkObservableUpDownCounter: AggregationTemporality.DELTA,
            SdkObservableGauge: AggregationTemporality.DELTA,
        }
        metric_exporter = GuardedMetricExporter(
            OTLPMetricExporter(
                endpoint=f"{url}/observability/metrics/otlp/v0",
                compression=Compression.Gzip,
                session=self._session,
                preferred_temporality=temporality,
            ),
            preferred_temporality=temporality,
            enabled=lambda: not destination.upload_gate.disabled,
        )
        reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
        provider = SdkMeterProvider(
            resource=resource, metric_readers=[reader], shutdown_on_exit=False
        )
        self._exit_targets.append((f"meter:{id(provider):x}", provider.shutdown))
        self._owned_meter_provider = provider
        _adopt_meter_provider(provider)
        return None

    def release(self, job_id: str, timeout: float = _TELEMETRY_SHUTDOWN_TIMEOUT) -> None:
        """Per-job teardown: flush the framework's exporters (bounded) while the
        job is still registered, then remove it from the exportable-jobs
        registry — records stamped with its id no longer upload. When no job
        remains, the framework log handler is detached.

        Nothing is shut down here: providers (ours or the integrator's) and the
        batch processors keep running for the next job in the process. Final
        shutdown happens once, at process exit (:meth:`shutdown_at_exit`).
        """
        with self._lock:
            if job_id not in self._export_jobs:
                return  # never configured, or already released

            flush_targets: list[tuple[str, Callable[[], object]]] = []
            for processor in self._span_batch_processors:
                flush_targets.append((f"spans:{id(processor):x}", processor.force_flush))
            if self._log_batch_processor is not None:
                flush_targets.append(
                    (
                        f"logs:{id(self._log_batch_processor):x}",
                        self._log_batch_processor.force_flush,
                    )
                )
            if self._owned_meter_provider is not None:
                flush_targets.append(
                    (
                        f"meter:{id(self._owned_meter_provider):x}",
                        self._owned_meter_provider.force_flush,
                    )
                )

        # flush before unregistering, so the job's remaining telemetry is exported
        run_bounded("flush", flush_targets, timeout)

        with self._lock:
            self._export_jobs.pop(job_id, None)
            for _, gate in self._span_pipelines.values():
                gate.close_job(job_id)
            if self._export_jobs:
                # another job is still running in this process; keep exporting
                return
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
            if self._closed:
                return
            self._closed = True
            targets = list(self._exit_targets)
            self._exit_targets.clear()
            handler = self._log_handler

        if handler is not None:
            logging.getLogger().removeHandler(handler)

        run_bounded("shutdown", targets, timeout)


_cloud = _CloudTelemetry()


def _cloud_log_handler() -> _TraceLevelLoggingHandler | None:
    """The framework's own root-logger OTLP handler, if configured for this job."""
    return _cloud.log_handler


def _prepare_cloud_tracer(
    *, room_id: str, job_id: str, agent_name: str = "", observability_url: str
) -> None:
    _cloud.prepare(
        room_id=room_id, job_id=job_id, agent_name=agent_name, observability_url=observability_url
    )


def _discard_cloud_tracer(job_id: str) -> None:
    _cloud.discard(job_id)


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
    destination.upload_gate.reset()
    return _cloud.configure(
        room_id=room_id,
        job_id=job_id,
        agent_name=agent_name,
        observability_url=observability_url,
        enable_traces=enable_traces,
        enable_logs=enable_logs,
        metadata=metadata,
    )


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
