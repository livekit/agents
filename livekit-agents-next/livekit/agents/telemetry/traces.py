"""The agent's traces: the tracer and its exporters, PII redaction, the GenAI attributes, the
session's trace context and RPC spans.

The job's trace is rooted at ``job_entrypoint`` and ``agent_session`` is a child of it. Code
running on a task whose context predates the session (the event loop monitor's heartbeat above
all) resolves the running session through the job (:func:`session_root_context`) so its spans
nest under ``agent_session`` while one exists. The session's own spans are the core's records,
opened and ended by ``VoiceSession``.

RPCs go through the room SDK's ``RpcInterceptor`` hook (``livekit-rtc`` >= 1.1.18): one
interceptor per local participant turns each call into a span following the OpenTelemetry RPC
conventions, ``rpc_call`` (``SpanKind.CLIENT``) under the current span for outgoing calls and
``rpc_handler`` (``SpanKind.SERVER``) under the session's root span for incoming ones. Payloads
are recorded truncated under ``lk.pii`` keys; participant identities are recorded as is. On an
SDK without the hook, ``install`` is a no-op.
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import logging
import os
import threading
import warnings
import weakref
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence, Set
from copy import copy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import aiohttp
import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry import context as otel_context, metrics as metrics_api, trace, trace as trace_api
from opentelemetry._logs import LogRecord as OTelLogRecord, get_logger_provider
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
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
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, Span, Status, StatusCode, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AnyValue, AttributeValue

from livekit import rtc

from .. import _core
from .._proto import encode_chat_item
from ..log import TRACE_LEVEL, logger
from ..types import ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED
from ..utils import is_given
from . import trace_types, utils as telemetry_utils
from ._export import (
    BoundedLogRecordProcessor,
    BoundedSpanProcessor,
    GuardedMetricExporter,
    attribute_bytes,
    exporting,
    run_bounded,
    span_key,
    worker,
)
from .utils import REDACTED_EXCEPTION_MESSAGE

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Event

    from .._core import Tagger
    from ..job import JobContext, SessionReport
    from ..llm import ChatItem
    from ..voice.agent_session import VoiceSession

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


def _describe_option_object(obj: object) -> str:
    """Render an object from the session options as ``module.Class`` or, when it implements
    :class:`DescribesOptions`, ``module.Class(k=v, ...)``.

    The OTel log exporter stringifies anything that is not a primitive, which for these
    objects yields the default ``<... object at 0x...>`` repr. The class alone is stable and
    safe; the object itself decides what else is worth showing."""
    cls = type(obj)
    name = f"{cls.__module__}.{cls.__name__}"
    describe = getattr(obj, "describe_options", None)
    if not callable(describe):
        return name
    try:
        options = describe()
    except Exception:
        logger.debug("describe_options() failed on %s", name, exc_info=True)
        return name
    parts: list[str] = []
    for key, value in options.items():
        if value is None or not is_given(value):
            continue
        rendered = (
            str(value)
            if isinstance(value, _OPTION_PRIMITIVES)
            else json.dumps(_serialize_option_value(value), sort_keys=True, default=str)
        )
        parts.append(f"{key}={rendered}")
    return f"{name}({', '.join(parts)})"


def _serialize_option_value(value: Any) -> Any:
    if value is None or isinstance(value, _OPTION_PRIMITIVES):
        return value
    if isinstance(value, Mapping):
        return {
            _SESSION_OPTION_KEY_ALIASES.get(k, k): _serialize_option_value(v)
            for k, v in value.items()
            if k not in _SESSION_OPTION_OMITTED_KEYS
        }
    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes)):
        # any Sequence is a valid option value (tts_text_transforms accepts one), so
        # serialize the elements rather than collapsing the container to its class name
        items = sorted(value, key=str) if isinstance(value, Set) else value
        return [_serialize_option_value(v) for v in items]
    return _describe_option_object(value)


def _serialize_session_options(options: dict[str, Any]) -> dict[str, Any]:
    serialized = _serialize_option_value(options)
    assert isinstance(serialized, dict)
    return serialized


_USE_SPAN_SIGNATURE = inspect.signature(trace_api.use_span)


_START_SPAN_SIGNATURE = inspect.signature(Tracer.start_span)


_START_AS_CURRENT_SPAN_SIGNATURE = inspect.signature(Tracer.start_as_current_span)


def _with_conversation_id(
    attributes: Mapping[str, AttributeValue] | None,
) -> Mapping[str, AttributeValue] | None:
    captured = {
        "lk.telemetry.capture_content": capture_content_enabled(),
        **(attributes or {}),
    }
    if (conversation_id := _conversation_id()) is None:
        return captured
    return {trace_types.ATTR_GEN_AI_CONVERSATION_ID: conversation_id, **captured}


class _DynamicTracer(Tracer):
    def __init__(self, instrumenting_module_name: str) -> None:
        self._instrumenting_module_name = instrumenting_module_name
        self._explicit_provider = False
        self._tracer_provider: trace_api.TracerProvider = trace_api.get_tracer_provider()
        self._tracer = trace_api.get_tracer(instrumenting_module_name)

    def set_provider(
        self, tracer_provider: trace_api.TracerProvider, *, explicit: bool = True
    ) -> None:
        self._explicit_provider = explicit
        self._tracer_provider = tracer_provider
        self._tracer = trace_api.get_tracer(
            self._instrumenting_module_name,
            tracer_provider=self._tracer_provider,
        )

    def start_span(self, *args: Any, **kwargs: Any) -> Span:
        bound = _START_SPAN_SIGNATURE.bind(self._tracer, *args, **kwargs)
        bound.arguments["attributes"] = _with_conversation_id(bound.arguments.get("attributes"))
        return self._tracer.start_span(*bound.args[1:], **bound.kwargs)

    @_agnosticcontextmanager
    def use_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        bound = _USE_SPAN_SIGNATURE.bind(*args, **kwargs)
        record_exception = bound.arguments.get("record_exception", True)
        set_status_on_exception = bound.arguments.get("set_status_on_exception", True)
        bound.arguments.update(record_exception=False, set_status_on_exception=False)
        with trace_api.use_span(*bound.args, **bound.kwargs) as span:
            try:
                yield span
            except Exception as exc:
                telemetry_utils.record_exception(
                    span, exc, record_event=record_exception, set_status=set_status_on_exception
                )
                raise

    @_agnosticcontextmanager
    def detached_span(
        self,
        name: str,
        *,
        context: otel_context.Context | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """A span that is never made current.

        For code that spawns long-lived tasks (connecting the room, publishing a track,
        connecting MCP servers): a *current* span is inherited by every task created under
        it and becomes the accidental parent of unrelated spans those tasks emit for the rest
        of the session. The parent is ``context`` when given, else the ambient context; the
        exception, if any, is recorded redaction-aware and the span is ended."""
        span = self.start_span(name, context=context, attributes=attributes)
        try:
            yield span
        except Exception as e:
            telemetry_utils.record_exception(span, e)
            raise
        finally:
            span.end()

    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        bound = _START_AS_CURRENT_SPAN_SIGNATURE.bind(self._tracer, *args, **kwargs)
        bound.arguments["attributes"] = _with_conversation_id(bound.arguments.get("attributes"))
        record_exception = bound.arguments.get("record_exception", True)
        set_status_on_exception = bound.arguments.get("set_status_on_exception", True)
        bound.arguments.update(record_exception=False, set_status_on_exception=False)
        with self._tracer.start_as_current_span(*bound.args[1:], **bound.kwargs) as span:
            try:
                yield span
            except Exception as exc:
                telemetry_utils.record_exception(
                    span, exc, record_event=record_exception, set_status=set_status_on_exception
                )
                raise


tracer: _DynamicTracer = _DynamicTracer("livekit-agents")


class _UploadGate:
    """Process-wide gate that stops observability uploads once LiveKit Cloud reports data
    recording is disabled for the project (``_core.UploadGate``). Reset per session from
    JobContext.init_recording()."""

    def __init__(self) -> None:
        self._core = _core.UploadGate()

    def reset(self) -> None:
        self._core.reset()

    @property
    def disabled(self) -> bool:
        return self._core.disabled

    def disable(self) -> None:
        if self._core.disable():
            logger.warning(
                "LiveKit Cloud data recording is disabled for this project; "
                "skipping telemetry and recording uploads for this session"
            )

    is_disabled_response = staticmethod(_core.UploadGate.is_disabled_response)


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
    attributes: Mapping[str, AttributeValue]
    traces_enabled: bool
    logs_enabled: bool

    def __post_init__(self) -> None:
        snapshot = {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in self.attributes.items()
        }
        object.__setattr__(self, "attributes", MappingProxyType(snapshot))


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
                "lk.telemetry.capture_content", capture_content_enabled()
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
                "lk.telemetry.capture_content", capture_content_enabled()
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


_MAX_PENDING_SPANS_PER_JOB = 64


class _GatedSpanExporter(SpanExporter):
    """Wraps the OTLP span exporter so only spans of registered, trace-enabled
    jobs are uploaded (see ``_job_export_state``).

    A job decides whether it records in ``session.start()``, but its first spans
    (``job_entrypoint``, ``room_connect``, a stall while models load) end before
    that. Spans of a job that has started but not decided yet are held here,
    oldest first up to a bound, and flushed or dropped with the decision; a job
    that ends without deciding drops them at cleanup."""

    def __init__(
        self,
        inner: SpanExporter,
        export_jobs: Mapping[str, _JobTelemetry],
        *,
        requeue: Callable[[ReadableSpan], None] | None = None,
    ) -> None:
        self._inner = inner
        self._export_jobs = export_jobs
        # hands a held span back to the batch processor so it is uploaded from the export
        # thread; job_registered runs on the event loop and must not call the exporter
        self._requeue = requeue
        self._lock = threading.Lock()
        self._open_jobs: set[str] = set()
        self._pending: dict[str, list[ReadableSpan]] = {}

    def open_job(self, job_id: str) -> None:
        """The job started; hold its spans until it registers or closes."""
        with self._lock:
            self._open_jobs.add(job_id)

    def job_registered(self, job_id: str, *, traces_enabled: bool, redacted: bool = False) -> None:
        """The job decided: upload what was held if it records traces, else drop it.

        Runs on the event loop (``init_recording``), so held spans go back on the batch
        processor's queue instead of being exported here: the OTLP exporter blocks on the
        network. ``redacted`` strips PII from them first: they ended before the job's
        redaction was known, so the PII processor let it through."""
        with self._lock:
            self._open_jobs.discard(job_id)
            held = self._pending.pop(job_id, [])
        if not traces_enabled:
            for span in held:
                worker().release(span_key(span))
            return
        if not held:
            return
        if redacted:
            held = [redact(s) for s in held]
        if self._requeue is not None:
            for s in held:
                self._requeue(s)
        else:
            worker().record_drop("capacity", len(held))
            for span in held:
                worker().release(span_key(span))

    def close_job(self, job_id: str) -> None:
        """The job ended; anything still held was never meant to upload."""
        with self._lock:
            self._open_jobs.discard(job_id)
            held = self._pending.pop(job_id, [])
        for span in held:
            worker().release(span_key(span))

    def holds(self, span: ReadableSpan) -> bool:
        job_id = (span.attributes or {}).get("job_id")
        with self._lock:
            return isinstance(job_id, str) and span in self._pending.get(job_id, ())

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        exportable: list[ReadableSpan] = []
        with self._lock:
            for s in spans:
                state = _job_export_state(self._export_jobs, s.attributes)
                if state is not None:
                    if state.traces_enabled:
                        # Cloud gets its own snapshot; application destinations are untouched.
                        exportable.append(
                            _span_snapshot(
                                s,
                                redacted=(
                                    _redact_content(
                                        s.attributes,
                                        allow_pii=True,
                                        project_redaction=bool(
                                            state.attributes.get(ATTRIBUTE_REDACTION_ENABLED)
                                        ),
                                    )
                                ),
                            )
                        )
                    continue
                job_id = (s.attributes or {}).get("job_id")
                if isinstance(job_id, str) and job_id in self._open_jobs:
                    held = self._pending.setdefault(job_id, [])
                    if len(held) < _MAX_PENDING_SPANS_PER_JOB:
                        held.append(s)
                    else:
                        worker().record_drop("capacity", 1)
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
        exportable: list[ReadableLogRecord] = []
        for record in batch:
            attrs = record.log_record.attributes
            state = _job_export_state(self._export_jobs, attrs)
            if state is None or not state.logs_enabled:
                continue
            exportable.append(
                _log_snapshot(
                    record,
                    allow_pii=True,
                    project_redaction=bool(state.attributes.get(ATTRIBUTE_REDACTION_ENABLED)),
                )
            )
        if not exportable:
            return LogRecordExportResult.SUCCESS
        return self._inner.export(exportable)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        # LogRecordExporter.shutdown is untyped in the OTel SDK
        cast(Callable[[], None], self._inner.shutdown)()


_LOG_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class _BufferingHandler(logging.Handler):
    """Retain early logs only within native telemetry admission."""

    def __init__(self) -> None:
        super().__init__()
        self._records: deque[tuple[str, logging.LogRecord]] = deque()
        self._keys: set[str] = set()
        self._sequence = 0
        self._finalizer: weakref.finalize | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if exporting():
            return
        runtime = worker()
        if attribute_bytes(record.__dict__) > 64 * 1024:
            runtime.record_drop("oversize", 1)
            return
        record = copy(record)
        captured = capture_content_enabled()
        identity = _job_stamp_attributes() or {}
        if not captured or telemetry_utils.redaction_enabled(identity):
            extras = filter_attributes(
                {
                    key: value
                    for key, value in record.__dict__.items()
                    if key not in _LOG_RECORD_FIELDS and key != "message"
                }
            )
            for key in record.__dict__.keys() - _LOG_RECORD_FIELDS:
                del record.__dict__[key]
            record.__dict__.update(extras)
            record.msg = "log body redacted"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            identity = filter_attributes(identity)
        record.__dict__.update(identity)
        record.__dict__["lk.telemetry.capture_content"] = captured
        charge = attribute_bytes(record.__dict__)
        self._sequence += 1
        key = f"prelog:{id(self):x}:{self._sequence}"
        if runtime.reserve(key, charge):
            self._keys.add(key)
            self._records.append((key, record))
            if self._finalizer is None:
                self._finalizer = weakref.finalize(self, _release_early_logs, runtime, self._keys)

    @property
    def buffer(self) -> Iterator[logging.LogRecord]:
        while self._records:
            key, record = self._records.popleft()
            try:
                yield record
            finally:
                worker().release(key)
                self._keys.discard(key)

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self._records.clear()
        super().close()


def _release_early_logs(runtime: _core._core_TelemetryWorker, keys: set[str]) -> None:
    for key in keys:
        runtime.release(key)
    keys.clear()


class _TraceLevelLoggingHandler(LoggingHandler):
    """Custom LoggingHandler that properly maps TRACE_LEVEL to OTel TRACE severity.

    The default OTel LoggingHandler maps any log level < 10 to UNSPECIFIED,
    but we want TRACE_LEVEL (5) to map to TRACE for proper severity in exports.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not exporting():
            super().emit(record)

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
            log_record.attributes = filter_attributes(attributes)

        # OTel's std_to_otel returns UNSPECIFIED for levels < 10
        # Map our TRACE_LEVEL to OTel's TRACE
        if record.levelno == TRACE_LEVEL:
            log_record.severity_number = SeverityNumber.TRACE
        return log_record


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
    tracer.set_provider(tracer_provider)


class _AuthHeaderProvider:
    """OTLP SDK adapter; credential construction and lifetime are runtime-owned."""

    def __init__(self) -> None:
        self._uploader = _core._core_SessionReportUpload(
            os.environ.get("LIVEKIT_API_KEY", ""), os.environ.get("LIVEKIT_API_SECRET", "")
        )

    def __call__(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._uploader.bearer()}"}


_TELEMETRY_SHUTDOWN_TIMEOUT = 10.0


_run_bounded = run_bounded


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
        self._admitted_jobs: set[str] = set()
        self._session: _AuthRefreshingSession | None = None
        self._observability_url: str | None = None

        # everything the framework created, shut down once at process exit:
        # providers it built (constructed with shutdown_on_exit=False) and every
        # batch processor it attached — appended at each creation site. A
        # provider supplied by the integrator is never in this list.
        self._exit_targets: list[tuple[str, Callable[[], Any]]] = []

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
            if self._admit_job(job_id) and self._span_gate is not None:
                self._span_gate.open_job(job_id)
            if not self._atexit_registered:
                self._atexit_registered = True
                atexit.register(self.shutdown_at_exit)
            selected_provider = tracer._tracer_provider

        _install_pii_redaction(selected_provider)
        _report_configuration_conflicts([conflict] if conflict else [])

    def _admit_job(self, job_id: str) -> bool:
        if job_id in self._admitted_jobs:
            return True
        if worker().reserve(f"job:{job_id}", 64 * 1024):
            self._admitted_jobs.add(job_id)
            return True
        return False

    def discard(self, job_id: str) -> None:
        """A job ended without ever registering: drop whatever the gate held for it."""
        with self._lock:
            for _, gate in self._span_pipelines.values():
                gate.close_job(job_id)
            self._admitted_jobs.discard(job_id)
        worker().release(f"job:{job_id}")

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
        base_valid = attribute_bytes(base_metadata) <= 64 * 1024
        if not base_valid:
            base_metadata = {}
        metadata_valid = base_valid and attribute_bytes(metadata) <= 64 * 1024
        session_metadata = dict(base_metadata)
        if metadata_valid and metadata:
            session_metadata.update(metadata)
        if attribute_bytes(session_metadata) > 64 * 1024:
            metadata_valid = False
            session_metadata = dict(base_metadata)
        if not metadata_valid:
            worker().record_drop("oversize", 1)

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
            if metadata_valid and self._admit_job(job_id):
                self._export_jobs[job_id] = state
            gates = [gate for _, gate in self._span_pipelines.values()]
            registered = job_id in self._export_jobs
            log_provider = self._logger_provider
            log_handler = self._log_handler
            selected_provider = tracer._tracer_provider
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
            tracer._tracer_provider
            if tracer._explicit_provider
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
        tracer.set_provider(provider, explicit=tracer._explicit_provider)
        if not isinstance(provider, trace_sdk.TracerProvider):
            self._span_gate = None
            return None

        if provider in self._span_pipelines:
            self._span_batch_processor, self._span_gate = self._span_pipelines[provider]
            self._trace_provider_attached = provider
            return None

        if len(self._span_batch_processors) >= 4:
            worker().record_drop("conflict", 1)
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
        self._span_batch_processor = BoundedSpanProcessor(
            self._span_gate, holds=self._span_gate.holds
        )
        self._span_batch_processors.append(self._span_batch_processor)
        self._span_gate._requeue = self._span_batch_processor.requeue
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

            flush_targets: list[tuple[str, Callable[[], Any]]] = []
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
        _run_bounded("flush", flush_targets, timeout)

        with self._lock:
            self._export_jobs.pop(job_id, None)
            self._admitted_jobs.discard(job_id)
            worker().release(f"job:{job_id}")
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

        _run_bounded("shutdown", targets, timeout)


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
    from .._core_view import from_core_item, to_core_items

    metadata = dict(metadata or {})
    emission_context = otel_context.get_current()
    uploader = _core._core_SessionReportUpload(
        os.environ.get("LIVEKIT_API_KEY", ""), os.environ.get("LIVEKIT_API_SECRET", "")
    )
    report_options = report.options
    options = (
        _core.RecordingOptions(**report_options["recording_options"])
        if report_options is not None
        else _core.RecordingOptions()
    )
    capture_content = capture_content_enabled()
    redacted = options.redaction or telemetry_utils.redaction_enabled(metadata)
    transcript_allowed, audio_allowed = uploader.content_policy(
        options.transcript, options.audio, capture_content, redacted
    )
    report_content_allowed, _ = uploader.content_policy(True, False, capture_content, redacted)
    prepared: list[list[_core.ChatItem]] = []
    failures: list[Exception] = []

    def _prepare() -> None:
        try:
            native_items = to_core_items(report.chat_history.items) if transcript_allowed else []
            records = _core.session_records(
                _upload_gate._core,
                {
                    "agent_name": agent_name,
                    "job_id": report.job_id,
                    "room_id": report.room_id,
                    "room": report.room,
                    "sdk_version": report.sdk_version,
                    "started_at": report.started_at,
                    "timestamp": report.timestamp,
                    "options": (
                        _serialize_session_options(report_options)
                        if report_content_allowed and report_options is not None
                        else {}
                    ),
                    "recording_options": _core.RecordingOptions(
                        audio=audio_allowed,
                        traces=options.traces,
                        logs=options.logs,
                        transcript=transcript_allowed,
                        redaction=redacted,
                    ),
                    "model_usage": [u.model_dump() for u in report.model_usage or []],
                    "chat_history": native_items,
                },
                tagger,
                lambda item: _chat_item_span_attribute(from_core_item(item)),
                capture_content=capture_content,
            )
            provider = _cloud.logger_provider or get_logger_provider()
            loggers: dict[str, Any] = {}
            for record in records:
                if (otel_logger := loggers.get(record["logger"])) is None:
                    otel_logger = loggers[record["logger"]] = provider.get_logger(
                        name=record["logger"],
                        attributes={
                            "room_id": report.room_id,
                            "job_id": report.job_id,
                            "room": report.room,
                            **metadata,
                        },
                    )
                otel_logger.emit(
                    body=record["body"],
                    timestamp=record["timestamp_nanos"],
                    context=emission_context,
                    attributes={**metadata, **record["attributes"]},
                    severity_number=SeverityNumber(record["severity_number"]),
                    severity_text=record["severity_text"],
                )
            prepared.append(native_items)
        except Exception as error:
            failures.append(error)

    worker = _core._core_TelemetryWorker()
    key = f"session-report/{report.job_id}"
    if not worker.reserve(key, 4 * 1024 * 1024):
        return
    try:
        if not worker.submit(key, _prepare, 4 * 1024 * 1024):
            return
        if not await worker.wait_async(key, 900.0):
            return
        if failures:
            raise failures[0]
        transcript = await uploader.transcript(prepared.pop()) if transcript_allowed else None
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
        _upload_gate.disabled,
    )
    if not enabled:
        _upload_gate.disable()


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


_TRUTHY = ("1", "true", "yes", "on")


# the env var name the GenAI conventions standardise for this opt-in
_capture_content: bool = (
    os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").strip().lower()
    in _TRUTHY
)


def set_capture_content(enabled: bool) -> None:
    """Whether spans carry message content (judge explanations); off by default unless
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is set."""
    global _capture_content
    _capture_content = enabled


def capture_content_enabled() -> bool:
    return _capture_content


def _conversation_id() -> str | None:
    """The convention forbids fabricating one (no UUIDs, trace ids or content hashes),
    so this is the room sid LiveKit already stamps on every record."""
    from ..job import get_job_context

    ctx = get_job_context(required=False)
    if ctx is None:
        return None
    return ctx.job.room.sid or None


def set_error_type(span: trace.Span, error: BaseException | str) -> None:
    """A low-cardinality identifier, never the error message: that is free-form and may
    carry user data."""
    if not span.is_recording():
        return
    if isinstance(error, str):
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, error)
        return

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        span.set_attribute(trace_types.ATTR_ERROR_TYPE, str(status_code))
        return
    span.set_attribute(trace_types.ATTR_ERROR_TYPE, type(error).__qualname__)


# exception details are recorded by `record_exception`, which resolves redaction from the
# ambient job context; that can disagree with the span's own stamp, so they are filtered
# here as well
_REDACTED_EXCEPTION_ATTRIBUTES = (
    trace_types.ATTR_EXCEPTION_MESSAGE,
    trace_types.ATTR_EXCEPTION_TRACE,
)


GEN_AI_PII_ATTRIBUTES: frozenset[str] = frozenset(
    {
        # flagged "likely to contain sensitive information including user/PII data"
        # by the GenAI semantic conventions
        trace_types.ATTR_GEN_AI_INPUT_MESSAGES,
        trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES,
        trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
        trace_types.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
        trace_types.ATTR_GEN_AI_TOOL_CALL_RESULT,
        trace_types.ATTR_GEN_AI_TOOL_DESCRIPTION,
        trace_types.ATTR_GEN_AI_TOOL_DEFINITIONS,
        # free-form text the caller supplied or the model produced
        trace_types.ATTR_GEN_AI_RETRIEVAL_QUERY_TEXT,
        trace_types.ATTR_GEN_AI_RETRIEVAL_DOCUMENTS,
        trace_types.ATTR_GEN_AI_MEMORY_QUERY_TEXT,
        trace_types.ATTR_GEN_AI_MEMORY_RECORDS,
        trace_types.ATTR_GEN_AI_EVALUATION_EXPLANATION,
    }
)


"""GenAI attributes carrying content. Fixed names, so they cannot be ``lk.pii.``-marked."""


PII_EVENT_NAMES: frozenset[str] = frozenset(
    {
        # the GenAI content events carry the message body in a generic `content`
        # attribute, so the whole event is dropped rather than filtered
        trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE,
        trace_types.EVENT_GEN_AI_USER_MESSAGE,
        trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
        trace_types.EVENT_GEN_AI_TOOL_MESSAGE,
        trace_types.EVENT_GEN_AI_CHOICE,
        trace_types.EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS,
    }
)


def is_pii_attribute(key: str) -> bool:
    if "pii" in key.split("."):
        return True
    if key in GEN_AI_PII_ATTRIBUTES:
        return True
    # gen_ai.prompt.variable.<key> holds the values interpolated into a prompt template
    return key.startswith(trace_types.ATTR_GEN_AI_PROMPT_VARIABLE)


def filter_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return _attribute_snapshot(attributes, redacted=True)


def _attribute_snapshot(attributes: Mapping[str, Any] | None, *, redacted: bool) -> dict[str, Any]:
    filtered = _filter_attribute_value(attributes or {}, depth=0, redacted=redacted)
    assert isinstance(filtered, dict)
    return filtered


def _filter_attribute_value(value: object, *, depth: int, redacted: bool) -> AnyValue:
    if depth > 4:
        return None
    if isinstance(value, Mapping):
        filtered: dict[str, AnyValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or (redacted and is_pii_attribute(key)):
                continue
            if redacted and key == trace_types.ATTR_EXCEPTION_TRACE:
                continue
            if redacted and key == trace_types.ATTR_EXCEPTION_MESSAGE:
                filtered[key] = REDACTED_EXCEPTION_MESSAGE
            else:
                filtered[key] = _filter_attribute_value(item, depth=depth + 1, redacted=redacted)
        return filtered
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_filter_attribute_value(item, depth=depth + 1, redacted=redacted) for item in value]
    return value if value is None or isinstance(value, (str, bool, int, float, bytes)) else None


def _contains_pii(attributes: Mapping[str, Any] | None) -> bool:
    return any(is_pii_attribute(k) or k in _REDACTED_EXCEPTION_ATTRIBUTES for k in attributes or ())


def _redact_content(
    attributes: Mapping[str, AttributeValue] | None,
    *,
    allow_pii: bool,
    project_redaction: bool = False,
) -> bool:
    attributes = attributes or {}
    return (
        not allow_pii
        or attributes.get("lk.telemetry.capture_content") is not True
        or project_redaction
        or bool(attributes.get(ATTRIBUTE_REDACTION_ENABLED))
    )


def _log_snapshot(
    record: ReadableLogRecord, *, allow_pii: bool, project_redaction: bool = False
) -> ReadableLogRecord:
    data = copy(record.log_record)
    redacted = _redact_content(
        data.attributes, allow_pii=allow_pii, project_redaction=project_redaction
    )
    safe_body = (data.attributes or {}).get("lk.telemetry.safe_body") is True
    attributes = _attribute_snapshot(data.attributes, redacted=redacted)
    attributes.pop("lk.telemetry.export_key", None)
    attributes.pop("lk.telemetry.safe_body", None)
    data.attributes = attributes
    data.body = (
        "log body redacted"
        if redacted and not safe_body
        else _filter_attribute_value(data.body, depth=0, redacted=False)
    )
    return replace(
        record,
        log_record=data,
        resource=_resource_snapshot(record.resource, redacted=redacted),
        instrumentation_scope=_scope_snapshot(record.instrumentation_scope, redacted=redacted),
    )


class RedactingLogExporter(LogRecordExporter):
    """Apply capture and redaction policy to one application's log destination."""

    def __init__(self, inner: LogRecordExporter, *, allow_pii: bool = False) -> None:
        self._inner = inner
        self._allow_pii = allow_pii

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        return self._inner.export(
            [_log_snapshot(record, allow_pii=self._allow_pii) for record in batch]
        )

    def shutdown(self) -> None:
        cast(Callable[[], None], self._inner.shutdown)()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _resource_snapshot(resource: Resource, *, redacted: bool) -> Resource:
    attributes = filter_attributes(resource.attributes) if redacted else dict(resource.attributes)
    return Resource(attributes, schema_url=resource.schema_url)


def _scope_snapshot(
    scope: InstrumentationScope | None, *, redacted: bool
) -> InstrumentationScope | None:
    if scope is None:
        return None
    attributes = _attribute_snapshot(scope.attributes, redacted=redacted)
    return InstrumentationScope(
        scope.name, version=scope.version, schema_url=scope.schema_url, attributes=attributes
    )


def _span_snapshot(span: ReadableSpan, *, redacted: bool) -> ReadableSpan:
    events = _filter_events(span.events) if redacted else None
    status = span.status
    if redacted and status.status_code is StatusCode.ERROR and status.description:
        status = Status(StatusCode.ERROR, REDACTED_EXCEPTION_MESSAGE)
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=_resource_snapshot(span.resource, redacted=redacted),
        attributes=_attribute_snapshot(span.attributes, redacted=redacted),
        events=tuple(events if events is not None else span.events),
        links=tuple(
            Link(
                link.context,
                attributes=_attribute_snapshot(link.attributes, redacted=redacted),
            )
            for link in span.links
        ),
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=_scope_snapshot(span.instrumentation_scope, redacted=redacted),
    )


def redact(span: ReadableSpan) -> ReadableSpan:
    """Return an independent destination snapshot with content and exception details removed."""
    return _span_snapshot(span, redacted=True)


class RedactingSpanExporter(SpanExporter):
    """Apply capture and redaction policy to this destination's independent span snapshots.

    Register this around an application exporter before attaching its processor. Explicit
    ``allow_pii=True`` also requires content capture opt-in; project redaction always wins.
    Select the provider with ``set_tracer_provider`` before spans start to capture consent.
    A span without captured consent is redacted. Other exporters retain their own policy.
    """

    def __init__(self, inner: SpanExporter, *, allow_pii: bool = False) -> None:
        self._inner = inner
        self._allow_pii = allow_pii

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        snapshots = [
            _span_snapshot(
                span,
                redacted=(_redact_content(span.attributes, allow_pii=self._allow_pii)),
            )
            for span in spans
        ]
        return self._inner.export(snapshots)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _filter_events(events: Sequence[Event]) -> list[Event] | None:
    """``None`` when nothing changed."""
    from opentelemetry.sdk.trace import Event as SdkEvent

    changed = False
    kept: list[Event] = []
    for event in events:
        if event.name in PII_EVENT_NAMES:
            changed = True
            continue
        if _contains_pii(event.attributes):
            changed = True
            kept.append(
                SdkEvent(
                    name=event.name,
                    attributes=filter_attributes(event.attributes),
                    timestamp=event.timestamp,
                )
            )
            continue
        kept.append(event)

    return kept if changed else None


def _current_job(job_ctx: JobContext | None) -> JobContext | None:
    if job_ctx is not None:
        return job_ctx
    from ..job import get_job_context

    return get_job_context(required=False)


def primary_session(job_ctx: JobContext | None = None) -> VoiceSession | None:
    job_ctx = _current_job(job_ctx)
    return job_ctx._primary_agent_session if job_ctx is not None else None


def session_root_context(job_ctx: JobContext | None = None) -> otel_context.Context | None:
    """The primary agent session's root span context, if a session is running."""
    session = primary_session(job_ctx)
    if session is not None and session._root_span_context is not None:
        return cast(otel_context.Context, session._root_span_context)
    return None


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
