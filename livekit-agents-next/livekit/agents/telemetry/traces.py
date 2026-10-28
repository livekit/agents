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
import random
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import aiofiles
import aiohttp
import requests
from google.protobuf.json_format import MessageToDict
from opentelemetry import context as otel_context, metrics as metrics_api, trace, trace as trace_api
from opentelemetry._logs import LogRecord as OTelLogRecord, get_logger_provider, set_logger_provider
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import NoOpMeterProvider
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
from opentelemetry.sdk.trace import ReadableSpan, Span as SdkSpan, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import AttributeValue

from livekit import api, rtc
from livekit.protocol import metrics as proto_metrics

from .. import _core
from .._proto import encode_chat_item
from ..log import TRACE_LEVEL, logger
from ..types import ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED
from ..utils import is_given
from . import trace_types, utils as telemetry_utils
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
    if (conversation_id := _conversation_id()) is None:
        return attributes
    return {trace_types.ATTR_GEN_AI_CONVERSATION_ID: conversation_id, **(attributes or {})}


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
        return self._core.disabled  # type: ignore[no-any-return]

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
    """Stamps provider metadata and the current job's attributes on every span.

    The last configured job's metadata is only a fallback outside a job context.
    """

    def __init__(self, metadata: dict[str, AttributeValue] | None = None) -> None:
        self._provider_metadata = dict(metadata) if metadata else {}
        self._fallback_metadata: dict[str, AttributeValue] = {}

    def set_metadata(self, metadata: dict[str, AttributeValue]) -> None:
        """Set the fallback for spans created outside a job context."""
        # rebind rather than mutate: on_start may read it from another thread
        self._fallback_metadata = dict(metadata)

    def clear_metadata(self) -> None:
        self._fallback_metadata = {}

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        attributes = _job_stamp_attributes()
        if attributes is None:
            attributes = self._fallback_metadata
        if self._provider_metadata:
            attributes = {**self._provider_metadata, **attributes}
        if attributes:
            span.set_attributes(attributes)


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


_MAX_PENDING_SPANS_PER_JOB = 1024


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
        if not held or not traces_enabled:
            return
        if redacted:
            held = [redact(s) for s in held]
        if self._requeue is not None:
            for s in held:
                self._requeue(s)
        else:
            self._inner.export([restore_pii(s) for s in held])

    def close_job(self, job_id: str) -> None:
        """The job ended; anything still held was never meant to upload."""
        with self._lock:
            self._open_jobs.discard(job_id)
            self._pending.pop(job_id, None)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        exportable: list[ReadableSpan] = []
        with self._lock:
            for s in spans:
                state = _job_export_state(self._export_jobs, s.attributes)
                if state is not None:
                    if state.traces_enabled:
                        # PII filtered for third-party exporters is put back here: what
                        # LiveKit Cloud may receive is the project's setting, applied at its
                        # collector. restore_pii is a no-op once that setting mandates
                        # redaction.
                        exportable.append(restore_pii(s))
                    continue
                job_id = (s.attributes or {}).get("job_id")
                if isinstance(job_id, str) and job_id in self._open_jobs:
                    held = self._pending.setdefault(job_id, [])
                    if len(held) < _MAX_PENDING_SPANS_PER_JOB:
                        held.append(s)
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
            log_record.attributes = filter_attributes(attributes)

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
        _PIIFilteringSpanProcessor(allow_pii=allow_pii if allow_pii is not None else True),
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
        # every span processor attached so far: a provider replaced mid-process keeps its
        # processor, since spans still open on it (the job's root) end there
        self._span_batch_processors: list[BatchSpanProcessor] = []
        self._span_gate: _GatedSpanExporter | None = None

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
            resource = Resource.create({SERVICE_NAME: "livekit-agents", **base_metadata})
            self._ensure_trace_pipeline(resource, self._observability_url)
            if self._span_gate is not None:
                self._span_gate.open_job(job_id)
            if not self._atexit_registered:
                self._atexit_registered = True
                atexit.register(self.shutdown_at_exit)

    def discard(self, job_id: str) -> None:
        """A job ended without ever registering: drop whatever the gate held for it."""
        with self._lock:
            if self._span_gate is not None:
                self._span_gate.close_job(job_id)

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

            # The meter provider outlives the job. Keep its resource process-stable;
            # per-job identity belongs on traces, never on metric measurements.
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
            if self._span_gate is not None:
                # release what the gate held for this job, or drop it
                self._span_gate.job_registered(
                    job_id,
                    traces_enabled=enable_traces,
                    redacted=bool((metadata or {}).get(ATTRIBUTE_REDACTION_ENABLED, False)),
                )

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
            # the provider was replaced mid-process (set_tracer_provider from an entrypoint):
            # attach to the new one. The old processor stays: spans still open on the old
            # provider, the job's root among them, end there and are exported through the
            # same gate. It shuts down with the rest at process exit.
            logger.info("tracer provider changed; attaching the LiveKit Cloud span exporter")
            if self._span_metadata_processor is not None:
                self._span_metadata_processor.clear_metadata()

        assert self._session is not None
        if self._span_gate is None:
            span_exporter = OTLPSpanExporter(
                endpoint=f"{url}/observability/traces/otlp/v0",
                compression=Compression.Gzip,
                session=self._session,
            )
            self._span_gate = _GatedSpanExporter(span_exporter, self._export_jobs)
        self._span_metadata_processor = _MetadataSpanProcessor()
        self._span_batch_processor = BatchSpanProcessor(self._span_gate)
        self._span_batch_processors.append(self._span_batch_processor)
        self._span_gate._requeue = self._span_batch_processor.on_end
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
        _prepend_log_processor(self._logger_provider, _PIIFilteringLogProcessor())

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
            for processor in self._span_batch_processors:
                flush_targets.append(("spans", processor.force_flush))
            if self._log_batch_processor is not None:
                flush_targets.append(("logs", self._log_batch_processor.force_flush))
            if self._owned_meter_provider is not None:
                flush_targets.append(("metrics", self._owned_meter_provider.force_flush))

        # flush before unregistering, so the job's remaining telemetry is exported
        _run_bounded("flush", flush_targets, timeout)

        with self._lock:
            self._export_jobs.pop(job_id, None)
            if self._span_gate is not None:
                self._span_gate.close_job(job_id)
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

    from .._core_view import from_core_item, to_core_items

    recording_options = report.options["recording_options"]
    records = _core.session_records(
        _upload_gate._core,
        {
            "agent_name": agent_name,
            "sdk_version": report.sdk_version,
            "started_at": report.started_at,
            "timestamp": report.timestamp,
            "options": _serialize_session_options(report.options),
            "recording_options": _core.RecordingOptions(**recording_options),
            "model_usage": [u.model_dump() for u in report.model_usage or []],
            "chat_history": to_core_items(report.chat_history.items),
        },
        tagger,
        lambda item: _chat_item_span_attribute(from_core_item(item)),
    )
    loggers: dict[str, Any] = {}
    for r in records:
        if (otel_logger := loggers.get(r["logger"])) is None:
            otel_logger = loggers[r["logger"]] = _get_logger(r["logger"])
        otel_logger.emit(
            body=r["body"],
            timestamp=r["timestamp_nanos"],
            attributes=r["attributes"],
            severity_number=SeverityNumber(r["severity_number"]),
            severity_text=r["severity_text"],
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
    if not attributes:
        return {}

    redacted: dict[str, Any] = {}
    for key, value in attributes.items():
        if is_pii_attribute(key):
            continue
        if key == trace_types.ATTR_EXCEPTION_TRACE:
            continue
        if key == trace_types.ATTR_EXCEPTION_MESSAGE:
            # `error.type` still names the class; only the free-form message goes
            redacted[key] = REDACTED_EXCEPTION_MESSAGE
            continue
        redacted[key] = value
    return redacted


def _contains_pii(attributes: Mapping[str, Any] | None) -> bool:
    return any(is_pii_attribute(k) or k in _REDACTED_EXCEPTION_ATTRIBUTES for k in attributes or ())


_RAW_ATTRIBUTES = "_lk_raw_attributes"


_RAW_EVENTS = "_lk_raw_events"


_RAW_STATUS = "_lk_raw_status"


class _PIIFilteringSpanProcessor(SpanProcessor):
    """Drops PII attributes so they never reach an exporter that is not LiveKit Cloud's.

    Must be registered **before** any exporting processor: ``on_end`` is dispatched in
    registration order over one shared :class:`ReadableSpan` snapshot, so rewriting the
    snapshot here is what the downstream processors go on to export.

    LiveKit Cloud is exempt — its PII handling is the project's setting in the dashboard,
    not ours to second-guess — so the original attributes are stashed on the snapshot for
    :func:`restore_pii` to put back on that one export path. When the project does mandate
    redaction, nothing is stashed and Cloud is stripped along with everyone else.

    ``allow_pii`` lifts the stripping for a provider whose exporters the integrator has
    explicitly granted PII (``set_tracer_provider(..., allow_pii=True)``).
    """

    def __init__(self, *, allow_pii: bool = False) -> None:
        self._allow_pii = allow_pii

    def on_start(self, span: SdkSpan, parent_context: otel_context.Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        from .utils import redaction_enabled

        attributes = span.attributes
        # the flag stamped at span start keeps this correct for a span ended off the
        # job's thread, and when concurrent jobs share the providers (THREAD executor)
        project_redaction = redaction_enabled(attributes)
        if self._allow_pii and not project_redaction:
            return

        events = span.events
        filtered_events = _filter_events(events)
        if not _contains_pii(attributes) and filtered_events is None:
            return

        if not project_redaction:
            # LiveKit Cloud still receives what the project allows
            setattr(span, _RAW_ATTRIBUTES, dict(attributes or {}))
            setattr(span, _RAW_EVENTS, tuple(events))
            setattr(span, _RAW_STATUS, span.status)

        # rebind rather than mutate: the snapshot's BoundedAttributes is shared with the
        # live span, and is immutable once the span has ended
        span._attributes = filter_attributes(attributes)
        if filtered_events is not None:
            span._events = filtered_events
        # record_exception also puts the message in the span status
        if span.status.status_code is StatusCode.ERROR and span.status.description:
            span._status = Status(StatusCode.ERROR, REDACTED_EXCEPTION_MESSAGE)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def redact(span: ReadableSpan) -> ReadableSpan:
    """A snapshot of ``span`` with PII stripped, whatever ``on_end`` stashed for restore.

    For a span that ended before its job decided on redaction (the gate held it): the
    processor saw redaction off at the time, so the attributes may be intact or stashed."""
    events = _filter_events(span.events)
    status = span.status
    if status.status_code is StatusCode.ERROR and status.description:
        status = Status(StatusCode.ERROR, REDACTED_EXCEPTION_MESSAGE)
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=filter_attributes(span.attributes),
        events=events if events is not None else span.events,
        links=span.links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def restore_pii(span: ReadableSpan) -> ReadableSpan:
    """The span as it was before PII was stripped for third-party exporters.

    Only LiveKit Cloud's own exporter calls this; every other destination sees the
    redacted snapshot. Returns ``span`` unchanged when nothing was stashed.
    """
    attributes = getattr(span, _RAW_ATTRIBUTES, None)
    if attributes is None:
        return span

    from .utils import redaction_enabled

    # nothing is stashed once the project mandates redaction, so this is belt-and-braces:
    # it keeps a change to that upstream check from turning into a leak here
    if redaction_enabled(span.attributes):
        return span

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=getattr(span, _RAW_EVENTS, span.events),
        links=span.links,
        kind=span.kind,
        status=getattr(span, _RAW_STATUS, span.status),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class _PIIFilteringLogProcessor(LogRecordProcessor):
    """Log counterpart of :class:`_PIIFilteringSpanProcessor`.

    ``_TraceLevelLoggingHandler`` redacts the records the framework's own handler creates;
    this covers every other emitter on the logger provider, including the exporters an
    integrator attached before handing us the provider.
    """

    def on_emit(self, log_data: ReadWriteLogRecord) -> None:
        # on_emit runs synchronously on the emitting thread, so the ambient job context is
        # the right source here — unlike spans, a log record has no earlier stamp to read
        from .utils import redaction_enabled

        if not redaction_enabled():
            return

        record = log_data.log_record
        if _contains_pii(record.attributes):
            record.attributes = filter_attributes(record.attributes)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


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
        return session._root_span_context
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
