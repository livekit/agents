"""Cloud SDK destination gating, authenticated requests and early log retention."""

from __future__ import annotations

import logging
import threading
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import requests
from opentelemetry._logs import LogRecord as OTelLogRecord
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.sdk._logs import (
    LoggingHandler,
    ReadableLogRecord,
)
from opentelemetry.sdk._logs.export import (
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.util.types import AttributeValue

from .. import _core
from ..log import TRACE_LEVEL
from ..types import ATTRIBUTE_REDACTION_ENABLED
from . import _context as tracing, utils as telemetry_utils
from ._export import (
    attribute_bytes,
    exporting,
    span_key,
    worker,
)
from ._redaction import (
    _log_snapshot,
    _redact_content,
    _source_snapshot,
    _span_snapshot,
    filter_attributes,
    redact,
)

upload_gate = _core.UploadGate()


class _RecordingDisabled(Exception):
    """Terminal Cloud export outcome, caught by the destination's SDK exporter boundary."""


class _AuthRefreshingSession(requests.Session):
    """Refresh SDK request authentication and surface the native recording-stop outcome."""

    def __init__(self, header_provider: Callable[[], dict[str, str]]) -> None:
        super().__init__()
        self._header_provider = header_provider

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        if upload_gate.disabled:
            raise _RecordingDisabled("LiveKit Cloud recording is disabled")
        self.headers.update(self._header_provider())
        response = super().request(*args, **kwargs)
        if upload_gate.is_disabled_response(response.status_code, response.content):
            upload_gate.disable()
            raise _RecordingDisabled("LiveKit Cloud recording is disabled")
        return response


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
        selected: list[tuple[ReadableSpan, bool]] = []
        policy = _core._core_TelemetryPolicy(True, True, False)
        with self._lock:
            for span in spans:
                state = _job_export_state(self._export_jobs, span.attributes)
                job_id = (span.attributes or {}).get("job_id")
                held = self._pending.get(job_id, []) if isinstance(job_id, str) else []
                action = policy.export_action(
                    state is not None,
                    state.traces_enabled if state else False,
                    isinstance(job_id, str) and job_id in self._open_jobs,
                    len(held),
                    upload_gate.disabled,
                )
                if action == "export":
                    assert state is not None
                    selected.append(
                        (
                            span,
                            _redact_content(
                                span.attributes,
                                allow_pii=True,
                                project_redaction=bool(
                                    state.attributes.get(ATTRIBUTE_REDACTION_ENABLED)
                                ),
                            ),
                        )
                    )
                elif action == "hold":
                    assert isinstance(job_id, str)
                    self._pending.setdefault(job_id, []).append(span)
                elif action == "capacity":
                    worker().record_drop("capacity", 1)
        exportable = [_span_snapshot(span, redacted=redacted) for span, redacted in selected]
        if not exportable:
            return SpanExportResult.FAILURE if upload_gate.disabled else SpanExportResult.SUCCESS
        try:
            return self._inner.export(exportable)
        except _RecordingDisabled:
            return SpanExportResult.FAILURE

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
            action = _core._core_TelemetryPolicy(True, True, False).export_action(
                state is not None,
                state.logs_enabled if state else False,
                False,
                0,
                upload_gate.disabled,
            )
            if action != "export":
                continue
            assert state is not None
            exportable.append(
                _log_snapshot(
                    record,
                    allow_pii=True,
                    project_redaction=bool(state.attributes.get(ATTRIBUTE_REDACTION_ENABLED)),
                )
            )
        if not exportable:
            return (
                LogRecordExportResult.FAILURE
                if upload_gate.disabled
                else LogRecordExportResult.SUCCESS
            )
        try:
            return self._inner.export(exportable)
        except _RecordingDisabled:
            return LogRecordExportResult.FAILURE

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
        self._finalizer: weakref.finalize[None] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if exporting():
            return
        runtime = worker()
        if (
            attribute_bytes(record.__dict__)
            > _core._core_TelemetryPolicy(False, False, False).max_record_bytes
        ):
            runtime.record_drop("oversize", 1)
            return
        record = copy(record)
        captured = tracing.capture_content_enabled()
        from ._providers import _job_stamp_attributes

        identity = _job_stamp_attributes() or {}
        if _core._core_TelemetryPolicy(
            True, captured, telemetry_utils.redaction_enabled(identity)
        ).redacted:
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
            record.msg = _core._core_TelemetryPolicy(False, False, False).log_body(False, None)
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            identity = filter_attributes(identity)
        record.__dict__.update(identity)
        record.__dict__["lk.telemetry.capture_content"] = captured
        if (
            attribute_bytes(record.__dict__)
            > _core._core_TelemetryPolicy(False, False, False).max_record_bytes
        ):
            runtime.record_drop("oversize", 1)
            return
        snapshot = _source_snapshot(record.__dict__)
        assert isinstance(snapshot, dict)
        if isinstance(record.args, tuple):
            snapshot["args"] = tuple(cast(list[object], snapshot["args"]))
        record.__dict__ = snapshot
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
            log_record.attributes = filter_attributes(log_record.attributes)

        # OTel's std_to_otel returns UNSPECIFIED for levels < 10
        # Map our TRACE_LEVEL to OTel's TRACE
        if record.levelno == TRACE_LEVEL:
            log_record.severity_number = SeverityNumber.TRACE
        return log_record
