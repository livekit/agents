"""Framework SDK processors: per-record size admission, failure counters and bounded cleanup."""

from __future__ import annotations

import contextvars
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import Literal, cast

from opentelemetry.sdk._logs import LogRecordProcessor, ReadableLogRecord, ReadWriteLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExporter,
    MetricExportResult,
    MetricsData,
)
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

from .. import _core

_exporting: contextvars.ContextVar[bool] = contextvars.ContextVar("lk_sdk_exporting", default=False)


def exporting() -> bool:
    return _exporting.get()


@contextmanager
def export_context() -> Iterator[None]:
    token = _exporting.set(True)
    try:
        yield
    finally:
        _exporting.reset(token)


Outcome = Literal["dropped", "failed", "conflicts", "incomplete_cleanup"]

_outcomes: Counter[str] = Counter()
_outcomes_lock = threading.Lock()


def record(outcome: Outcome, count: int = 1) -> None:
    """Count a refused record, a failed export, a configuration conflict or a late cleanup."""
    with _outcomes_lock:
        _outcomes[outcome] += count


def diagnostics() -> dict[str, int]:
    """Return process-wide export drops, failures and cleanup counters without exporting them."""
    with _outcomes_lock:
        return {
            outcome: _outcomes[outcome]
            for outcome in ("dropped", "failed", "conflicts", "incomplete_cleanup")
        }


def scope_bytes(scope: InstrumentationScope | None) -> int:
    if scope is None:
        return 0
    return _core._core_record_bytes(
        "scope",
        [
            _core._core_value_bytes(scope.name),
            _core._core_value_bytes(scope.version),
            _core._core_value_bytes(scope.schema_url),
            _core._core_value_bytes(scope.attributes),
        ],
        0,
        0,
    )


def span_bytes(span: ReadableSpan) -> int:
    if (
        len(span.events) > _core.TELEMETRY_MAX_EVENTS
        or len(span.links) > _core.TELEMETRY_MAX_EVENTS
    ):
        return _core._core_record_bytes("span", [], len(span.events), len(span.links))
    fields = [
        _core._core_value_bytes(span.name),
        _core._core_value_bytes(span.attributes),
        _core._core_value_bytes(span.resource.attributes),
        _core._core_value_bytes(span.status.description),
        _core._core_value_bytes(span.resource.schema_url),
        scope_bytes(span.instrumentation_scope),
        _core._core_value_bytes(dict(span.context.trace_state) if span.context else None),
        _core._core_value_bytes(dict(span.parent.trace_state) if span.parent else None),
    ]
    for event in span.events:
        fields.append(
            _core._core_record_bytes(
                "event",
                [_core._core_value_bytes(event.name), _core._core_value_bytes(event.attributes)],
                0,
                0,
            )
        )
    for link in span.links:
        fields.append(
            _core._core_record_bytes(
                "link",
                [
                    _core._core_value_bytes(link.attributes),
                    _core._core_value_bytes(dict(link.context.trace_state)),
                ],
                0,
                0,
            )
        )
    return _core._core_record_bytes("span", fields, len(span.events), len(span.links))


def log_bytes(record: ReadableLogRecord | ReadWriteLogRecord) -> int:
    """Account the SDK log envelope once for queue and destination admission."""
    data = record.log_record
    return _core._core_record_bytes(
        "log",
        [
            _core._core_value_bytes(data.body),
            _core._core_value_bytes(data.attributes),
            _core._core_value_bytes(record.resource.attributes if record.resource else None),
            _core._core_value_bytes(record.resource.schema_url if record.resource else None),
            _core._core_value_bytes(data.severity_text),
            _core._core_value_bytes(data.event_name),
            scope_bytes(record.instrumentation_scope),
        ],
        0,
        0,
    )


class _GuardedSpanExporter(SpanExporter):
    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with export_context():
                result = self.inner.export(spans)
            if result is SpanExportResult.FAILURE:
                record("failed")
            return result
        except Exception:
            record("failed")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        with export_context():
            self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with export_context():
            return self.inner.force_flush(timeout_millis)


class BoundedSpanProcessor(SpanProcessor):
    """Refuse oversize spans before the SDK batch queue, which bounds the record count."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._processor = BatchSpanProcessor(
            _GuardedSpanExporter(exporter),
            max_queue_size=_core.SDK_PENDING_RECORDS,
            max_export_batch_size=_core.SDK_BATCH_ITEMS,
            schedule_delay_millis=_core.SDK_SCHEDULE_DELAY_MILLIS,
            export_timeout_millis=_core.SDK_EXPORT_TIMEOUT_MILLIS,
        )

    def on_end(self, span: ReadableSpan) -> None:
        if span_bytes(span) > _core.SDK_RECORD_BYTES:
            record("dropped")
            return
        self._processor.on_end(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._processor.force_flush(timeout_millis)

    def shutdown(self) -> None:
        cast(Callable[[], None], self._processor.shutdown)()


def run_bounded(
    action: str, targets: list[tuple[str, Callable[[], object]]], timeout: float
) -> None:
    """Run SDK flush or shutdown callbacks concurrently, waiting at most ``timeout`` in total.

    Application processors may block: a callback still running at the deadline keeps its daemon
    thread and is counted as incomplete cleanup."""

    def run(callback: Callable[[], object]) -> None:
        try:
            # SDK force_flush reports failure with False; shutdown returns None.
            succeeded = callback() is not False
        except Exception:
            succeeded = False
        if not succeeded:
            record("failed")

    threads = [
        threading.Thread(target=run, args=(callback,), name=f"lk-{action}-{name}", daemon=True)
        for name, callback in targets
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + max(0.0, timeout)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            record("incomplete_cleanup")


class _GuardedLogExporter(LogRecordExporter):
    def __init__(self, inner: LogRecordExporter) -> None:
        self._inner = inner

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        try:
            with export_context():
                result = self._inner.export(batch)
            if result is LogRecordExportResult.FAILURE:
                record("failed")
            return result
        except Exception:
            record("failed")
            return LogRecordExportResult.FAILURE

    def shutdown(self) -> None:
        with export_context():
            cast(Callable[[], None], self._inner.shutdown)()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with export_context():
            return self._inner.force_flush(timeout_millis)


class BoundedLogRecordProcessor(LogRecordProcessor):
    """Admit an independent SDK log snapshot before batching it."""

    def __init__(self, exporter: LogRecordExporter) -> None:
        self._processor = BatchLogRecordProcessor(
            _GuardedLogExporter(exporter),
            max_queue_size=_core.SDK_PENDING_RECORDS,
            max_export_batch_size=_core.SDK_BATCH_ITEMS,
            schedule_delay_millis=_core.SDK_SCHEDULE_DELAY_MILLIS,
            export_timeout_millis=_core.SDK_EXPORT_TIMEOUT_MILLIS,
        )

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        from ._redaction import _source_snapshot

        data = copy(log_record.log_record)
        # Admission examines the original object before copying; a refused body never crosses
        # into the SDK queue, and an admitted body cannot grow through a caller's mutable alias.
        if (
            _core._core_value_bytes(data.body) > _core.SDK_RECORD_BYTES
            or _core._core_value_bytes(data.attributes) > _core.SDK_RECORD_BYTES
        ):
            record("dropped")
            return
        data.body = _source_snapshot(data.body)
        attributes = _source_snapshot(data.attributes or {})
        assert isinstance(attributes, dict)
        data.attributes = attributes
        snapshot = replace(log_record, log_record=data)
        if log_bytes(snapshot) > _core.SDK_RECORD_BYTES:
            record("dropped")
            return
        self._processor.on_emit(snapshot)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._processor.force_flush(timeout_millis)

    def shutdown(self) -> None:
        cast(Callable[[], None], self._processor.shutdown)()


class GuardedMetricExporter(MetricExporter):
    """Keep SDK failure logs out of the framework's own telemetry destination."""

    def __init__(
        self,
        inner: MetricExporter,
        *,
        preferred_temporality: dict[type[object], AggregationTemporality] | None = None,
        enabled: Callable[[], bool] = lambda: True,
    ) -> None:
        super().__init__(preferred_temporality=preferred_temporality)
        self._inner = inner
        self._enabled = enabled

    def export(
        self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: object
    ) -> MetricExportResult:
        if not self._enabled():
            return MetricExportResult.FAILURE
        try:
            with export_context():
                result = self._inner.export(metrics_data, timeout_millis=timeout_millis, **kwargs)
            if result is MetricExportResult.FAILURE:
                record("failed")
            return result
        except Exception:
            record("failed")
            return MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        with export_context():
            return self._inner.force_flush(timeout_millis=timeout_millis)

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        with export_context():
            self._inner.shutdown(timeout_millis=timeout_millis, **kwargs)
