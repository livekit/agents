"""Host SDK admission and accounting. Native telemetry owns capacity and cleanup work."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import cast

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


_MAX_RECORD_BYTES = 64 * 1024
_worker: _core._core_TelemetryWorker | None = None
_worker_lock = threading.Lock()


def worker() -> _core._core_TelemetryWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _core._core_TelemetryWorker()
        return _worker


def attribute_bytes(value: object, *, depth: int = 0) -> int:
    """Conservative retained-byte charge, refusing unbounded host SDK payloads."""
    if depth > 4:
        return _MAX_RECORD_BYTES + 1
    if value is None or isinstance(value, (bool, int, float)):
        return 32
    if isinstance(value, str):
        return 64 + len(value) * 4
    if isinstance(value, bytes):
        return 64 + len(value)
    if isinstance(value, Mapping):
        if len(value) > 128:
            return _MAX_RECORD_BYTES + 1
        total = 128
        for key, item in value.items():
            total += attribute_bytes(key, depth=depth + 1)
            total += attribute_bytes(item, depth=depth + 1)
            if total > _MAX_RECORD_BYTES:
                break
        return total
    if isinstance(value, Sequence):
        if len(value) > 128:
            return _MAX_RECORD_BYTES + 1
        total = 64
        for item in value:
            total += attribute_bytes(item, depth=depth + 1)
            if total > _MAX_RECORD_BYTES:
                break
        return total
    return _MAX_RECORD_BYTES + 1


def scope_bytes(scope: InstrumentationScope | None) -> int:
    if scope is None:
        return 0
    return (
        128
        + attribute_bytes(scope.name)
        + attribute_bytes(scope.version)
        + attribute_bytes(scope.schema_url)
        + attribute_bytes(scope.attributes)
    )


def span_bytes(span: ReadableSpan) -> int:
    if len(span.events) > 32 or len(span.links) > 32:
        return _MAX_RECORD_BYTES + 1
    total = (
        512
        + attribute_bytes(span.name)
        + attribute_bytes(span.attributes)
        + attribute_bytes(span.resource.attributes)
        + attribute_bytes(span.status.description)
        + attribute_bytes(span.resource.schema_url)
        + scope_bytes(span.instrumentation_scope)
        + attribute_bytes(dict(span.context.trace_state) if span.context else None)
        + attribute_bytes(dict(span.parent.trace_state) if span.parent else None)
    )
    if total > _MAX_RECORD_BYTES:
        return total
    for event in span.events:
        total += 128 + attribute_bytes(event.name) + attribute_bytes(event.attributes)
        if total > _MAX_RECORD_BYTES:
            return total
    for link in span.links:
        total += (
            128 + attribute_bytes(link.attributes) + attribute_bytes(dict(link.context.trace_state))
        )
        if total > _MAX_RECORD_BYTES:
            return total
    return total


def span_key(span: ReadableSpan) -> str:
    context = span.context
    if context is None:
        return f"span:{id(span):x}"
    return f"span:{context.trace_id:032x}:{context.span_id:016x}"


class _ReleasingSpanExporter(SpanExporter):
    def __init__(self, inner: SpanExporter, holds: Callable[[ReadableSpan], bool]) -> None:
        self.inner = inner
        self._holds = holds

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with export_context():
                result = self.inner.export(spans)
            if result is SpanExportResult.FAILURE:
                worker().record_drop("export_failed", 1)
            return result
        except Exception:
            worker().record_drop("export_failed", 1)
            return SpanExportResult.FAILURE
        finally:
            for span in spans:
                # A gate holding an undecided job's span retains this reservation itself.
                if not self._holds(span):
                    worker().release(span_key(span))

    def shutdown(self) -> None:
        with export_context():
            self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with export_context():
            return self.inner.force_flush(timeout_millis)


class BoundedSpanProcessor(SpanProcessor):
    """Bound host SDK retention by records and bytes before its batch queue admits a span."""

    def __init__(self, exporter: SpanExporter, *, holds: Callable[[ReadableSpan], bool]) -> None:
        self._processor = BatchSpanProcessor(
            _ReleasingSpanExporter(exporter, holds),
            max_queue_size=64,
            max_export_batch_size=16,
            schedule_delay_millis=1000,
            export_timeout_millis=5000,
        )

    def on_end(self, span: ReadableSpan) -> None:
        charge = span_bytes(span)
        if charge > _MAX_RECORD_BYTES:
            worker().record_drop("oversize", 1)
            return
        if worker().reserve(span_key(span), charge):
            self._processor.on_end(span)

    def requeue(self, span: ReadableSpan) -> None:
        """The gate already owns the reservation for an early span."""
        self._processor.on_end(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._processor.force_flush(timeout_millis)

    def shutdown(self) -> None:
        cast(Callable[[], None], self._processor.shutdown)()


def run_bounded(
    action: str, targets: list[tuple[str, Callable[[], object]]], timeout: float
) -> None:
    """Wait only from job cleanup's worker thread or process exit, retaining late owners."""
    import time

    if not targets:
        return
    runtime = worker()
    keys: list[str] = []
    for name, callback in targets:
        key = f"{action}:{name}"
        if runtime.submit(key, callback, 0):
            keys.append(key)
    deadline = time.monotonic() + max(0.0, timeout)
    for key in keys:
        runtime.wait(key, max(0.0, deadline - time.monotonic()))


class _ReleasingLogExporter(LogRecordExporter):
    def __init__(self, inner: LogRecordExporter) -> None:
        self._inner = inner

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        try:
            with export_context():
                result = self._inner.export(batch)
            if result is LogRecordExportResult.FAILURE:
                worker().record_drop("export_failed", 1)
            return result
        except Exception:
            worker().record_drop("export_failed", 1)
            return LogRecordExportResult.FAILURE
        finally:
            for record in batch:
                key = (record.log_record.attributes or {}).get("lk.telemetry.export_key")
                if isinstance(key, str):
                    worker().release(key)

    def shutdown(self) -> None:
        with export_context():
            cast(Callable[[], None], self._inner.shutdown)()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with export_context():
            return self._inner.force_flush(timeout_millis)


class BoundedLogRecordProcessor(LogRecordProcessor):
    """Admit an independent SDK log snapshot before batching it."""

    def __init__(self, exporter: LogRecordExporter) -> None:
        self._sequence = 0
        self._lock = threading.Lock()
        self._processor = BatchLogRecordProcessor(
            _ReleasingLogExporter(exporter),
            max_queue_size=64,
            max_export_batch_size=16,
            schedule_delay_millis=1000,
            export_timeout_millis=5000,
        )

    def on_emit(self, record: ReadWriteLogRecord) -> None:
        data = record.log_record
        charge = (
            512
            + attribute_bytes(data.body)
            + attribute_bytes(data.attributes)
            + attribute_bytes(record.resource.attributes if record.resource else None)
            + attribute_bytes(record.resource.schema_url if record.resource else None)
            + attribute_bytes(data.severity_text)
            + attribute_bytes(data.event_name)
            + scope_bytes(record.instrumentation_scope)
        )
        if charge > _MAX_RECORD_BYTES:
            worker().record_drop("oversize", 1)
            return
        with self._lock:
            self._sequence += 1
            key = f"log:{id(self):x}:{self._sequence}"
        if not worker().reserve(key, charge):
            return
        snapshot = copy(data)
        snapshot.attributes = {**(data.attributes or {}), "lk.telemetry.export_key": key}
        self._processor.on_emit(replace(record, log_record=snapshot))

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
    ) -> None:
        super().__init__(preferred_temporality=preferred_temporality)
        self._inner = inner

    def export(
        self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: object
    ) -> MetricExportResult:
        try:
            with export_context():
                result = self._inner.export(metrics_data, timeout_millis=timeout_millis, **kwargs)
            if result is MetricExportResult.FAILURE:
                worker().record_drop("export_failed", 1)
            return result
        except Exception:
            worker().record_drop("export_failed", 1)
            return MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        with export_context():
            return self._inner.force_flush(timeout_millis=timeout_millis)

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        with export_context():
            self._inner.shutdown(timeout_millis=timeout_millis, **kwargs)
