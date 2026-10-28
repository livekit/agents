"""Framework SDK processors: per-record size admission, failure counters and bounded cleanup."""

from __future__ import annotations

import contextvars
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import fields, is_dataclass, replace
from functools import cache
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
from pydantic import BaseModel

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


@cache
def policy() -> _core._core_TelemetryPolicy:
    return _core._core_TelemetryPolicy(False, False, False)


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


def attribute_bytes(value: object, *, depth: int = 0) -> int:
    """Read bounded SDK object shape; native policy owns every retention charge."""
    native = policy()
    if depth > native.max_depth:
        return native.value_bytes("unsupported", 0, depth, [])
    if value is None or isinstance(value, (bool, float)):
        return native.value_bytes("scalar", 0, depth, [])
    if isinstance(value, int):
        return native.value_bytes("integer", value.bit_length(), depth, [])
    if isinstance(value, str):
        return native.value_bytes("string", len(value), depth, [])
    if isinstance(value, bytes):
        return native.value_bytes("bytes", len(value), depth, [])
    children: list[int] = []
    mappings: tuple[tuple[int, Iterable[tuple[object, object]]], ...] | None = None
    if isinstance(value, BaseModel):
        mappings = tuple(
            (len(mapping), mapping.items())
            for mapping in (value.__dict__, value.__pydantic_extra__, value.__pydantic_private__)
            if mapping is not None
        )
    elif is_dataclass(value) and not isinstance(value, type):
        record_fields = fields(value)
        storage = getattr(value, "__dict__", {})
        if not isinstance(storage, Mapping):
            return native.value_bytes("unsupported", 0, depth, [])
        if len(record_fields) > native.max_collection or len(storage) > native.max_collection:
            return native.value_bytes("mapping", max(len(record_fields), len(storage)), depth, [])
        slots = tuple(field for field in record_fields if field.name not in storage)
        mappings = (
            (len(storage), storage.items()),
            (len(slots), ((field.name, getattr(value, field.name)) for field in slots)),
        )
    elif isinstance(value, Mapping):
        mappings = ((len(value), value.items()),)
    if mappings is not None:
        length = sum(size for size, _ in mappings)
        if length <= native.max_collection:
            for _, entries in mappings:
                for key, item in entries:
                    children.extend(
                        (
                            attribute_bytes(key, depth=depth + 1),
                            attribute_bytes(item, depth=depth + 1),
                        )
                    )
                    if (
                        native.value_bytes("mapping", length, depth, children)
                        > native.max_record_bytes
                    ):
                        return native.value_bytes("mapping", length, depth, children)
        return native.value_bytes("mapping", length, depth, children)
    if isinstance(value, Sequence):
        if len(value) <= native.max_collection:
            for item in value:
                children.append(attribute_bytes(item, depth=depth + 1))
                if (
                    native.value_bytes("sequence", len(value), depth, children)
                    > native.max_record_bytes
                ):
                    break
        return native.value_bytes("sequence", len(value), depth, children)
    return native.value_bytes("unsupported", 0, depth, [])


def scope_bytes(scope: InstrumentationScope | None) -> int:
    if scope is None:
        return 0
    return policy().record_bytes(
        "scope",
        [
            attribute_bytes(scope.name),
            attribute_bytes(scope.version),
            attribute_bytes(scope.schema_url),
            attribute_bytes(scope.attributes),
        ],
        0,
        0,
    )


def span_bytes(span: ReadableSpan) -> int:
    native = policy()
    if len(span.events) > native.max_events or len(span.links) > native.max_events:
        return native.record_bytes("span", [], len(span.events), len(span.links))
    fields = [
        attribute_bytes(span.name),
        attribute_bytes(span.attributes),
        attribute_bytes(span.resource.attributes),
        attribute_bytes(span.status.description),
        attribute_bytes(span.resource.schema_url),
        scope_bytes(span.instrumentation_scope),
        attribute_bytes(dict(span.context.trace_state) if span.context else None),
        attribute_bytes(dict(span.parent.trace_state) if span.parent else None),
    ]
    for event in span.events:
        fields.append(
            native.record_bytes(
                "event", [attribute_bytes(event.name), attribute_bytes(event.attributes)], 0, 0
            )
        )
    for link in span.links:
        fields.append(
            native.record_bytes(
                "link",
                [attribute_bytes(link.attributes), attribute_bytes(dict(link.context.trace_state))],
                0,
                0,
            )
        )
    return native.record_bytes("span", fields, len(span.events), len(span.links))


def log_bytes(record: ReadableLogRecord | ReadWriteLogRecord) -> int:
    """Account the SDK log envelope once for queue and destination admission."""
    data = record.log_record
    return policy().record_bytes(
        "log",
        [
            attribute_bytes(data.body),
            attribute_bytes(data.attributes),
            attribute_bytes(record.resource.attributes if record.resource else None),
            attribute_bytes(record.resource.schema_url if record.resource else None),
            attribute_bytes(data.severity_text),
            attribute_bytes(data.event_name),
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
            max_queue_size=policy().max_records,
            max_export_batch_size=policy().batch_items,
            schedule_delay_millis=policy().schedule_delay_millis,
            export_timeout_millis=policy().export_timeout_millis,
        )

    def on_end(self, span: ReadableSpan) -> None:
        if span_bytes(span) > policy().max_record_bytes:
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
            max_queue_size=policy().max_records,
            max_export_batch_size=policy().batch_items,
            schedule_delay_millis=policy().schedule_delay_millis,
            export_timeout_millis=policy().export_timeout_millis,
        )

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        from ._redaction import _source_snapshot

        data = copy(log_record.log_record)
        # Admission examines the original object before copying; a refused body never crosses
        # into the SDK queue, and an admitted body cannot grow through a caller's mutable alias.
        if (
            attribute_bytes(data.body) > policy().max_record_bytes
            or attribute_bytes(data.attributes) > policy().max_record_bytes
        ):
            record("dropped")
            return
        data.body = _source_snapshot(data.body)
        attributes = _source_snapshot(data.attributes or {})
        assert isinstance(attributes, dict)
        data.attributes = attributes
        snapshot = replace(log_record, log_record=data)
        if log_bytes(snapshot) > policy().max_record_bytes:
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
