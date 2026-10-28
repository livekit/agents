"""Host SDK admission and accounting. Native telemetry owns capacity and cleanup work."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from copy import copy
from dataclasses import fields, is_dataclass, replace
from functools import cache
from typing import TypeVar, cast

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


_worker: _core._core_TelemetryWorker | None = None
_worker_lock = threading.Lock()


def worker() -> _core._core_TelemetryWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _core._core_TelemetryWorker()
        return _worker


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
            max_queue_size=policy().max_records,
            max_export_batch_size=policy().batch_items,
            schedule_delay_millis=policy().schedule_delay_millis,
            export_timeout_millis=policy().export_timeout_millis,
        )

    def on_end(self, span: ReadableSpan) -> None:
        charge = span_bytes(span)
        if charge > policy().max_record_bytes:
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


_ResultT = TypeVar("_ResultT")


async def call_bounded(
    key: str, callback: Callable[[], _ResultT], retained_bytes: int, timeout: float
) -> _ResultT:
    """Run host work on the native worker, retaining late callbacks after caller timeout."""
    result: Future[_ResultT] = Future()
    context = contextvars.copy_context()

    def invoke() -> bool:
        try:
            result.set_result(context.run(callback))
        except BaseException as error:
            result.set_exception(error)
            return False
        return True

    runtime = worker()
    if not runtime.submit(key, invoke, retained_bytes):
        raise RuntimeError("telemetry worker admission refused")
    if not await runtime.wait_async(key, timeout):
        raise TimeoutError("telemetry worker did not settle before its deadline")
    if not result.done():
        raise RuntimeError("telemetry callback settled without a result")
    return result.result()


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
            max_queue_size=policy().max_records,
            max_export_batch_size=policy().batch_items,
            schedule_delay_millis=policy().schedule_delay_millis,
            export_timeout_millis=policy().export_timeout_millis,
        )

    def on_emit(self, record: ReadWriteLogRecord) -> None:
        from ._redaction import _source_snapshot

        data = copy(record.log_record)
        # Admission examines the original object before copying; a refused body never crosses
        # into the SDK queue, and an admitted body cannot grow through a caller's mutable alias.
        if (
            attribute_bytes(data.body) > policy().max_record_bytes
            or attribute_bytes(data.attributes) > policy().max_record_bytes
        ):
            worker().record_drop("oversize", 1)
            return
        data.body = _source_snapshot(data.body)
        attributes = _source_snapshot(data.attributes or {})
        assert isinstance(attributes, dict)
        with self._lock:
            self._sequence += 1
            key = f"log:{id(self):x}:{self._sequence}"
        data.attributes = {**attributes, "lk.telemetry.export_key": key}
        snapshot = replace(record, log_record=data)
        charge = log_bytes(snapshot)
        if charge > policy().max_record_bytes:
            worker().record_drop("oversize", 1)
            return
        if worker().reserve(key, charge):
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


def diagnostics() -> dict[str, int]:
    """Return process-wide export drops, failures and cleanup counters without exporting them."""
    return dict(worker().diagnostics)
