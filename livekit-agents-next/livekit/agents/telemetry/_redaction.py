"""Independent host SDK snapshots applying native destination policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from opentelemetry.sdk._logs import (
    ReadableLogRecord,
)
from opentelemetry.sdk._logs.export import (
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, Status, StatusCode
from opentelemetry.util.types import AnyValue, AttributeValue

from .. import _core
from ..types import ATTRIBUTE_REDACTION_ENABLED
from . import _export
from ._export import log_bytes, span_bytes

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Event


def _policy(
    attributes: Mapping[str, Any] | None, *, allow_pii: bool, project_redaction: bool = False
) -> _core._core_TelemetryPolicy:
    attrs = attributes or {}
    return _core._core_TelemetryPolicy(
        allow_pii,
        attrs.get("lk.telemetry.capture_content") is True,
        project_redaction or bool(attrs.get(ATTRIBUTE_REDACTION_ENABLED)),
    )


def filter_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return _attribute_snapshot(attributes, redacted=True)


def _attribute_snapshot(attributes: Mapping[str, Any] | None, *, redacted: bool) -> dict[str, Any]:
    policy = _core._core_TelemetryPolicy(not redacted, True, False)
    filtered = _filter_attribute_value(attributes or {}, policy=policy, depth=0)
    return filtered if isinstance(filtered, dict) else {}


def _filter_attribute_value(
    value: object,
    *,
    policy: _core._core_TelemetryPolicy,
    depth: int,
    destination: bool = True,
    parent: str | None = None,
) -> AnyValue:
    if depth > _core.TELEMETRY_MAX_DEPTH:
        return None
    if isinstance(value, Mapping):
        if len(value) > _core.TELEMETRY_MAX_COLLECTION:
            return None
        filtered: dict[str, AnyValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            keep, replacement = policy.attribute_action(parent, key, destination)
            if keep:
                filtered[key] = (
                    replacement
                    if replacement is not None
                    else _filter_attribute_value(
                        item, policy=policy, depth=depth + 1, destination=destination, parent=key
                    )
                )
        return filtered
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > _core.TELEMETRY_MAX_COLLECTION:
            return None
        return [
            _filter_attribute_value(
                item, policy=policy, depth=depth + 1, destination=destination, parent=parent
            )
            for item in value
        ]
    return value if value is None or isinstance(value, (str, bool, int, float, bytes)) else None


def _source_snapshot(value: object) -> AnyValue:
    """Copy SDK primitives before queue admission without stripping destination-private markers."""
    return _filter_attribute_value(
        value,
        policy=_core._core_TelemetryPolicy(True, True, False),
        depth=0,
        destination=False,
    )


def _redact_content(
    attributes: Mapping[str, AttributeValue] | None,
    *,
    allow_pii: bool,
    project_redaction: bool = False,
) -> bool:
    return _policy(attributes, allow_pii=allow_pii, project_redaction=project_redaction).redacted


def _log_snapshot(
    record: ReadableLogRecord, *, allow_pii: bool, project_redaction: bool = False
) -> ReadableLogRecord:
    data = copy(record.log_record)
    policy = _policy(data.attributes, allow_pii=allow_pii, project_redaction=project_redaction)
    safe_body = (data.attributes or {}).get("lk.telemetry.safe_body") is True
    data.attributes = _attribute_snapshot(data.attributes, redacted=policy.redacted)
    replacement = policy.log_body(safe_body, data.body if isinstance(data.body, str) else None)
    data.body = (
        replacement
        if replacement is not None
        else _filter_attribute_value(
            data.body, policy=_core._core_TelemetryPolicy(True, True, False), depth=0
        )
    )
    return replace(
        record,
        log_record=data,
        resource=_resource_snapshot(record.resource, redacted=policy.redacted),
        instrumentation_scope=_scope_snapshot(
            record.instrumentation_scope, redacted=policy.redacted
        ),
    )


class RedactingLogExporter(LogRecordExporter):
    """Apply capture and redaction policy to one application's log destination."""

    def __init__(self, inner: LogRecordExporter, *, allow_pii: bool = False) -> None:
        self._inner = inner
        self._allow_pii = allow_pii

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        snapshots: list[ReadableLogRecord] = []
        refused = False
        for record in batch:
            if log_bytes(record) > _core.SDK_RECORD_BYTES:
                _export.record("dropped")
                refused = True
            else:
                snapshots.append(_log_snapshot(record, allow_pii=self._allow_pii))
        result = self._inner.export(snapshots) if snapshots else LogRecordExportResult.SUCCESS
        return LogRecordExportResult.FAILURE if refused else result

    def shutdown(self) -> None:
        cast(Callable[[], None], self._inner.shutdown)()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _resource_snapshot(resource: Resource, *, redacted: bool) -> Resource:
    attributes = _attribute_snapshot(resource.attributes, redacted=redacted)
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
    policy = _core._core_TelemetryPolicy(not redacted, True, False)
    events = _event_snapshots(span.events, policy=policy)
    status = span.status
    status = Status(
        status.status_code,
        policy.status_description(status.status_code is StatusCode.ERROR, status.description),
    )
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=_resource_snapshot(span.resource, redacted=redacted),
        attributes=_attribute_snapshot(span.attributes, redacted=redacted),
        events=tuple(events),
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
        snapshots: list[ReadableSpan] = []
        refused = False
        for span in spans:
            if span_bytes(span) > _core.SDK_RECORD_BYTES:
                _export.record("dropped")
                refused = True
            else:
                snapshots.append(
                    _span_snapshot(
                        span,
                        redacted=_redact_content(span.attributes, allow_pii=self._allow_pii),
                    )
                )
        result = self._inner.export(snapshots) if snapshots else SpanExportResult.SUCCESS
        return SpanExportResult.FAILURE if refused else result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _event_snapshots(
    events: Sequence[Event], *, policy: _core._core_TelemetryPolicy
) -> list[Event]:
    from opentelemetry.sdk.trace import Event as SdkEvent

    return [
        SdkEvent(
            name=event.name,
            attributes=_attribute_snapshot(event.attributes, redacted=policy.redacted),
            timestamp=event.timestamp,
        )
        for event in events
        if policy.keep_event(event.name)
    ]
