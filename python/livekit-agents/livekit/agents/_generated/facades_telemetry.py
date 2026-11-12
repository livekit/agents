# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span

from .. import _bindings, _core

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.util.types import AttributeValue

    from .telemetry import SpanContext


class _Span(Span):
    """`telemetry._Span`: the host's OpenTelemetry span over a native one."""

    def __init__(self, native: _core.TelemetrySpan) -> None:
        super().__init__()
        self._native = native

    def get_span_context(self) -> SpanContext:
        return self._native.context

    def end(self, end_time: int | None = None) -> None:
        self._native.end(end_time)

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        self._native.set_attributes(attributes)

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._native.set_attribute(key, value)

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
        timestamp: int | None = None,
    ) -> None:
        self._native.add_event(name, attributes, timestamp)

    def update_name(self, name: str) -> None:
        self._native.update_name(name)

    def is_recording(self) -> bool:
        return self._native.is_recording()
