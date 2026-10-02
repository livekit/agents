from __future__ import annotations

from collections.abc import Iterator
from types import MappingProxyType

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


@pytest.mark.parametrize("conversation_id", [None, "RM_test"])
@pytest.mark.parametrize("explicit_id", [None, "explicit"])
@pytest.mark.parametrize("api", ["start_span", "start_as_current_span", "detached_span"])
def test_conversation_id_preserves_caller_attributes(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    conversation_id: str | None,
    explicit_id: str | None,
    api: str,
) -> None:
    monkeypatch.setattr(gen_ai, "_conversation_id", lambda: conversation_id)
    attributes = {"test.attribute": "value"}
    if explicit_id:
        attributes["gen_ai.conversation.id"] = explicit_id
    original = attributes.copy()
    # The standard tracer APIs also allow positional, read-only attribute mappings.
    if api == "start_span":
        tracer.start_span("test", None, SpanKind.INTERNAL, MappingProxyType(attributes)).end()
    elif api == "start_as_current_span":
        with tracer.start_as_current_span(
            "test", None, SpanKind.INTERNAL, MappingProxyType(attributes)
        ):
            pass
    else:
        with tracer.detached_span("test", attributes=attributes):
            pass
    assert attributes == original
    expected = dict(original)
    if explicit_id or conversation_id:
        expected["gen_ai.conversation.id"] = explicit_id or conversation_id
    assert dict(span_exporter.get_finished_spans()[0].attributes) == expected


def test_spans_without_attributes_receive_conversation_id(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen_ai, "_conversation_id", lambda: "RM_test")
    tracer.start_span("test").end()
    with tracer.start_as_current_span("test"):
        pass
    with tracer.detached_span("test"):
        pass
    assert len(span_exporter.get_finished_spans()) == 3
    for span in span_exporter.get_finished_spans():
        assert span.attributes["gen_ai.conversation.id"] == "RM_test"
