from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

_CONVERSATION_ID = "gen_ai.conversation.id"


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


@pytest.fixture
def in_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen_ai, "_conversation_id", lambda: "RM_test")


def test_every_span_carries_the_conversation_id(
    span_exporter: InMemorySpanExporter, in_conversation: None
) -> None:
    tracer.start_span("tts_node").end()
    with tracer.start_as_current_span("eou_detection"):
        pass

    spans = span_exporter.get_finished_spans()
    assert {span.name for span in spans} == {"tts_node", "eou_detection"}
    for span in spans:
        assert (span.attributes or {})[_CONVERSATION_ID] == "RM_test"


def test_caller_attributes_are_preserved(
    span_exporter: InMemorySpanExporter, in_conversation: None
) -> None:
    tracer.start_span("function_tool", attributes={"lk.function_tool.name": "lookup"}).end()

    attributes = span_exporter.get_finished_spans()[0].attributes or {}
    assert attributes["lk.function_tool.name"] == "lookup"
    assert attributes[_CONVERSATION_ID] == "RM_test"


def test_no_conversation_id_outside_a_job(span_exporter: InMemorySpanExporter) -> None:
    tracer.start_span("tts_node").end()

    assert _CONVERSATION_ID not in (span_exporter.get_finished_spans()[0].attributes or {})
