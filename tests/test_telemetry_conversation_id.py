from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from types import MappingProxyType

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, SpanKind

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


@pytest.mark.parametrize(
    "set_attributes",
    [
        pytest.param(partial(gen_ai.set_request_attributes, operation="chat"), id="request"),
        pytest.param(partial(gen_ai.set_tool_attributes, name="get_weather"), id="tool"),
        pytest.param(
            partial(gen_ai.set_agent_attributes, operation="invoke_agent", agent_name="agent"),
            id="agent",
        ),
        pytest.param(partial(gen_ai.set_workflow_attributes, name="agent_session"), id="workflow"),
    ],
)
@pytest.mark.parametrize("explicit_id", [None, "explicit"])
@pytest.mark.parametrize("conversation_id", [None, "RM_test"])
@pytest.mark.parametrize("api", ["start_span", "start_as_current_span", "detached_span"])
def test_gen_ai_setters_preserve_conversation_id(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    set_attributes: Callable[[Span], None],
    explicit_id: str | None,
    conversation_id: str | None,
    api: str,
) -> None:
    monkeypatch.setattr(gen_ai, "_conversation_id", lambda: conversation_id)
    attributes = {"gen_ai.conversation.id": explicit_id} if explicit_id is not None else {}
    span_context: AbstractContextManager[Span]
    if api == "start_span":
        span_context = nullcontext(tracer.start_span("test", attributes=attributes))
    elif api == "start_as_current_span":
        span_context = tracer.start_as_current_span("test", attributes=attributes)
    else:
        span_context = tracer.detached_span("test", attributes=attributes)

    with span_context as span:
        set_attributes(span)
        monkeypatch.setattr(gen_ai, "_conversation_id", lambda: "RM_other")
        set_attributes(span)
        if api == "start_span":
            span.end()

    exported = span_exporter.get_finished_spans()[0]
    assert exported.attributes.get("gen_ai.conversation.id") == (explicit_id or conversation_id)
