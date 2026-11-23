"""Attributes the tracer stamps on every span, and the process resource."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from types import MappingProxyType, SimpleNamespace

import pytest
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, SpanKind
from opentelemetry.util.types import AttributeValue

from livekit.agents.job import _JobContextVar
from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer, traces
from livekit.agents.telemetry.traces import _JobTelemetry, _MetadataSpanProcessor
from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED

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


# -- provider and job metadata --


@pytest.fixture
def metadata_provider() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(
        provider,
        metadata={
            "langfuse.session.id": "customer-session",
            "job_id": "provider-job",
            "room_id": "provider-room",
        },
    )
    try:
        yield provider, exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


@contextmanager
def _job_context(*, initialized: bool = True, traces_enabled: bool = True) -> Iterator[None]:
    ctx = SimpleNamespace(
        job=SimpleNamespace(id="job-a", room=SimpleNamespace(sid="room-a")),
        _telemetry_state=(
            _JobTelemetry(
                attributes={"job_id": "job-a", "room_id": "room-a"},
                traces_enabled=traces_enabled,
                logs_enabled=False,
            )
            if initialized
            else None
        ),
        _redaction_enabled=False,
    )
    token = _JobContextVar.set(ctx)  # type: ignore[arg-type]
    try:
        yield
    finally:
        _JobContextVar.reset(token)


@pytest.mark.parametrize("state", ["outside", "uninitialized", "recorded", "disabled"])
def test_exported_spans_keep_provider_metadata(
    metadata_provider: tuple[TracerProvider, InMemorySpanExporter], state: str
) -> None:
    _, exporter = metadata_provider
    context = (
        nullcontext()
        if state == "outside"
        else _job_context(initialized=state != "uninitialized", traces_enabled=state != "disabled")
    )
    with context, tracer.start_as_current_span("agent_session"):
        with tracer.start_as_current_span("llm_request"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.attributes == {
            "langfuse.session.id": "customer-session",
            "job_id": "provider-job" if state == "outside" else "job-a",
            "room_id": "provider-room" if state == "outside" else "room-a",
            **({"gen_ai.conversation.id": "room-a"} if state != "outside" else {}),
        }


@pytest.mark.parametrize("initialized", [False, True])
def test_job_fallback_metadata_does_not_cross_jobs(
    metadata_provider: tuple[TracerProvider, InMemorySpanExporter], initialized: bool
) -> None:
    provider, exporter = metadata_provider
    processor = _MetadataSpanProcessor()
    fallback: dict[str, AttributeValue] = {
        "job_id": "job-b",
        "room_id": "room-b",
        ATTRIBUTE_SIMULATION_ENABLED: True,
        ATTRIBUTE_REDACTION_ENABLED: True,
    }
    processor.set_metadata(fallback)
    provider.add_span_processor(processor)

    with _job_context(initialized=initialized), tracer.start_as_current_span("job_a"):
        pass

    assert exporter.get_finished_spans()[-1].attributes == {
        "langfuse.session.id": "customer-session",
        "job_id": "job-a",
        "room_id": "room-a",
        "gen_ai.conversation.id": "room-a",
    }

    with tracer.start_as_current_span("worker"):
        pass
    assert exporter.get_finished_spans()[-1].attributes == {
        "langfuse.session.id": "customer-session",
        **fallback,
    }

    processor.clear_metadata()
    with tracer.start_as_current_span("worker_after_cleanup"):
        pass
    assert exporter.get_finished_spans()[-1].attributes == {
        "langfuse.session.id": "customer-session",
        "job_id": "provider-job",
        "room_id": "provider-room",
    }


# -- resource --


@pytest.fixture
def detections(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    monkeypatch.setattr(traces, "_detected_resource", None)
    calls: list[int] = []
    create = Resource.create

    def _counting_create(*args, **kwargs) -> Resource:
        calls.append(1)
        return create(*args, **kwargs)

    monkeypatch.setattr(Resource, "create", staticmethod(_counting_create))
    return calls


def test_matches_resource_create_and_detects_once(detections: list[int]) -> None:
    attrs = {SERVICE_NAME: "livekit-agents", "room_id": "RM_x"}
    first = traces._resource(attrs)
    second = traces._resource({})
    assert len(detections) == 1

    detections.clear()
    assert first.attributes == Resource.create(attrs).attributes
    assert second.attributes == Resource.create({}).attributes


def test_detects_again_after_fork(detections: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    traces._resource({})
    monkeypatch.setattr(traces.os, "getpid", lambda: -1)
    traces._resource({})
    assert len(detections) == 2
