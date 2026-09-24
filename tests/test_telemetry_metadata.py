from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from livekit.agents.job import _JobContextVar
from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.telemetry.traces import _JobTelemetry, _MetadataSpanProcessor
from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED, ATTRIBUTE_SIMULATION_ENABLED

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def span_exporter() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
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
    span_exporter: tuple[TracerProvider, InMemorySpanExporter], state: str
) -> None:
    _, exporter = span_exporter
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
        }


@pytest.mark.parametrize("initialized", [False, True])
def test_job_fallback_metadata_does_not_cross_jobs(
    span_exporter: tuple[TracerProvider, InMemorySpanExporter], initialized: bool
) -> None:
    provider, exporter = span_exporter
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
