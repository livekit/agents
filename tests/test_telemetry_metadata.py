from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.job import _JobContextVar
from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.telemetry.traces import _JobTelemetry

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
def _job_context(
    *,
    initialized: bool = True,
    traces_enabled: bool = True,
    job_id: str = "job-a",
    room_id: str = "room-a",
) -> Iterator[None]:
    ctx = SimpleNamespace(
        job=SimpleNamespace(id=job_id, room=SimpleNamespace(sid=room_id)),
        _telemetry_state=(
            _JobTelemetry(
                attributes={"job_id": job_id, "room_id": room_id},
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
        expected = {
            "langfuse.session.id": "customer-session",
            "job_id": "provider-job" if state == "outside" else "job-a",
            "room_id": "provider-room" if state == "outside" else "room-a",
            **({"gen_ai.conversation.id": "room-a"} if state != "outside" else {}),
        }
        assert span.attributes is not None
        for key, value in expected.items():
            assert span.attributes[key] == value
        if state == "outside":
            assert "gen_ai.conversation.id" not in span.attributes


def test_span_identity_is_captured_before_it_ends_in_another_job(
    span_exporter: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    _, exporter = span_exporter
    with _job_context(job_id="creator-job", room_id="creator-room"):
        created = tracer.start_span("created-in-first-job")

    with _job_context(job_id="exporting-job", room_id="exporting-room"):
        created.end()
        tracer.start_span("created-in-second-job").end()

    first, second = exporter.get_finished_spans()
    assert first.attributes is not None and second.attributes is not None
    assert first.attributes["job_id"] == "creator-job"
    assert first.attributes["room_id"] == "creator-room"
    assert first.attributes["gen_ai.conversation.id"] == "creator-room"
    assert second.attributes["job_id"] == "exporting-job"
    assert second.attributes["room_id"] == "exporting-room"
    assert second.attributes["gen_ai.conversation.id"] == "exporting-room"
    assert first.attributes["langfuse.session.id"] == "customer-session"
    assert second.attributes["langfuse.session.id"] == "customer-session"
