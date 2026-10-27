"""The session's span records, as ``VoiceSession`` materialises them with the OTel SDK: each
``span_started`` opens a span under the parent the core named (the root under the ambient
context), each ``span_ended`` adds its attributes and events, sets the error status, records its
duration metrics and ends it at the core's time."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import VoiceSession
from livekit.agents.telemetry import set_tracer_provider

pytestmark = pytest.mark.unit


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def started(span_id: str, parent: str | None, name: str, at: float, **attrs: Any) -> dict:
    return {
        "type": "span_started",
        "span_id": span_id,
        "parent_id": parent,
        "name": name,
        "start_time": at,
        "attributes": attrs,
    }


def ended(span_id: str, at: float, **extra: Any) -> dict:
    return {
        "type": "span_ended",
        "span_id": span_id,
        "end_time": at,
        "attributes": extra.get("attributes", {}),
        "error": extra.get("error"),
        "events": extra.get("events", []),
        "metrics": extra.get("metrics", []),
    }


def test_records_become_a_tree_of_spans_at_the_cores_times(
    exporter: InMemorySpanExporter,
) -> None:
    session = VoiceSession()
    session._core_event(
        started("agent_session", None, "agent_session", 10.0, **{"lk.agent_label": "a"})
    )
    assert session._root_span_context is not None, "the job's spans need the session's root"
    session._core_event(started("agent_turn_2", "agent_session", "agent_turn", 11.0))
    generation = {"name": "generation", "time": 11.0, "attributes": {"lk.generation_id": "s_1"}}
    session._core_event(
        ended(
            "agent_turn_2",
            12.5,
            attributes={"lk.generation_count": 1},
            error="the speech failed",
            events=[generation],
            metrics=[
                {"instrument": "gen_ai.invoke_agent.duration", "value": 1.5, "attributes": {}}
            ],
        )
    )
    # an id the session no longer holds is ignored, not raised
    session._core_event(ended("agent_turn_2", 13.0))
    session._core_event(ended("agent_session", 14.0))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    root, turn = spans["agent_session"], spans["agent_turn"]
    assert root.parent is None
    assert turn.parent is not None and turn.parent.span_id == root.context.span_id
    assert (turn.start_time, turn.end_time) == (11_000_000_000, 12_500_000_000)
    assert turn.attributes["lk.generation_count"] == 1
    assert turn.status.status_code == trace.StatusCode.ERROR
    assert turn.status.description == "the speech failed"
    assert [e.name for e in turn.events] == ["generation"]
    assert turn.events[0].timestamp == 11_000_000_000


def test_a_loop_stall_marks_the_session_span(exporter: InMemorySpanExporter) -> None:
    session = VoiceSession()
    session._core_event(started("agent_session", None, "agent_session", 10.0))
    session._record_loop_stall(0.2, timestamp_ns=10_500_000_000)
    session._record_loop_stall(0.4, timestamp_ns=10_900_000_000)
    session._core_event(ended("agent_session", 11.0))

    (root,) = exporter.get_finished_spans()
    assert [e.name for e in root.events] == ["event_loop_blocked", "event_loop_blocked"]
    assert root.attributes["lk.blocking.count"] == 2
    assert root.attributes["lk.blocking.total_duration"] == pytest.approx(0.6)
    assert root.attributes["lk.blocking.max_duration"] == 0.4
