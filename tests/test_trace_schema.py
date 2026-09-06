"""The trace-shape checker itself: the rules catch the mistakes they exist for, both span
sources agree, and a full fake session passes clean."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import Agent
from livekit.agents.telemetry import set_tracer_provider, tracer

from .fake_session import FakeActions, create_session, run_session
from .trace_schema import (
    ANY,
    MAY_OUTLIVE_PARENT,
    SPAN_PARENTS,
    SpanRecord,
    assert_trace_well_formed,
    check_trace,
    from_otlp_json,
    from_readable_spans,
)

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


def _span(
    name: str,
    span_id: str,
    parent: str | None,
    start: float,
    end: float,
    **attributes: object,
) -> SpanRecord:
    events = attributes.pop("events", [])
    return SpanRecord(
        name=name,
        span_id=span_id,
        parent_id=parent,
        trace_id="t1",
        start_ns=int(start * 1e9),
        end_ns=int(end * 1e9),
        attributes=dict(attributes),
        events=list(events),  # type: ignore[arg-type]
    )


def _sound_trace() -> list[SpanRecord]:
    return [
        _span("job_entrypoint", "j", None, 0.0, 30.0),
        _span("agent_session", "s", "j", 1.0, 28.0),
        _span("user_turn", "u", "s", 5.0, 7.0),
        _span("eou_wait", "w", "u", 6.5, 7.0, **{"lk.eou.outcome": "committed"}),
        _span("eou_detection", "d", "w", 6.8, 6.9),
        _span("on_user_turn_completed", "h", "u", 6.95, 7.0),
        _span(
            "agent_turn",
            "a",
            "s",
            7.0,
            12.0,
            **{
                "lk.speech_id": "speech_1",
                "lk.generation_count": 2,
                "events": ["generation", "generation"],
            },
        ),
        _span("llm_node", "l", "a", 7.0, 8.0),
        _span("llm_request", "r", "l", 7.0, 8.0),
        _span("function_tool", "f", "a", 8.0, 8.1),
        _span("job_shutdown", "x", "j", 28.0, 30.0),
    ]


def test_schema_is_self_consistent() -> None:
    # every parent named in the rules is itself a known span (or ROOT / ANY)
    for name, parents in SPAN_PARENTS.items():
        for p in parents:
            assert p is None or p == ANY or p in SPAN_PARENTS, f"{name}: unknown parent {p}"
    for child, parent in MAY_OUTLIVE_PARENT:
        assert child in SPAN_PARENTS
        assert parent == ANY or parent in SPAN_PARENTS


def test_sound_trace_has_no_violations() -> None:
    assert check_trace(_sound_trace()) == []


def test_wrong_parent_is_reported() -> None:
    spans = _sound_trace()
    # the keyterm-detection style mistake: an llm_request straight under agent_turn
    spans.append(_span("llm_request", "k", "a", 7.1, 7.9))
    [v] = check_trace(spans)
    assert v.startswith("llm_request: parent is agent_turn")

    # eou_detection outside its wait
    spans = _sound_trace()
    spans.append(_span("eou_detection", "d2", "u", 6.0, 6.1))
    assert any(v.startswith("eou_detection: parent is user_turn") for v in check_trace(spans))


def test_unknown_span_and_missing_parent_are_reported() -> None:
    spans = _sound_trace() + [_span("mystery", "m", "s", 2.0, 3.0)]
    assert any("mystery: unknown span" in v for v in check_trace(spans))

    spans = _sound_trace() + [_span("user_speaking", "p", "gone", 2.0, 3.0)]
    assert any("parent gone is not in the trace" in v for v in check_trace(spans))
    # a partial export (a view keyed to one span drops the ancestors): the orphan's edge is
    # simply not checked
    assert check_trace(spans, allow_missing_parents=True) == []


def test_bounds_are_checked_except_where_deliberately_allowed() -> None:
    spans = _sound_trace()
    spans.append(_span("tts_node", "t", "a", 11.0, 12.5))  # ends after agent_turn
    assert any(
        v.startswith("tts_node: ends 500.0 ms after its parent agent_turn")
        for v in check_trace(spans)
    )

    spans = _sound_trace()
    spans.append(_span("user_speaking", "sp", "u", 4.0, 6.0))  # starts before user_turn
    assert any(v.startswith("user_speaking: starts 1000.0 ms before") for v in check_trace(spans))

    # session.start() returning before the participant is linked is a known shape
    spans = _sound_trace()
    spans.append(_span("session_start", "ss", "s", 1.0, 2.0))
    spans.append(_span("wait_for_participant", "wp", "ss", 2.0, 4.0))
    assert check_trace(spans) == []
    # and a stall's end is one tick late by construction, whatever it is under
    spans.append(_span("event_loop_blocked", "b", "f", 8.05, 8.15))
    assert check_trace(spans) == []


def test_turn_invariants_are_checked() -> None:
    spans = _sound_trace()
    spans.append(
        _span(
            "agent_turn",
            "a2",
            "s",
            13.0,
            14.0,
            **{"lk.speech_id": "speech_1", "lk.generation_count": 1, "events": ["generation"]},
        )
    )
    assert any("speech speech_1 has 2 turns" in v for v in check_trace(spans))

    spans = _sound_trace()
    spans[6].attributes["lk.generation_count"] = 3
    assert any("lk.generation_count=3 but 2 generation events" in v for v in check_trace(spans))

    spans = _sound_trace()
    del spans[3].attributes["lk.eou.outcome"]
    assert "eou_wait: no lk.eou.outcome" in check_trace(spans)

    spans = _sound_trace()
    spans[1].trace_id = "t2"
    assert any("2 traces" in v for v in check_trace(spans))


def test_otlp_json_and_readable_spans_agree(span_exporter: InMemorySpanExporter) -> None:
    with tracer.start_as_current_span("agent_session") as root:
        with tracer.start_as_current_span("user_turn", attributes={"lk.speech_id": "x"}) as turn:
            turn.add_event("generation")
    readable = from_readable_spans(span_exporter.get_finished_spans())

    def hex_id(value: int, width: int) -> str:
        return format(value, f"0{width}x")

    document = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": s.name,
                                "spanId": s.span_id,
                                "parentSpanId": s.parent_id or "",
                                "traceId": s.trace_id,
                                "startTimeUnixNano": str(s.start_ns),
                                "endTimeUnixNano": str(s.end_ns),
                                "attributes": [
                                    {"key": k, "value": {"stringValue": str(v)}}
                                    for k, v in s.attributes.items()
                                ],
                                "events": [{"name": e} for e in s.events],
                            }
                            for s in readable
                        ]
                    }
                ]
            }
        ]
    }
    converted = from_otlp_json(document)
    assert [(s.name, s.parent_id, s.events) for s in converted] == [
        (s.name, s.parent_id, s.events) for s in readable
    ]
    [turn_record] = [s for s in converted if s.name == "user_turn"]
    assert turn_record.parent_id == hex_id(root.get_span_context().span_id, 16)
    assert check_trace(converted) == check_trace(readable) == []


async def test_full_fake_session_is_well_formed(span_exporter: InMemorySpanExporter) -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello there", stt_delay=0.1)
    actions.add_llm("Hi!", ttft=0.1, duration=0.2)
    actions.add_tts(0.5, ttfb=0.1, duration=0.2)
    session = create_session(actions, speed_factor=2.0)
    await asyncio.wait_for(
        run_session(session, Agent(instructions="t"), drain_delay=1.0), timeout=60
    )
    assert_trace_well_formed(span_exporter.get_finished_spans())
