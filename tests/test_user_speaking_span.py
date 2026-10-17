"""The ``user_speaking`` span never ends before it starts.

Its start comes from the STT turn anchor and its end from a VAD anchor that is backdated by the
silence the VAD waited on. The two detectors do not share a clock, so on a short burst the end
lands first. OpenTelemetry keeps the duration unsigned, so a backend that subtracts a negative
one reads 2^64 minus the gap (#7307)."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import AgentSession
from livekit.agents.telemetry import set_tracer_provider, tracer

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


def _user_speaking(exporter: InMemorySpanExporter) -> ReadableSpan:
    [span] = [s for s in exporter.get_finished_spans() if s.name == "user_speaking"]
    return span


async def test_backdated_end_is_clamped_to_the_start(span_exporter: InMemorySpanExporter) -> None:
    session = AgentSession()
    started_at = time.time()
    session._update_user_state("speaking", last_speaking_time=started_at)
    # a VAD end-of-speech anchor, backdated past the STT start by the silence it waited on
    session._update_user_state("listening", last_speaking_time=started_at - 0.55)

    span = _user_speaking(span_exporter)
    assert span.end_time == span.start_time


async def test_end_after_the_start_is_kept(span_exporter: InMemorySpanExporter) -> None:
    session = AgentSession()
    started_at = time.time()
    session._update_user_state("speaking", last_speaking_time=started_at)
    session._update_user_state("listening", last_speaking_time=started_at + 1.25)

    span = _user_speaking(span_exporter)
    assert span.end_time - span.start_time == pytest.approx(1.25e9, rel=1e-6)
