"""A fully collected TTS request must close its span.

``ChunkedStream.collect()`` is the one-call way to synthesize a whole text; it is
expected to finish the request. The request span is ended in ``aclose`` rather than
in the streaming task, so collecting without closing left it open forever.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.telemetry import set_tracer_provider, tracer

from .fake_tts import FakeTTS

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


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


async def test_collect_finishes_the_tts_request_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    tts_impl = FakeTTS(fake_audio_duration=0.01)
    stream = tts_impl.synthesize("hello world")

    frame = await stream.collect()

    assert frame.duration > 0
    [request] = _spans(span_exporter, "tts_request")
    assert request.end_time is not None
    assert stream._tts_request_span is None
