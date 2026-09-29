from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit import rtc
from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer
from livekit.agents.voice.generation import perform_tts_inference
from livekit.agents.voice.io import ModelSettings

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    original = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original)
        provider.shutdown()


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("finish", ["complete", "early", "cancel"])
async def test_tts_capture_observes_consumption_without_draining(
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    capture: bool,
    finish: str,
) -> None:
    monkeypatch.setattr(gen_ai, "_capture_content", capture)
    produced: list[str] = []
    consumed: list[str] = []
    first_consumed = asyncio.Event()

    async def source() -> AsyncIterable[str]:
        for chunk in ["one ", "two ", "three"]:
            produced.append(chunk)
            yield chunk

    async def uppercase(source: AsyncIterable[str]) -> AsyncIterable[str]:
        async for chunk in source:
            yield chunk.upper()

    async def node(
        text: AsyncIterable[str], settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        async for chunk in text:
            consumed.append(chunk)
            first_consumed.set()
            if finish == "cancel":
                await asyncio.Event().wait()
            # Give a background reader a chance to consume ahead of this node.
            await asyncio.sleep(0.01)
            yield rtc.AudioFrame.create(sample_rate=24000, num_channels=1, samples_per_channel=240)
            if finish == "early":
                break

    task, _ = perform_tts_inference(
        node=node,
        input=source(),
        model_settings=ModelSettings(),
        text_transforms=[uppercase],
        provider="google",
        model="test-tts",
    )
    if finish == "cancel":
        await asyncio.wait_for(first_consumed.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await task
    expected = ["one ", "two ", "three"] if finish == "complete" else ["one "]
    assert produced == expected
    assert consumed == [chunk.upper() for chunk in expected]
    [span] = [span for span in exporter.get_finished_spans() if span.name == "tts_node"]
    assert span.attributes["gen_ai.provider.name"] == "gcp.gen_ai"
    if capture:
        assert json.loads(span.attributes["gen_ai.input.messages"]) == [
            {"role": "assistant", "parts": [{"type": "text", "content": "".join(consumed)}]}
        ]
    else:
        assert "gen_ai.input.messages" not in span.attributes
