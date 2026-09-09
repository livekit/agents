"""Tests for model attribution in TTS stream metrics."""

from __future__ import annotations

import pytest

from livekit.agents.metrics import TTSMetrics

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class _MutableModelTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__(fake_audio_duration=0.1)
        self.model_name = "model-a"

    @property
    def model(self) -> str:
        return self.model_name


@pytest.mark.parametrize("new_stream_first", [False, True], ids=["old-first", "new-first"])
async def test_stream_metrics_use_current_model_by_default(new_stream_first: bool) -> None:
    tts = _MutableModelTTS()
    metrics: list[TTSMetrics] = []
    tts.on("metrics_collected", metrics.append)

    try:
        async with tts.stream() as old_stream:
            old_stream.push_text("hello")
            tts.model_name = "model-b"

            async with tts.stream() as new_stream:
                new_stream.push_text("hello")
                # Providers can select the model after stream creation. Keep the default
                # metrics behavior dynamic; providers with fixed options can override it.
                streams = [old_stream, new_stream]
                if new_stream_first:
                    streams.reverse()

                for count, stream in enumerate(streams, start=1):
                    stream.end_input()
                    audio = [event async for event in stream]

                    assert audio
                    assert len(metrics) == count
                    assert metrics[-1].request_id == audio[0].request_id
                    assert metrics[-1].metadata.model_name == "model-b"
    finally:
        await tts.aclose()
