from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import vad
from livekit.agents.inference import vad as inference_vad

if TYPE_CHECKING:
    from livekit.plugins.silero import vad as silero_vad

pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000
_WINDOW_SAMPLES = 512


class _Model:
    sample_rate = _SAMPLE_RATE
    window_size_samples = _WINDOW_SAMPLES

    def predict(self, samples: np.ndarray) -> float:
        return float(np.any(samples))

    __call__ = predict

    def reset(self) -> None:
        pass


@pytest.fixture(params=["inference", "silero"])
def vad_impl(request: pytest.FixtureRequest) -> Iterator[inference_vad.VAD | silero_vad.VAD]:
    options = {
        "min_speech_duration": 0.032,
        "min_silence_duration": 0.032,
        "prefix_padding_duration": 0.064,
        "max_buffered_speech": 0.256,
        "activation_threshold": 0.5,
        "deactivation_threshold": 0.4,
    }
    if request.param == "inference":
        with patch.object(inference_vad, "_NativeVAD", _Model):
            yield inference_vad.VAD(**options)
    else:
        silero_vad = pytest.importorskip("livekit.plugins.silero.vad", exc_type=ModuleNotFoundError)
        with (
            patch.object(silero_vad.onnx_model, "new_inference_session"),
            patch.object(silero_vad.onnx_model, "OnnxModel", side_effect=lambda **_: _Model()),
        ):
            yield silero_vad.VAD.load(**options)


async def _push_window(stream: vad.VADStream, value: int) -> list[vad.VADEvent]:
    stream.push_frame(
        rtc.AudioFrame(
            data=np.full(_WINDOW_SAMPLES, value, dtype=np.int16).tobytes(),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=_WINDOW_SAMPLES,
        )
    )
    events = []
    while True:
        event = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
        events.append(event)
        if event.type == vad.VADEventType.INFERENCE_DONE:
            return events


@pytest.mark.parametrize("option", ["max_buffered_speech", "prefix_padding_duration"])
@pytest.mark.parametrize("resize", ["shrink", "grow", "shrink_then_grow"])
async def test_resize_buffer_during_speech(
    vad_impl: inference_vad.VAD | silero_vad.VAD, option: str, resize: str
) -> None:
    stream = vad_impl.stream()
    try:
        # Fill beyond the capacity that will remain after either update.
        for value in range(1, 10):
            await _push_window(stream, value)

        samples = np.repeat(np.arange(1, 10, dtype=np.int16), _WINDOW_SAMPLES)
        if resize != "grow":
            vad_impl.update_options(**{option: 0.016})
            retained = int((0.080 if option == "max_buffered_speech" else 0.272) * _SAMPLE_RATE)
            samples = samples[:retained]
        if resize != "shrink":
            vad_impl.update_options(**{option: 0.512})
            samples = np.concatenate([samples, np.zeros(_WINDOW_SAMPLES, dtype=np.int16)])

        await _push_window(stream, 0)
        event = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
        assert event.type == vad.VADEventType.END_OF_SPEECH
        assert event.frames[0].samples_per_channel == len(samples)
        np.testing.assert_array_equal(event.frames[0].data, samples)

        # Both the next speech segment and an explicit reset must remain usable.
        for reset in (False, True):
            if reset:
                stream.flush()
            await _push_window(stream, 10)
            events = await _push_window(stream, 0)
            assert any(event.type == vad.VADEventType.START_OF_SPEECH for event in events)
            event = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
            assert event.type == vad.VADEventType.END_OF_SPEECH
            assert event.frames[0].samples_per_channel == len(event.frames[0].data)
            assert 10 in event.frames[0].data
    finally:
        await stream.aclose()
