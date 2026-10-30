from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Iterator
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.inference import vad as inference_vad

from .fake_stt import FakeSTT

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


def _window(value: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.full(_WINDOW_SAMPLES, value, dtype=np.int16).tobytes(),
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=_WINDOW_SAMPLES,
    )


async def _read_to_end(stream: AsyncIterable[Any]) -> list[Any]:
    async def read() -> list[Any]:
        return [event async for event in stream]

    return await asyncio.wait_for(read(), timeout=2.0)


async def _push_window(stream: vad.VADStream, value: int) -> list[vad.VADEvent]:
    stream.push_frame(_window(value))
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


async def test_end_input_closes_the_open_speech_segment(
    vad_impl: inference_vad.VAD | silero_vad.VAD,
) -> None:
    """Input ending mid-speech emits END_OF_SPEECH with that speech, then ends the stream."""
    stream = vad_impl.stream()
    try:
        for value in (1, 2, 3):
            await _push_window(stream, value)

        stream.end_input()
        events = await _read_to_end(stream)
    finally:
        await stream.aclose()

    assert [event.type for event in events] == [vad.VADEventType.END_OF_SPEECH]
    event = events[0]
    assert event.speaking is False
    assert event.samples_index == 3 * _WINDOW_SAMPLES
    assert event.silence_duration == 0.0
    assert event.speech_duration == pytest.approx(0.096)
    samples = np.repeat(np.arange(1, 4, dtype=np.int16), _WINDOW_SAMPLES)
    np.testing.assert_array_equal(event.frames[0].data, samples)


@pytest.mark.parametrize("sample_rate", [_SAMPLE_RATE, 48000])
async def test_end_input_keeps_the_samples_short_of_an_inference_window(
    vad_impl: inference_vad.VAD | silero_vad.VAD, sample_rate: int
) -> None:
    """Samples too few to run inference on are still part of the speech the input ended in."""
    window = sample_rate // 100  # 10ms
    samples = np.full(window * 16 + 95, 7, dtype=np.int16)
    stream = vad_impl.stream()
    try:
        for start in range(0, len(samples), window):
            chunk = samples[start : start + window]
            stream.push_frame(
                rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
            )
        stream.end_input()
        events = await _read_to_end(stream)
    finally:
        await stream.aclose()

    ends = [event for event in events if event.type == vad.VADEventType.END_OF_SPEECH]
    assert len(ends) == 1
    np.testing.assert_array_equal(ends[0].frames[0].data, samples)


async def test_end_input_excludes_trailing_silence_from_the_speech_duration(
    vad_impl: inference_vad.VAD | silero_vad.VAD,
) -> None:
    """Silence shorter than min_silence_duration counts as silence when input ends on it."""
    vad_impl.update_options(min_silence_duration=0.064)
    stream = vad_impl.stream()
    try:
        for value in (1, 2, 3, 0):
            await _push_window(stream, value)

        stream.end_input()
        events = await _read_to_end(stream)
    finally:
        await stream.aclose()

    assert [event.type for event in events] == [vad.VADEventType.END_OF_SPEECH]
    assert events[0].silence_duration == pytest.approx(0.032)
    assert events[0].speech_duration == pytest.approx(0.096)


async def test_end_input_adds_nothing_after_the_segment_already_closed(
    vad_impl: inference_vad.VAD | silero_vad.VAD,
) -> None:
    """A segment the VAD closed on silence is not closed a second time by the end of input."""
    stream = vad_impl.stream()
    try:
        await _push_window(stream, 1)
        await _push_window(stream, 2)
        await _push_window(stream, 0)
        event = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
        assert event.type == vad.VADEventType.END_OF_SPEECH

        stream.end_input()
        assert await _read_to_end(stream) == []
    finally:
        await stream.aclose()


async def test_flush_still_drops_the_open_segment_when_input_ends_after_it(
    vad_impl: inference_vad.VAD | silero_vad.VAD,
) -> None:
    """`flush()` stays a silent reset: only the end of input closes a segment with an event."""
    stream = vad_impl.stream()
    try:
        await _push_window(stream, 1)
        await _push_window(stream, 2)

        stream.flush()
        stream.end_input()
        assert await _read_to_end(stream) == []
    finally:
        await stream.aclose()


@pytest.mark.parametrize("flush_midway", [False, True])
async def test_stream_adapter_transcribes_speech_cut_off_by_the_end_of_input(
    vad_impl: inference_vad.VAD | silero_vad.VAD, flush_midway: bool
) -> None:
    """A non-streaming STT behind the adapter hears the utterance the input ended in."""
    stream = stt.StreamAdapter(stt=FakeSTT(fake_transcript="hello"), vad=vad_impl).stream()
    try:
        for value in (1, 2, 3):
            stream.push_frame(_window(value))
            if flush_midway and value == 2:
                stream.flush()

        stream.end_input()
        events = await _read_to_end(stream)
    finally:
        await stream.aclose()

    assert [event.type for event in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert events[2].alternatives[0].text == "hello"
