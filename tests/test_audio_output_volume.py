"""The room sink applies the output volume to the frames it hands to the audio source."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice import io
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

pytestmark = pytest.mark.unit

RATE = 48000


class _RecordingSource:
    """Stands in for rtc.AudioSource and keeps every frame it is given."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.queued_duration = 0.0
        self.frames: list[rtc.AudioFrame] = []

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)

    def clear_queue(self) -> None:
        pass

    async def wait_for_playout(self) -> None:
        pass

    def samples(self) -> np.ndarray:
        return np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in self.frames])


def _sink() -> tuple[_ParticipantAudioOutput, _RecordingSource]:
    with patch("livekit.agents.voice.room_io._output.rtc.AudioSource", _RecordingSource):
        sink = _ParticipantAudioOutput(
            MagicMock(), sample_rate=RATE, num_channels=1, track_publish_options=MagicMock()
        )
    sink._subscribed_fut.set_result(None)
    return sink, sink._audio_source  # type: ignore[return-value]


async def _play(sink: _ParticipantAudioOutput, seconds: float) -> None:
    """Capture a constant-amplitude segment and let the forwarding task hand it over.

    `seconds` is a whole number of 50 ms frames, so nothing waits in the byte stream.
    """
    n = round(seconds * RATE)
    task = asyncio.create_task(sink._forward_audio())
    await sink.capture_frame(rtc.AudioFrame(np.full(n, 10000, np.int16).tobytes(), RATE, 1, n))
    for _ in range(200):
        if sink._audio_buf.empty() and sink._forwarding_idle.is_set():
            break
        await asyncio.sleep(0)
    if sink._bstream_flush_timer:
        sink._bstream_flush_timer.cancel()
    await utils.aio.cancel_and_wait(task)


async def test_default_volume_leaves_audio_untouched() -> None:
    sink, source = _sink()
    await _play(sink, 0.1)

    assert np.all(source.samples() == 10000)


async def test_volume_without_ramp_scales_every_sample() -> None:
    sink, source = _sink()
    sink.set_volume(0.25, ramp_duration=0.0)
    await _play(sink, 0.1)

    assert np.all(source.samples() == 2500)


async def test_volume_ramps_to_the_target_without_a_step() -> None:
    sink, source = _sink()
    sink.set_volume(0.0, ramp_duration=0.01)  # 480 samples
    await _play(sink, 0.1)

    samples = source.samples().astype(np.int32)
    assert len(samples) == round(0.1 * RATE)
    assert 9900 < samples[0] < 10000
    assert np.all(np.diff(samples[:480]) <= 0)  # monotonic, no jump
    assert np.max(np.abs(np.diff(samples))) <= 10000 / 480 + 1
    assert np.all(samples[479:] == 0)


async def test_volume_changes_from_where_the_previous_ramp_is() -> None:
    sink, source = _sink()
    sink.set_volume(0.0, ramp_duration=0.0)
    sink.set_volume(1.0, ramp_duration=0.01)
    await _play(sink, 0.1)

    samples = source.samples()
    assert 0 < samples[0] < 100
    assert np.all(samples[479:] == 10000)


def test_ramp_reverses_mid_way_on_every_channel() -> None:
    with patch("livekit.agents.voice.room_io._output.rtc.AudioSource", _RecordingSource):
        sink = _ParticipantAudioOutput(
            MagicMock(), sample_rate=RATE, num_channels=2, track_publish_options=MagicMock()
        )
    n = 240
    frame = rtc.AudioFrame(np.tile(np.array([10000, -10000], np.int16), n).tobytes(), RATE, 2, n)

    sink.set_volume(0.0, ramp_duration=0.02)  # 960 samples, a quarter of it per frame
    down = np.frombuffer(sink._apply_volume(frame).data, np.int16).reshape(-1, 2)
    assert sink._volume == pytest.approx(0.75)
    assert np.all(down[:, 0] == -down[:, 1])  # the same gain on both channels

    sink.set_volume(1.0, ramp_duration=0.02)
    up = np.frombuffer(sink._apply_volume(frame).data, np.int16).reshape(-1, 2)
    assert abs(int(up[0, 0]) - int(down[-1, 0])) <= 10000 / 960 + 1  # no step at the turn
    assert np.all(np.diff(up[:, 0].astype(np.int32)) >= 0)


def test_non_finite_volume_is_rejected() -> None:
    sink, _ = _sink()
    with pytest.raises(ValueError):
        sink.set_volume(float("nan"))
    with pytest.raises(ValueError):
        sink.set_volume(float("inf"))


async def test_volume_is_forwarded_down_the_output_chain() -> None:
    sink, _ = _sink()
    proxy = io._AudioSinkProxy(sink)
    proxy.set_volume(0.5, ramp_duration=0.0)

    assert sink._volume == 0.5


def test_negative_volume_is_rejected() -> None:
    sink, _ = _sink()
    with pytest.raises(ValueError):
        sink.set_volume(-0.1)
