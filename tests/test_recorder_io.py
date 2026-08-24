"""RecorderIO places both channels on one absolute timeline.

The user's audio goes where it arrived, the agent's where the sink reported it played, and
whatever nothing was written over stays silent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice import io
from livekit.agents.voice.recorder_io import recorder_io as recorder_module
from livekit.agents.voice.recorder_io.recorder_io import (
    RecorderAudioOutput,
    RecorderIO,
    _Track,
)

pytestmark = pytest.mark.unit

RATE = 48000
_CLOCK = "livekit.agents.voice.recorder_io.recorder_io.time.time"


def _tone(
    seconds: float, *, sample_rate: int = RATE, channels: int = 1, level: int = 0
) -> rtc.AudioFrame:
    """Silence, or a 1kHz tone that survives the opus encoder."""
    n = round(seconds * sample_rate)
    if level:
        phase = 2 * np.pi * 1000 * np.arange(n) / sample_rate
        wave = (level * np.sin(phase)).astype(np.int16)
    else:
        wave = np.zeros(n, dtype=np.int16)
    data = np.repeat(wave, channels) if channels > 1 else wave
    return rtc.AudioFrame(data.tobytes(), sample_rate, channels, n)


# ---------------------------------------------------------------------------
# RecorderAudioOutput: audio goes where the sink says it played
# ---------------------------------------------------------------------------


def _placing_output() -> tuple[RecorderAudioOutput, list[tuple[float, rtc.AudioFrame]]]:
    placed: list[tuple[float, rtc.AudioFrame]] = []
    output = RecorderAudioOutput(
        recording_io=MagicMock(recording=True),
        audio_output=None,
        on_played=lambda started_at, frame: placed.append((started_at, frame)),
    )
    return output, placed


async def test_reports_place_audio_at_the_time_it_played() -> None:
    """A gap between two reports is a gap in the recording, not silence moved to the front."""
    output, placed = _placing_output()
    await output.capture_frame(_tone(1.0))
    output.flush()

    output.on_playback_progressed(started_at=100.0, offset=0.0, duration=0.4)
    output.on_playback_progressed(started_at=101.0, offset=0.4, duration=0.6)
    output.on_playback_finished(playback_position=1.0, interrupted=False)

    assert [t for t, _ in placed] == [100.0, 101.0]
    assert [round(f.duration, 3) for _, f in placed] == [0.4, 0.6]


async def test_a_report_skips_the_audio_that_never_played() -> None:
    """Audio dropped from the sink's queue is a hole in the middle, not a shortened end."""
    output, placed = _placing_output()
    await output.capture_frame(_tone(1.0))
    output.flush()

    # 0.2s of the segment was discarded on pause, so the next run resumes past it
    output.on_playback_progressed(started_at=100.0, offset=0.0, duration=0.3)
    output.on_playback_progressed(started_at=100.8, offset=0.5, duration=0.5)
    output.on_playback_finished(playback_position=0.8, interrupted=False)

    assert [t for t, _ in placed] == [100.0, 100.8]
    assert [round(f.duration, 3) for _, f in placed] == [0.3, 0.5]


async def test_a_sink_that_reports_nothing_is_anchored_at_playback_started() -> None:
    """The fallback is the shape a remote sink sends: one report for the whole segment."""
    output, placed = _placing_output()
    await output.capture_frame(_tone(1.0))
    output.flush()

    output.on_playback_started(created_at=100.0)
    now = 105.0  # noticed long after the audio really stopped
    with patch(_CLOCK, side_effect=lambda: now):
        output.on_playback_finished(playback_position=1.0, interrupted=False)

    assert len(placed) == 1
    started_at, frame = placed[0]
    assert started_at == 100.0  # not 105.0 - 1.0, which is where the old recorder put it
    assert round(frame.duration, 3) == 1.0


async def test_a_sink_with_no_start_of_its_own_falls_back_to_the_segment_end() -> None:
    """Last resort for a remote sink that never reports a start: date it from the finish."""
    output, placed = _placing_output()
    await output.capture_frame(_tone(1.0))
    output.flush()

    now = 105.0
    with patch(_CLOCK, side_effect=lambda: now):
        output.on_playback_finished(playback_position=1.0, interrupted=False)

    assert len(placed) == 1
    assert placed[0][0] == pytest.approx(104.0)


async def test_the_fallback_truncates_an_interrupted_segment() -> None:
    output, placed = _placing_output()
    await output.capture_frame(_tone(1.0))
    output.flush()

    output.on_playback_started(created_at=100.0)
    output.on_playback_finished(playback_position=0.4, interrupted=True)

    assert len(placed) == 1
    assert round(placed[0][1].duration, 3) == 0.4


async def test_a_segment_in_flight_holds_the_timeline() -> None:
    """The writer cannot settle a window whose agent audio has not been reported yet."""
    output, _ = _placing_output()
    assert output.pending_since is None

    await output.capture_frame(_tone(1.0))
    assert output.pending_since is not None

    output.on_playback_progressed(started_at=100.0, offset=0.0, duration=1.0)
    output.on_playback_finished(playback_position=1.0, interrupted=False)
    assert output.pending_since is None


# ---------------------------------------------------------------------------
# _Track: the absolute timeline
# ---------------------------------------------------------------------------


def _loud(samples: int, sample_rate: int = 1000, channels: int = 1) -> rtc.AudioFrame:
    data = np.full(samples * channels, 1000, dtype=np.int16)
    return rtc.AudioFrame(data.tobytes(), sample_rate, channels, samples)


def test_placed_audio_lands_at_its_own_timestamp() -> None:
    track = _Track(sample_rate=1000, t0=0.0)
    track.push(2.0, _loud(100))

    block = track.take(0, 3000)
    assert np.all(block[:2000] == 0.0)  # unwritten time is silence
    assert np.all(block[2000:2100] > 0.0)
    assert np.all(block[2100:] == 0.0)


def test_a_gap_between_runs_stays_a_gap() -> None:
    track = _Track(sample_rate=1000, t0=0.0)
    track.push(0.0, _loud(100))
    track.push(0.5, _loud(100))

    block = track.take(0, 1000)
    assert np.all(block[:100] > 0.0)
    assert np.all(block[100:500] == 0.0)
    assert np.all(block[500:600] > 0.0)


def test_audio_arriving_after_its_window_was_written_is_dropped_and_counted() -> None:
    track = _Track(sample_rate=1000, t0=0.0)
    track.take(0, 2000)  # the first two seconds have already gone to the encoder
    track.push(0.5, _loud(100))

    assert np.all(track.take(2000, 3000) == 0.0)
    assert track.dropped_samples == 100


def test_a_run_is_re_anchored_when_its_clock_drifts() -> None:
    track = _Track(sample_rate=1000, t0=0.0)
    track.push(0.0, _loud(100))
    track.push(0.1, _loud(100))  # contiguous, extends the run
    track.push(5.0, _loud(100))  # beyond tolerance, anchors on its own timestamp

    assert [pos for pos, _ in track._placed] == [0, 100, 5000]


def test_a_source_below_the_recording_rate_is_resampled_in_place() -> None:
    track = _Track(sample_rate=48000, t0=0.0)
    track.push(1.0, _loud(2400, sample_rate=24000))  # 100ms at 24kHz
    track.push(9.0, _loud(2400, sample_rate=24000))  # a new run, so the first one is complete

    block = track.take(0, 48000 * 2)
    assert np.all(block[:48000] == 0.0)
    # 100ms of audio, twice as many samples at the recording rate; the samples the resampler
    # still held when the run ended are part of it
    assert np.count_nonzero(block[48000:]) == pytest.approx(4800, abs=50)


def test_a_stereo_source_is_mixed_down() -> None:
    track = _Track(sample_rate=1000, t0=0.0)
    track.push(0.0, _loud(100, channels=2))

    block = track.take(0, 200)
    assert np.count_nonzero(block) == 100  # samples, not interleaved values
    assert block[0] == pytest.approx(1000 / 32768, abs=1e-4)


# ---------------------------------------------------------------------------
# RecorderIO end to end: a file whose channels sit where the audio happened
# ---------------------------------------------------------------------------


class _Source(io.AudioInput):
    """Hands over pre-made frames, one per pull."""

    def __init__(self) -> None:
        super().__init__(label="test")
        self.frames: list[rtc.AudioFrame] = []

    async def __anext__(self) -> rtc.AudioFrame:
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)


class _Sink(io.AudioOutput):
    """A leaf that accepts audio and reports nothing of its own."""

    def __init__(self) -> None:
        super().__init__(label="test", capabilities=io.AudioOutputCapabilities(pause=False))

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


def _decode(path: Path) -> tuple[np.ndarray, int]:
    with av.open(str(path)) as container:
        rate = container.streams.audio[0].rate
        blocks = [f.to_ndarray() for f in container.decode(audio=0)]
    pcm = np.concatenate(blocks, axis=1)
    return (pcm.reshape(-1, 2).T if pcm.shape[0] == 1 else pcm), rate


def _energy_bins(channel: np.ndarray, rate: int, width: float = 0.25) -> list[bool]:
    step = int(rate * width)
    return [
        bool(np.max(np.abs(channel[i : i + step])) > 0.01) for i in range(0, len(channel), step)
    ]


async def test_a_recording_matches_when_each_channel_happened(tmp_path: Path) -> None:
    """A stalled agent and a muted microphone both record as themselves.

    The agent speaks 0.5s, goes quiet for 0.5s, speaks 0.5s. The microphone delivers for the
    first second, then stops. Neither hole may be filled by moving the audio around it.
    """
    now = 1000.0
    path = tmp_path / "session.ogg"

    with patch(_CLOCK, side_effect=lambda: now):
        source, sink = _Source(), _Sink()
        recorder = RecorderIO(agent_session=MagicMock(), sample_rate=RATE)
        audio_in = recorder.record_input(source)
        audio_out = recorder.record_output(sink)
        await recorder.start(output_path=path)
        t0 = now

        # a microphone that delivers for one second, then goes quiet
        for _ in range(10):
            now += 0.1
            source.frames.append(_tone(0.1, level=8000))
            await audio_in.__anext__()

        # the agent speaks, stalls, then speaks again
        await audio_out.capture_frame(_tone(1.0, level=8000))
        audio_out.flush()
        audio_out.on_playback_started(created_at=t0 + 1.5)
        audio_out.on_playback_progressed(started_at=t0 + 1.5, offset=0.0, duration=0.5)
        audio_out.on_playback_progressed(started_at=t0 + 2.5, offset=0.5, duration=0.5)
        now = t0 + 3.0
        audio_out.on_playback_finished(playback_position=1.0, interrupted=False)

        await recorder.aclose()

    pcm, rate = _decode(path)
    assert pcm.shape[1] / rate == pytest.approx(3.0, abs=0.1)

    # the microphone's first second is there, and its silence is silence
    assert _energy_bins(pcm[0], rate) == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    # the agent's two utterances sit either side of the stall, not packed together
    assert _energy_bins(pcm[1], rate) == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


async def test_the_writer_waits_on_the_agent_but_not_on_a_quiet_source() -> None:
    """A segment in flight holds the timeline; a microphone that went quiet does not."""
    now = 1000.0
    with patch(_CLOCK, side_effect=lambda: now):
        recorder = RecorderIO(agent_session=MagicMock(), sample_rate=RATE)
        recorder.record_input(_Source())
        audio_out = recorder.record_output(_Sink())
        recorder._t0 = recorder._input_settled = now
        recorder._started = True

        flushed: list[float] = []
        recorder._q.put_nowait = lambda item: flushed.append(item.until)  # type: ignore[method-assign]

        with patch.object(recorder_module, "WRITE_INTERVAL", 0):
            task = asyncio.create_task(recorder._write_task())

            now += 5.0  # the source has delivered nothing for five seconds
            for _ in range(3):
                await asyncio.sleep(0)
            assert flushed[-1] == pytest.approx(now - recorder_module.INPUT_STALL_TIMEOUT)

            await audio_out.capture_frame(_tone(0.2))  # a segment opens
            segment_began = audio_out.pending_since
            now += 5.0
            for _ in range(3):
                await asyncio.sleep(0)
            assert flushed[-1] == pytest.approx(segment_began)

            task.cancel()
