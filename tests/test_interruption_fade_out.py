"""Unit tests for the interruption fade-out frame builder.

``_build_fade_out_frame`` rebuilds a short faded tail from the rolling history of
frames already handed to the ``rtc.AudioSource``: on interruption the playback
position is ``end_of_history - unplayed_duration``, and a linear gain ramp
(1 -> 0) is applied to the next ``fade_out_duration`` seconds from that position.

These tests are hermetic — the builder tests exercise the pure function, and the
playout tests drive ``_ParticipantAudioOutput`` against a fake audio source; no
room involved.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.room_io._output import (
    _build_fade_out_frame,
    _ParticipantAudioOutput,
)

pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000


def _make_frame(value: int, duration: float, *, num_channels: int = 1) -> rtc.AudioFrame:
    samples = int(duration * SAMPLE_RATE)
    data = np.full(samples * num_channels, value, dtype=np.int16)
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=num_channels,
        samples_per_channel=samples,
    )


def _frame_samples(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16)


def test_linear_ramp_from_playback_position() -> None:
    # 500ms of history, 200ms unplayed, 80ms fade
    history = [_make_frame(10000, 0.05) for _ in range(10)]
    frame = _build_fade_out_frame(
        history,
        unplayed_duration=0.2,
        fade_out_duration=0.08,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )

    assert frame is not None
    expected_samples = int(0.08 * SAMPLE_RATE)
    assert frame.samples_per_channel == expected_samples

    samples = _frame_samples(frame)
    # ramp starts at full gain and decays monotonically to (near) zero
    assert samples[0] == 10000
    assert samples[-1] <= 10000 * (1 / expected_samples) + 1
    assert np.all(np.diff(samples) <= 0)


def test_fade_sourced_from_correct_offset() -> None:
    # history: 100ms of "1000" followed by 100ms of "2000"; the unplayed part is
    # exactly the second half, so the fade must be built from the "2000" segment
    history = [_make_frame(1000, 0.1), _make_frame(2000, 0.1)]
    frame = _build_fade_out_frame(
        history,
        unplayed_duration=0.1,
        fade_out_duration=0.05,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )

    assert frame is not None
    samples = _frame_samples(frame)
    gain = np.linspace(1.0, 0.0, num=len(samples), endpoint=False)
    np.testing.assert_allclose(samples, (2000 * gain).astype(np.int16), atol=1)


def test_fade_clamped_to_unplayed_duration() -> None:
    # only 30ms left unplayed: the fade cannot resurrect more audio than that
    history = [_make_frame(10000, 0.05) for _ in range(4)]
    frame = _build_fade_out_frame(
        history,
        unplayed_duration=0.03,
        fade_out_duration=0.08,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )

    assert frame is not None
    assert frame.samples_per_channel == int(0.03 * SAMPLE_RATE)


def test_unplayed_longer_than_history_is_clamped() -> None:
    # queued_duration may slightly exceed the recorded history; clamp instead of failing
    history = [_make_frame(10000, 0.05)]
    frame = _build_fade_out_frame(
        history,
        unplayed_duration=1.0,
        fade_out_duration=0.08,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )

    assert frame is not None
    assert frame.samples_per_channel <= int(0.05 * SAMPLE_RATE)


def test_stereo_gain_applied_per_sample_frame() -> None:
    history = [_make_frame(10000, 0.1, num_channels=2)]
    frame = _build_fade_out_frame(
        history,
        unplayed_duration=0.1,
        fade_out_duration=0.05,
        sample_rate=SAMPLE_RATE,
        num_channels=2,
    )

    assert frame is not None
    assert frame.samples_per_channel == int(0.05 * SAMPLE_RATE)
    samples = _frame_samples(frame).reshape(-1, 2)
    # both channels must carry the identical ramp
    np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
    assert samples[0, 0] == 10000
    assert np.all(np.diff(samples[:, 0]) <= 0)


@pytest.mark.parametrize(
    ("unplayed", "fade", "history_frames"),
    [
        (0.0, 0.08, 4),  # everything already played out
        (0.2, 0.0, 4),  # fade disabled
        (0.2, 0.08, 0),  # no history recorded
    ],
)
def test_no_frame_when_nothing_to_fade(unplayed: float, fade: float, history_frames: int) -> None:
    history = [_make_frame(10000, 0.05) for _ in range(history_frames)]
    frame = _build_fade_out_frame(
        history,
        unplayed_duration=unplayed,
        fade_out_duration=fade,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )
    assert frame is None


class _FakeAudioSource:
    """Stand-in for rtc.AudioSource: records captures, supports a held capture."""

    def __init__(self) -> None:
        self.queued_duration: float = 0.0
        self.captured: list[rtc.AudioFrame] = []
        self.clear_queue_calls = 0
        self.hold_capture: asyncio.Event | None = None
        self.capture_reached = asyncio.Event()

    def clear_queue(self) -> None:
        self.clear_queue_calls += 1
        self.queued_duration = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.capture_reached.set()
        if self.hold_capture is not None:
            await self.hold_capture.wait()
        self.captured.append(frame)
        self.queued_duration += frame.duration

    async def wait_for_playout(self) -> None:
        return


def _make_output(
    fade_out_duration: float,
) -> tuple[_ParticipantAudioOutput, _FakeAudioSource, MagicMock]:
    out = _ParticipantAudioOutput.__new__(_ParticipantAudioOutput)
    src = _FakeAudioSource()
    out._audio_source = src  # type: ignore[assignment]
    out._source_sample_rate = SAMPLE_RATE
    out._num_channels = 1
    out._fade_out_duration = fade_out_duration
    out._fade_history = deque()
    out._fade_history_duration = 0.0
    out._fade_history_max_duration = 0.2 + fade_out_duration + 0.1
    out._fade_tail_until = 0.0
    out._audio_buf = utils.aio.Chan()
    out._flush_task = None
    out._interrupted_event = asyncio.Event()
    out._forwarding_task = None
    out._pushed_duration = 0.0
    out._source_pushed_duration = 0.0
    out._source_discarded_duration = 0.0
    out._interruption_generation = 0
    out._playback_enabled = asyncio.Event()
    out._playback_enabled.set()
    out._forwarding_idle = asyncio.Event()
    out._forwarding_idle.set()
    out._first_frame_event = asyncio.Event()
    finished = MagicMock()
    out.on_playback_finished = finished  # type: ignore[method-assign]
    out.on_playback_started = MagicMock()  # type: ignore[method-assign]
    return out, src, finished


@pytest.mark.asyncio
async def test_cancelled_fade_push_still_finishes_playout() -> None:
    """A cancellation landing on the fade capture (overlapping flush()/aclose())
    must not skip the state reset and on_playback_finished: _forward_audio keeps
    discarding frames while _interrupted_event stays set, and the speech handle
    waits on the finish notification."""
    out, src, finished = _make_output(0.5)
    out._fade_history.extend(_make_frame(1000, 0.1) for _ in range(10))
    out._fade_history_duration = 1.0
    out._pushed_duration = 2.0
    out._source_pushed_duration = 2.0
    src.queued_duration = 0.3
    src.hold_capture = asyncio.Event()  # never set: capture blocks like a full source

    out._interrupted_event.set()
    task = asyncio.create_task(out._wait_for_playout())
    await asyncio.wait_for(src.capture_reached.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    finished.assert_called_once()
    kwargs = finished.call_args.kwargs
    assert kwargs["interrupted"] is True
    assert kwargs["playback_position"] == pytest.approx(1.7)
    assert not out._interrupted_event.is_set()
    # unknown whether the tail landed; the deadline must not shave the next segment
    assert out._fade_tail_until == 0.0


@pytest.mark.asyncio
async def test_leftover_fade_tail_excluded_from_playback_position() -> None:
    """A previous segment's fade tail still draining in the source was never
    counted as pushed; subtracting it with the queued audio would under-report
    this segment's playback position by up to the fade duration."""
    out, src, finished = _make_output(0.5)  # fade enabled, but empty history -> no new tail
    out._pushed_duration = 2.0
    out._source_pushed_duration = 2.0
    src.queued_duration = 0.7  # 0.3s of this segment + 0.4s fade leftover
    out._fade_tail_until = time.time() + 0.4

    out._interrupted_event.set()
    await out._wait_for_playout()

    kwargs = finished.call_args.kwargs
    assert kwargs["interrupted"] is True
    assert kwargs["playback_position"] == pytest.approx(1.7, abs=0.05)


@pytest.mark.asyncio
async def test_pause_discard_excludes_leftover_fade_tail() -> None:
    """The pause path adds queued_duration to _source_discarded_duration; the
    leftover fade tail in that queue belongs to the previous segment and must
    not shorten this segment's playback position."""
    out, src, _ = _make_output(0.5)
    out._pushed_duration = 1.0  # segment in progress
    src.queued_duration = 0.5  # 0.1s of this segment + 0.4s fade leftover
    out._fade_tail_until = time.time() + 0.4
    out._playback_enabled.clear()  # paused

    forward = asyncio.create_task(out._forward_audio())
    out._audio_buf.send_nowait(_make_frame(1000, 0.05))
    await asyncio.sleep(0.1)

    assert out._source_discarded_duration == pytest.approx(0.1, abs=0.06)
    assert out._fade_tail_until == 0.0

    out._playback_enabled.set()
    await asyncio.sleep(0.1)
    assert src.captured  # the held frame resumed into the source

    out._audio_buf.close()
    await forward
