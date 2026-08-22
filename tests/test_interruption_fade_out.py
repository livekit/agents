"""Unit tests for the interruption fade-out frame builder.

``_build_fade_out_frame`` rebuilds a short faded tail from the rolling history of
frames already handed to the ``rtc.AudioSource``: on interruption the playback
position is ``end_of_history - unplayed_duration``, and a linear gain ramp
(1 -> 0) is applied to the next ``fade_out_duration`` seconds from that position.

These tests are hermetic — they exercise the pure builder only, no room or
audio source involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.voice.room_io._output import _build_fade_out_frame

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
