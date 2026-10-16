"""Unit tests for the STT pipeline's pushed-audio → wall-clock mapping.

Provider timestamps are positions on the pushed-audio timeline: they only advance
while frames flow, so the difference between a position and the wall clock grows
by the length of every gap in the mic audio. ``_STTPipeline`` records the lag of
each plateau (how late that audio arrived relative to its position) and
``wall_time`` adds the plateau a position falls in, which keeps audio captured
before a gap at its wall time, shifts everything after the gap by the gap, and
never lets audio pushed later move a timestamp from earlier.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice import audio_recognition
from livekit.agents.voice.audio_recognition import _STTPipeline

pytestmark = pytest.mark.unit


class _FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _frame(duration: float = 0.1) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(
        sample_rate=16000, num_channels=1, samples_per_channel=int(16000 * duration)
    )


def _make_pipeline(anchor: float) -> _STTPipeline:
    """A real pipeline carrying only the state ``push_frame`` touches (no pump task)."""
    pipeline = _STTPipeline.__new__(_STTPipeline)
    pipeline._audio_ch = MagicMock()  # type: ignore[attr-defined]
    pipeline.input_started_at = anchor
    pipeline.pushed_duration = 0.0
    pipeline._arrival_lags = []  # type: ignore[attr-defined]
    return pipeline


def _push(pipeline: _STTPipeline, clock: _FakeClock, count: int) -> None:
    """Push ``count`` frames in real time, one frame duration apart."""
    for _ in range(count):
        pipeline.push_frame(_frame())
        clock.now += 0.1


async def _empty_node(audio: Any, model_settings: Any) -> None:
    return None


async def test_anchor_is_stamped_before_the_first_frame() -> None:
    """A pipeline created mid-session (a handoff) must have its anchor already: the
    node seeds the stream's ``start_time_offset`` from it, and without it the offset
    falls back to the session start and the session time is counted twice."""
    pipeline = _STTPipeline(_empty_node)  # type: ignore[arg-type]
    try:
        assert pipeline.input_started_at > 0.0
        # no audio pushed yet: a position maps straight onto the anchor, like a
        # stream whose node reports timings without feeding the input
        assert pipeline.wall_time(0.0) == pipeline.input_started_at
    finally:
        await pipeline.aclose()


def test_wall_time_before_the_first_frame_follows_the_anchor() -> None:
    pipeline = _make_pipeline(anchor=999.9)
    assert pipeline.wall_time(0.25) == pytest.approx(1000.15)


def test_steady_stream_maps_audio_positions_to_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(audio_recognition.time, "time", clock)
    pipeline = _make_pipeline(anchor=999.9)

    _push(pipeline, clock, 3)

    assert pipeline.wall_time(0.05) == pytest.approx(999.95)
    assert pipeline.wall_time(0.25) == pytest.approx(1000.15)
    # a position past the pushed audio keeps the last plateau
    assert pipeline.wall_time(0.35) == pytest.approx(1000.25)


def test_gap_shifts_only_the_audio_after_it(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(audio_recognition.time, "time", clock)
    pipeline = _make_pipeline(anchor=999.9)

    _push(pipeline, clock, 3)  # 0.3 s of audio, positions 0.0–0.3
    clock.now += 5.0  # the mic goes quiet for 5 s: no frames are pushed
    _push(pipeline, clock, 3)  # positions 0.3–0.6, captured 5 s later

    assert pipeline.wall_time(0.25) == pytest.approx(1000.15)  # before the gap
    assert pipeline.wall_time(0.35) == pytest.approx(1005.25)  # after it
    assert pipeline.wall_time(0.55) == pytest.approx(1005.45)


def test_silence_flush_does_not_move_earlier_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``commit_user_turn`` flushes silence into the STT in a tight loop.

    That synthetic audio makes the lag go negative, which must apply to the audio
    pushed after it (it maps back onto real time) and never to the speech that was
    captured before the flush.
    """
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(audio_recognition.time, "time", clock)
    pipeline = _make_pipeline(anchor=999.9)

    _push(pipeline, clock, 3)  # speech, positions 0.0–0.3
    for _ in range(10):  # 1 s of silence pushed at once
        pipeline.push_frame(_frame())
    clock.now += 0.05
    pipeline.push_frame(_frame())  # real audio resumes, position 1.4

    assert pipeline.wall_time(0.25) == pytest.approx(1000.15)  # unchanged by the flush
    assert pipeline.wall_time(1.35) == pytest.approx(1000.3)  # the frame after it


def test_word_ending_exactly_at_a_gap_boundary_keeps_the_earlier_plateau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new plateau is recorded at the position of the frame that established it: an
    event ending exactly there describes audio that ended just before that frame."""
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(audio_recognition.time, "time", clock)
    pipeline = _make_pipeline(anchor=999.9)

    _push(pipeline, clock, 3)
    clock.now += 5.0
    pipeline.push_frame(_frame())  # the new plateau starts at position 0.3

    assert pipeline.wall_time(0.3) == pytest.approx(1000.2)  # a pre-gap word
    assert pipeline.wall_time(0.4) == pytest.approx(1005.3)  # inside the late frame


def test_lag_plateaus_collapse_repeated_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """One entry per lag plateau: a long session must not grow the list."""
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(audio_recognition.time, "time", clock)
    pipeline = _make_pipeline(anchor=1000.0)

    _push(pipeline, clock, 50)

    lags = pipeline._arrival_lags  # type: ignore[attr-defined]
    assert len(lags) == 1
    assert lags[0][1] == pytest.approx(-0.1, abs=1e-6)  # anchor is a frame early
    assert pipeline.wall_time(4.95) == pytest.approx(1004.85)


def test_a_late_plateau_never_applies_to_earlier_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reverse of the flush case: audio that fell behind must not move audio that
    was pushed before it."""
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(audio_recognition.time, "time", clock)
    pipeline = _make_pipeline(anchor=999.9)

    _push(pipeline, clock, 3)
    assert math.isclose(pipeline.wall_time(0.15), 1000.05, abs_tol=1e-6)

    clock.now += 2.0  # a stall: the next frame arrives 2 s late
    pipeline.push_frame(_frame())

    assert pipeline.wall_time(0.15) == pytest.approx(1000.05)  # earlier audio intact
    assert pipeline.wall_time(0.35) == pytest.approx(1002.25)  # the late frame moved
