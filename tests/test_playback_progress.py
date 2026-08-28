"""The room sink reports where its audio actually went.

Playback is continuous between discontinuities, so an event exists to describe a boundary: the
source queue drained, the queue was cleared, or the segment finished.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice import io
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput

pytestmark = pytest.mark.unit

RATE = 48000


class _FakeSource:
    """Stands in for rtc.AudioSource; the test drives `queued_duration` by hand."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.queued_duration = 0.0
        self.cleared = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.queued_duration += frame.duration

    def clear_queue(self) -> None:
        self.cleared += 1
        self.queued_duration = 0.0

    async def wait_for_playout(self) -> None:
        self.queued_duration = 0.0


def _frame(seconds: float) -> rtc.AudioFrame:
    n = round(seconds * RATE)
    return rtc.AudioFrame(bytes(n * 2), RATE, 1, n)


class _Harness:
    def __init__(self) -> None:
        self.now = 100.0
        self.progress: list[io.PlaybackProgressedEvent] = []
        with patch("livekit.agents.voice.room_io._output.rtc.AudioSource", _FakeSource):
            self.sink = _ParticipantAudioOutput(
                MagicMock(),
                sample_rate=RATE,
                num_channels=1,
                track_publish_options=MagicMock(),
            )
        self.source: _FakeSource = self.sink._audio_source  # type: ignore[assignment]
        self.sink._subscribed_fut.set_result(None)
        self.sink.on("playback_progressed", self.progress.append)

    async def __aenter__(self) -> _Harness:
        self._patch = patch(
            "livekit.agents.voice.room_io._output.time.time", side_effect=lambda: self.now
        )
        self._patch.start()
        self._task = asyncio.create_task(self.sink._forward_audio())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._task.cancel()
        self._patch.stop()

    async def push(self, seconds: float) -> None:
        """Capture a frame and let the forwarding task hand it to the source."""
        await self.sink.capture_frame(_frame(seconds))
        await self.settle()

    async def settle(self) -> None:
        for _ in range(200):
            if self.sink._audio_buf.empty() and self.sink._forwarding_idle.is_set():
                return
            await asyncio.sleep(0)


async def test_uninterrupted_playback_is_reported_once() -> None:
    async with _Harness() as h:
        h.source.queued_duration = 0.5  # a backlog the clock never catches up with
        for _ in range(3):
            await h.push(0.1)
            h.now += 0.1

        assert h.progress == []  # nothing to report while playback is continuous

        h.sink.flush()
        for _ in range(50):
            await asyncio.sleep(0)

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.offset == 0.0
    assert ev.duration == pytest.approx(0.3)


async def test_a_drained_source_ends_its_run_when_it_ran_dry() -> None:
    """Not when we noticed: the gap the listener heard is the gap in the recording."""
    async with _Harness() as h:
        await h.push(0.2)
        played, dry_at = h.sink._source_pushed_duration, h.sink._dry_at
        assert dry_at is not None

        h.source.queued_duration = 0.0
        h.now = dry_at + 0.5  # the next audio arrives half a second after the source ran out
        await h.push(0.1)

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.offset == 0.0
    assert ev.duration == pytest.approx(played)
    # the run ended when the queue emptied, not when the next push revealed it
    assert ev.started_at == pytest.approx(dry_at - played)


async def test_a_flush_after_the_source_drained_ends_the_run_when_it_ran_dry() -> None:
    """A segment whose flush trails its own playout still sits where it played."""
    async with _Harness() as h:
        # 60ms is the progressive ramp, 20ms then 40ms, so the byte stream holds nothing back
        await h.push(0.06)
        played, dry_at = h.sink._source_pushed_duration, h.sink._dry_at
        assert dry_at is not None

        h.source.queued_duration = 0.0
        h.now = dry_at + 0.7  # the flush arrives long after the last audio played
        h.sink.flush()
        for _ in range(50):
            await asyncio.sleep(0)

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.offset == 0.0
    assert ev.duration == pytest.approx(played)
    assert ev.started_at == pytest.approx(dry_at - played)


async def test_a_cleared_queue_leaves_a_hole_rather_than_a_short_tail() -> None:
    """pause() drops what the source holds, so the audio that never played is in the middle."""
    async with _Harness() as h:
        await h.push(0.5)
        pushed = h.sink._source_pushed_duration
        h.now += 0.3
        h.source.queued_duration = 0.2  # 0.2s queued and about to be thrown away

        h.sink.pause()
        await h.sink.capture_frame(_frame(0.1))
        await h.settle()  # the forwarding task notices the pause and clears the queue
        assert h.source.cleared == 1

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.offset == 0.0
    assert ev.duration == pytest.approx(pushed - 0.2)
    # the next run resumes past the discarded audio rather than replaying it
    assert h.sink._run_offset == pytest.approx(pushed)


async def test_a_pause_and_resume_report_two_runs_around_the_hole() -> None:
    async with _Harness() as h:
        await h.push(0.5)
        pushed = h.sink._source_pushed_duration
        h.now += 0.3
        h.source.queued_duration = 0.2

        h.sink.pause()
        await h.sink.capture_frame(_frame(0.5))
        await h.settle()

        h.now += 1.0  # the agent stays silent while the user speaks
        resumed_at = h.now
        h.source.queued_duration = 2.0  # a backlog the clock never catches up with
        h.sink.resume()
        await h.settle()

        h.now += 1.0  # the resumed audio plays out
        h.sink.flush()
        for _ in range(50):
            await asyncio.sleep(0)

    assert len(h.progress) == 2
    first, second = h.progress
    assert first.offset == 0.0
    assert first.duration == pytest.approx(pushed - 0.2)
    # the second run resumes past the discarded audio, and only after playback came back
    assert second.offset == pytest.approx(pushed)
    assert second.started_at > resumed_at
    assert second.started_at > first.started_at + first.duration


async def test_a_pause_then_interrupt_reports_the_run_once() -> None:
    """The pause already ended the run; the interruption has nothing left to add."""
    async with _Harness() as h:
        await h.push(0.5)
        pushed = h.sink._source_pushed_duration
        h.source.queued_duration = 0.2

        h.sink.pause()
        await h.sink.capture_frame(_frame(0.1))
        await h.settle()
        assert len(h.progress) == 1

        h.sink.flush()
        h.sink.clear_buffer()
        for _ in range(50):
            await asyncio.sleep(0)

    assert len(h.progress) == 1
    assert h.progress[0].duration == pytest.approx(pushed - 0.2)


async def test_an_interruption_ends_the_run_at_the_playhead() -> None:
    async with _Harness() as h:
        await h.push(0.5)
        pushed = h.sink._source_pushed_duration
        h.source.queued_duration = 0.15  # queued and about to be dropped

        h.sink.flush()
        h.sink.clear_buffer()
        for _ in range(50):
            await asyncio.sleep(0)

    assert len(h.progress) == 1
    assert h.progress[0].duration == pytest.approx(pushed - 0.15)


async def test_offsets_restart_with_each_segment() -> None:
    async with _Harness() as h:
        for _ in range(2):
            h.source.queued_duration = 0.5
            await h.push(0.2)
            h.sink.flush()
            for _ in range(50):
                await asyncio.sleep(0)
            h.now += 1.0

    assert len(h.progress) == 2
    assert [ev.offset for ev in h.progress] == [0.0, 0.0]
    assert all(ev.duration == pytest.approx(0.2) for ev in h.progress)
