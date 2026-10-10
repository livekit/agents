from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, BufferedAudioOutput

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit

SR = 16000


def _frame(duration_s: float) -> rtc.AudioFrame:
    n = int(SR * duration_s)
    return rtc.AudioFrame(
        data=b"\x00\x00" * n,
        sample_rate=SR,
        num_channels=1,
        samples_per_channel=n,
    )


def _buffered(sink: AudioOutput, buffer_duration: float = 0.3) -> BufferedAudioOutput:
    return BufferedAudioOutput(next_in_chain=sink, buffer_duration=buffer_duration)


# -- the buffer holds audio back ----------------------------------------------


async def test_first_frames_are_held_until_the_buffer_fills() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.3)

    await buf.capture_frame(_frame(0.1))
    await buf.capture_frame(_frame(0.1))
    assert sink.captured_playout_segments == 0
    assert buf.buffered_duration == pytest.approx(0.2)

    # crossing the threshold releases everything held, in order
    await buf.capture_frame(_frame(0.1))
    assert sink.captured_playout_segments == 1
    assert buf.buffered_duration == 0.0


async def test_audio_past_the_buffer_goes_straight_through() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.1)

    for _ in range(5):
        await buf.capture_frame(_frame(0.1))

    assert sink.captured_playout_segments == 1


# -- short segments -----------------------------------------------------------


async def test_flush_releases_a_segment_shorter_than_the_buffer() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.3)

    await buf.capture_frame(_frame(0.1))
    await buf.capture_frame(_frame(0.1))
    assert sink.captured_playout_segments == 0

    buf.flush()
    await buf.wait_for_playout()

    assert sink.captured_playout_segments == 1, "a short reply must not be swallowed"
    assert buf.buffered_duration == 0.0


async def test_clear_buffer_drops_the_held_audio() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.3)

    await buf.capture_frame(_frame(0.1))
    buf.clear_buffer()

    assert buf.buffered_duration == 0.0
    assert sink.captured_playout_segments == 0, "interruption must not play held audio"


async def test_clear_buffer_joins_an_in_flight_release_without_hanging() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.3)

    await buf.capture_frame(_frame(0.1))
    buf.flush()
    buf.clear_buffer()

    # a barge-in right after a flush must not leave wait_for_playout() pending forever
    await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)


# -- per-segment isolation ----------------------------------------------------


async def test_every_segment_primes_again() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.1)

    for _ in range(3):
        await buf.capture_frame(_frame(0.1))
        buf.flush()
        await buf.wait_for_playout()

    assert sink.captured_playout_segments == 3


# -- re-priming when the sink drains ------------------------------------------


async def test_reserve_drains_over_time_so_the_buffer_refills() -> None:
    sink = FakeAudioOutput()
    # small buffer, so the reserve runs out while the sink is still playing
    buf = _buffered(sink, 0.1)

    await buf.capture_frame(_frame(0.1))  # primes, then releases: reserve is now 0.1s
    assert buf.buffered_duration == 0.0

    # let the sink play out the reserve
    await asyncio.sleep(0.2)
    await buf.capture_frame(_frame(0.1))

    assert buf.buffered_duration > 0.0, "a drained reserve should be rebuilt"


async def test_reserve_that_has_not_drained_keeps_forwarding() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.1)

    await buf.capture_frame(_frame(0.1))  # primes, then releases
    await buf.capture_frame(_frame(0.1))

    assert buf.buffered_duration == 0.0, "a full reserve must not re-prime"


# -- pause --------------------------------------------------------------------


async def test_pause_stops_the_reserve_from_draining() -> None:
    sink = FakeAudioOutput(can_pause=True)
    buf = _buffered(sink, 0.1)

    await buf.capture_frame(_frame(0.1))  # primes, then releases
    buf.pause()
    await asyncio.sleep(0.2)

    assert buf.can_pause is True
    await buf.capture_frame(_frame(0.1))
    assert buf.buffered_duration == 0.0, "a paused sink is not draining its reserve"

    buf.resume()


# -- configuration ------------------------------------------------------------


async def test_zero_duration_forwards_every_frame_immediately() -> None:
    sink = FakeAudioOutput()
    buf = _buffered(sink, 0.0)

    await buf.capture_frame(_frame(0.1))

    assert sink.captured_playout_segments == 1
    assert buf.buffered_duration == 0.0


async def test_sample_rate_and_pause_are_taken_from_the_sink() -> None:
    sink = FakeAudioOutput(sample_rate=SR, can_pause=True)
    buf = _buffered(sink, 0.1)

    assert buf.sample_rate == SR
    assert buf.can_pause is True


async def test_a_sink_that_cannot_pause_is_reported_as_such() -> None:
    buf = _buffered(FakeAudioOutput(can_pause=False), 0.1)

    assert buf.can_pause is False


# -- Devin regression tests ---------------------------------------------------


async def test_interruption_during_flush_does_not_hang() -> None:
    """
    Regression test for: when clear_buffer() is called while a flush task is
    in-flight, the task must complete and flush the downstream sink for any
    frames already forwarded, so wait_for_playout() does not hang.
    """
    sink = _TrackingSink()
    buf = _buffered(sink, 0.3)

    # push enough frames to cross the buffer threshold and auto-release
    for _ in range(4):
        await buf.capture_frame(_frame(0.1))  # 0.4s > 0.3s buffer

    # flush schedules the release task
    buf.flush()

    # immediate interruption before the task completes
    buf.clear_buffer()

    # must not hang - the in-flight task completes and flushes the sink
    await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)

    # the sink receives flush (from the interrupted segment) which completes
    # it; wait_for_playout completes without hanging.
    assert sink.flushed is True


async def test_adjacent_replies_do_not_merge_segments() -> None:
    """
    Regression test for: flush() closes the wrapper's segment synchronously
    but delays the sink's flush to a task. If a new reply captures audio
    before that task runs, it must not enter the previous sink segment.
    """
    sink = _TrackingSink()
    buf = _buffered(sink, 0.1)

    # first reply: fill buffer, auto-releases, then flush
    await buf.capture_frame(_frame(0.1))
    await buf.capture_frame(_frame(0.1))
    buf.flush()
    await buf.wait_for_playout()

    # second reply must be a separate segment
    await buf.capture_frame(_frame(0.1))
    buf.flush()
    await buf.wait_for_playout()

    assert sink.segments == 2, f"each reply must be its own segment, got {sink.segments}"


class _TrackingSink(AudioOutput):
    """Sink that counts segments and tracks flush state."""

    def __init__(self) -> None:
        super().__init__(
            label="TrackingSink",
            capabilities=AudioOutputCapabilities(pause=True),
        )
        self.segments = 0
        self.flushed = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self.segments += 1
        self.flushed = True
        self.on_playback_finished(playback_position=0.0, interrupted=False)

    def clear_buffer(self) -> None:
        super().clear_buffer()
        # emit interrupted playback event so wait_for_playout() completes
        self.on_playback_finished(playback_position=0.0, interrupted=True)
