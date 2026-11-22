from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice.avatar._queue_io import QueueAudioOutput
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, BufferedAudioOutput
from livekit.agents.voice.recorder_io.recorder_io import RecorderAudioOutput, RecorderIO
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

from .fake_io import FakeAudioOutput, FakeTextOutput

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
    in-flight, cancelling that task must not leave the already-forwarded
    segment unreported. The wrapper re-arms the sink's completion before
    clearing it, so wait_for_playout() does not hang.
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

    # must not hang: the sink reports the interrupted segment
    ev = await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)

    assert ev.interrupted is True
    assert sink.cleared is True
    # the segment is reported exactly once, so the next reply starts clean
    assert sink.segments == 1, f"expected one completed segment, got {sink.segments}"


async def test_interruption_is_reported_as_interrupted() -> None:
    """
    A stopped reply must be reported to the caller as interrupted, never as a
    normal completion: the sink's flush only arms reporting, and the clear that
    immediately follows is what settles it.
    """
    sink = _TrackingSink()
    buf = _buffered(sink, 0.3)

    for _ in range(4):
        await buf.capture_frame(_frame(0.1))  # 0.4s > 0.3s buffer, forwards downstream

    buf.flush()
    buf.clear_buffer()

    ev = await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)
    assert ev.interrupted is True, "a cleared segment must report interrupted=True"


async def test_flush_after_interruption_still_completes_normally() -> None:
    """
    The flush-before-clear ordering must not leak: a reply that plays to its
    end afterwards still reports a normal completion.
    """
    sink = _TrackingSink()
    buf = _buffered(sink, 0.3)

    for _ in range(4):
        await buf.capture_frame(_frame(0.1))

    buf.flush()
    await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)

    for _ in range(4):
        await buf.capture_frame(_frame(0.1))
    buf.flush()

    ev = await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)
    assert ev.interrupted is False, "an uninterrupted reply must report a completion"
    assert sink.segments == 2


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
    """Sink that counts segments and tracks flush/clear state.

    Mirrors the room sink's shape: ``flush()`` only *arms* completion reporting by
    creating a playout-wait task, and that task reports ``interrupted=True`` if
    ``clear_buffer()`` arrives first. Reporting is asynchronous in both paths.

    This sink has ``flush_ends_segment=False``: its ``flush()`` only arms a local
    completion task, and an extra ``flush()`` before ``clear_buffer()`` is required
    to arm the task so the interruption is consumed.
    """

    def __init__(self) -> None:
        super().__init__(
            label="TrackingSink",
            capabilities=AudioOutputCapabilities(pause=True, flush_ends_segment=False),
        )
        self.segments = 0
        self.flushed = False
        self.cleared = False
        self.flush_events = 0
        self._interrupted = False
        self._flush_task: asyncio.Task[None] | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    async def _report(self) -> None:
        # one loop turn, so a clear_buffer() arriving right after flush() wins
        await asyncio.sleep(0)
        interrupted = self._interrupted
        self.segments += 1
        self.on_playback_finished(playback_position=0.0, interrupted=interrupted)

    def flush(self) -> None:
        super().flush()
        self.flushed = True
        self.flush_events += 1
        self._interrupted = False
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._report())

    def clear_buffer(self) -> None:
        super().clear_buffer()
        self.cleared = True
        self._interrupted = True
        # like the room sink, this only signals the interruption; a flush-armed
        # completion task is what turns it into a reported segment. clearing
        # without a flush therefore leaves the segment unreported, which is what
        # the wrapper's flush-before-clear ordering prevents.


async def test_interruption_does_not_close_an_avatar_stream() -> None:
    """
    Regression test for: an interruption must not flush a sink whose flush() ends
    the segment for the remote receiver. The avatar data-stream sink closes its
    stream on flush, so a flush issued by clear_buffer() would make the avatar
    report a normal completion and the stopped reply would look uninterrupted.
    """
    sink = _StreamClosingSink()
    buf = _buffered(sink, 0.3)

    for _ in range(4):
        await buf.capture_frame(_frame(0.1))  # 0.4s > 0.3s buffer, forwards downstream

    buf.flush()
    # the interruption lands before the scheduled release reaches the sink
    buf.clear_buffer()

    ev = await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)
    assert ev.interrupted is True, "the stopped reply must report interrupted=True"
    # exactly one end of speech, and it must come from the clear, not a flush
    assert sink.streams_closed == 0, "clearing must not close the avatar's stream"
    assert sink.clear_calls == 1


class _StreamClosingSink(AudioOutput):
    """Sink whose ``flush()`` has an externally visible end-of-segment effect.

    Mirrors the avatar data-stream sink: ``flush()`` closes the remote stream, so
    flushing before clearing would end the segment as a normal completion before
    the clear could interrupt it. Such a sink must not receive an extra flush.
    """

    def __init__(self) -> None:
        super().__init__(
            label="StreamClosingSink",
            capabilities=AudioOutputCapabilities(pause=False, flush_ends_segment=True),
        )
        self.streams_closed = 0
        self.clear_calls = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        # closing the remote stream is what the avatar sees as end of speech
        self.streams_closed += 1
        self.on_playback_finished(playback_position=0.0, interrupted=False)

    def clear_buffer(self) -> None:
        super().clear_buffer()
        self.clear_calls += 1
        self.on_playback_finished(playback_position=0.0, interrupted=True)


class _RecordingSink(AudioOutput):
    """Records the sink-level frame/flush order without simulating playout."""

    def __init__(self) -> None:
        super().__init__(
            label="RecordingSink",
            capabilities=AudioOutputCapabilities(pause=False, flush_ends_segment=False),
        )
        self.events: list[rtc.AudioFrame | str] = []
        self._flush_task: asyncio.Task[None] | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.events.append(frame)

    def flush(self) -> None:
        super().flush()
        self.events.append("flush")
        self._flush_task = asyncio.create_task(self._report())

    async def _report(self) -> None:
        await asyncio.sleep(0)
        self.on_playback_finished(playback_position=0.0, interrupted=False)

    def clear_buffer(self) -> None:
        super().clear_buffer()


class _BlockingSink(AudioOutput):
    """Blocks on the second capture so waiters can observe an in-flight release."""

    def __init__(self) -> None:
        super().__init__(
            label="BlockingSink",
            capabilities=AudioOutputCapabilities(pause=True, flush_ends_segment=False),
        )
        self.captures: list[rtc.AudioFrame] = []
        self.flush_events = 0
        self.cleared = False
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()
        self.accepted_after_clear = False
        self._flush_wait_task: asyncio.Task[None] | None = None
        self._interrupted = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if len(self.captures) < 1:
            self.captures.append(frame)
            return

        self.second_started.set()
        await self.release_second.wait()
        self.accepted_after_clear = self.cleared
        self.captures.append(frame)

    def flush(self) -> None:
        super().flush()
        self.flush_events += 1
        self._flush_wait_task = asyncio.create_task(self._report())

    async def _report(self) -> None:
        await asyncio.sleep(0)
        self.on_playback_finished(playback_position=0.0, interrupted=self._interrupted)

    def clear_buffer(self) -> None:
        super().clear_buffer()
        self.cleared = True
        self._interrupted = True


# -- release ordering and cancellation ----------------------------------------


async def test_adjacent_releases_forward_in_flush_order() -> None:
    """A later reply cannot overtake the prior reply's pending release."""
    sink = _RecordingSink()
    buf = _buffered(sink, 0.3)

    first = _frame(0.1)
    second = _frame(0.3)
    await buf.capture_frame(first)
    buf.flush()
    await buf.capture_frame(second)
    buf.flush()

    await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)

    assert sink.events == [first, "flush", second, "flush"]


async def test_immediately_flushed_adjacent_replies_keep_order() -> None:
    """Two short replies flushed back to back still get separate boundaries."""
    sink = _RecordingSink()
    buf = _buffered(sink, 0.3)

    first = _frame(0.1)
    second = _frame(0.1)
    await buf.capture_frame(first)
    buf.flush()
    await buf.capture_frame(second)
    buf.flush()

    await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)

    assert sink.events == [first, "flush", second, "flush"]


async def test_waiter_does_not_consume_an_in_flight_release() -> None:
    """wait_for_playout() must leave the release visible to clear_buffer()."""
    sink = _BlockingSink()
    buf = _buffered(sink, 0.3)

    await buf.capture_frame(_frame(0.1))
    await buf.capture_frame(_frame(0.1))
    buf.flush()

    # let the release accept the first frame and block on the second
    await asyncio.wait_for(sink.second_started.wait(), timeout=1.0)
    first_waiter = asyncio.create_task(buf.wait_for_playout())
    second_waiter = asyncio.create_task(buf.wait_for_playout())
    await asyncio.sleep(0)

    buf.clear_buffer()
    sink.release_second.set()

    first = await asyncio.wait_for(first_waiter, timeout=1.0)
    second = await asyncio.wait_for(second_waiter, timeout=1.0)
    assert first.interrupted is True
    assert second.interrupted is True
    assert sink.accepted_after_clear is False
    assert len(sink.captures) == 1
    assert sink.flush_events == 1


async def test_wholly_held_interruption_settles_an_outer_recorder() -> None:
    """An outer wrapper that counted a held segment is released by cancellation."""
    sink = _RecordingSink()
    buf = _buffered(sink, 0.3)
    recorder = RecorderAudioOutput(
        recording_io=MagicMock(recording=True, spec=RecorderIO),
        audio_output=buf,
        on_played=MagicMock(),
    )

    await recorder.capture_frame(_frame(0.1))
    assert recorder._pending_playback_count == 1
    assert buf.captured_playout_segments == 1
    assert sink.events == []

    recorder.clear_buffer()
    event = await asyncio.wait_for(recorder.wait_for_playout(), timeout=1.0)

    assert event.interrupted is True
    assert event.playback_position == 0
    assert recorder._pending_playback_count == 0
    assert buf._pending_playback_count == 0
    assert sink.events == []


async def test_wholly_held_interruption_settles_transcript_sync() -> None:
    """The synchronizer wrapper above a wholly held buffer is also released."""
    sink = _RecordingSink()
    buf = _buffered(sink, 0.3)
    sync = TranscriptSynchronizer(next_in_chain_audio=buf, next_in_chain_text=FakeTextOutput())
    output = sync.audio_output

    await sync.text_output.capture_text("interrupted")
    sync.text_output.flush()
    await output.capture_frame(_frame(0.1))
    assert output._pending_playback_count == 1

    output.clear_buffer()
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)

    assert event.interrupted is True
    assert output._pending_playback_count == 0
    assert buf._pending_playback_count == 0
    assert sink.events == []
    await sync.aclose()


async def test_dynamic_sink_capabilities_follow_a_swap() -> None:
    """The wrapper contract reflects the current leaf after replace_audio_tail."""
    room_sink = FakeAudioOutput()
    buf = BufferedAudioOutput(next_in_chain=room_sink, buffer_duration=0.1)
    assert buf.capabilities.flush_ends_segment is False

    # Auto-wrapping gives the stable proxy used by output tail replacement.
    proxy = buf.next_in_chain
    assert proxy is not None
    avatar_sink = _StreamClosingSink()
    proxy.set_next_in_chain(avatar_sink)

    assert proxy.capabilities.flush_ends_segment is True
    assert buf.capabilities.flush_ends_segment is True


# -- real avatar sinks --------------------------------------------------------


async def test_queue_output_interrupt_resets_capture_for_the_next_reply() -> None:
    """A cleared queue segment cannot swallow playback_started for the next reply."""
    queue = QueueAudioOutput()
    buf = _buffered(queue, 0.05)
    clear_events = 0

    def on_clear() -> None:
        nonlocal clear_events
        clear_events += 1
        # the avatar runner's clear callback reports the interrupted segment
        if queue._pending_playback_count:
            queue.notify_playback_finished(playback_position=0.0, interrupted=True)

    queue.on("clear_buffer", on_clear)

    async def consume() -> None:
        async for _frame_or_end in queue:
            pass

    consumer_task = asyncio.create_task(consume())

    try:
        await buf.capture_frame(_frame(0.1))
        assert queue.captured_playout_segments == 1

        buf.clear_buffer()
        await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)

        await buf.capture_frame(_frame(0.1))
        buf.flush()
        queue.notify_playback_finished(playback_position=0.1, interrupted=False)
        await asyncio.wait_for(buf.wait_for_playout(), timeout=1.0)
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)

    assert clear_events == 1
    assert queue.captured_playout_segments == 2
    assert queue._pending_playback_count == 0
