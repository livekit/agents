from __future__ import annotations

import asyncio
import gc
import time
import weakref
from collections.abc import AsyncIterable, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.voice import io
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.generation import (
    _audio_forwarding_task,
    _AudioOutput,
    forward_generation,
    perform_audio_forwarding,
)
from livekit.agents.voice.room_io._output import _ParticipantAudioOutput
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

RATE = 48000


def _frame(seconds: float) -> rtc.AudioFrame:
    n = round(seconds * RATE)
    return rtc.AudioFrame(bytes(n * 2), RATE, 1, n)


# --- room sink playback progress: an event marks a boundary (drained, cleared or finished) ---


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
        # the test drains the queue; a real source takes as long as the audio it holds
        while self.queued_duration:
            await asyncio.sleep(0)


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
        await h.settle()
        h.source.queued_duration = 0.0  # the backlog plays out
        for _ in range(50):
            await asyncio.sleep(0)

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.offset == 0.0
    assert ev.duration == pytest.approx(0.3)


async def test_the_tail_follows_before_the_source_runs_dry() -> None:
    """A packet that ends mid-frame still plays whole, with no gap after the frame it filled."""
    async with _Harness() as h:
        loop = asyncio.get_running_loop()
        await h.push(0.08)  # 50 ms out; 30 ms waits, for now
        assert h.source.queued_duration == pytest.approx(0.05)
        assert h.sink._bstream_flush_timer is not None
        assert h.sink._bstream_flush_timer.when() - loop.time() == pytest.approx(0.03, abs=0.01)
        await asyncio.sleep(0.04)  # the deadline is 20 ms before the 50 ms plays out
        await h.settle()
        assert h.source.queued_duration == pytest.approx(0.08)


async def test_the_tail_waits_to_fill_a_frame_while_enough_is_queued() -> None:
    """The deadline counts the source queue and the channel; every push moves it."""
    async with _Harness() as h:
        loop = asyncio.get_running_loop()
        h.source.queued_duration = 0.5
        await h.push(0.08)  # 50 ms out, 30 ms held
        assert h.source.queued_duration == pytest.approx(0.55)
        assert h.sink._bstream_flush_timer.when() - loop.time() == pytest.approx(0.53, abs=0.01)

        await h.push(0.08)  # 30 + 80 ms: two frames out, 10 ms held
        assert h.source.queued_duration == pytest.approx(0.65)
        assert h.sink._bstream_flush_timer.when() - loop.time() == pytest.approx(0.63, abs=0.01)

        h.sink._flush_bstream()  # the deadline arrives
        await h.settle()
        assert h.source.queued_duration == pytest.approx(0.66)
        assert h.sink._bstream_flush_timer is None


async def test_a_small_first_frame_waits_for_more() -> None:
    """Nothing plays yet, so a first packet under 20 ms waits for the next one to fill a frame."""
    async with _Harness() as h:
        await h.push(0.01)
        await asyncio.sleep(0.001)  # the timer fires at once and leaves it waiting
        assert h.sink._bstream_flush_timer is None
        assert h.source.queued_duration == 0.0
        await h.push(0.04)
        assert h.source.queued_duration == pytest.approx(0.05)  # one frame, no 10 ms one


async def test_a_segment_flush_releases_a_small_first_frame() -> None:
    async with _Harness() as h:
        await h.push(0.01)
        h.sink.flush()
        await h.settle()
        assert h.source.queued_duration == pytest.approx(0.01)
        h.source.queued_duration = 0.0  # the backlog plays out
        for _ in range(50):
            await asyncio.sleep(0)


async def test_a_clear_disarms_the_bstream_flush_timer() -> None:
    async with _Harness() as h:
        h.source.queued_duration = 0.5
        await h.push(0.08)
        h.sink.clear_buffer()
        assert h.sink._bstream_flush_timer is None
        assert h.sink._audio_bstream.flush() == []


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


async def test_a_stalled_capture_still_ends_the_run_when_it_ran_dry() -> None:
    """The event loop can return from capture_frame after the source played out what it
    was handed; the run still ends where the audio did."""
    async with _Harness() as h:
        capture = h.source.capture_frame

        async def stalled(frame: rtc.AudioFrame) -> None:
            await capture(frame)
            h.now += 0.3  # the loop resumes 300 ms later, the frame long played out
            h.source.queued_duration = 0.0

        submitted_at = h.now
        h.source.capture_frame = stalled  # type: ignore[method-assign]
        await h.push(0.05)
        h.source.capture_frame = capture  # type: ignore[method-assign]
        played = h.sink._source_pushed_duration

        h.now += 0.5
        await h.push(0.1)

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.duration == pytest.approx(played)
    assert ev.started_at == pytest.approx(submitted_at)


async def test_a_flush_after_the_source_drained_ends_the_run_when_it_ran_dry() -> None:
    """A segment whose flush trails its own playout still sits where it played."""
    async with _Harness() as h:
        # 100ms frames as two 50ms frames, so the byte stream holds nothing back
        await h.push(0.1)
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


async def test_a_cleared_buffer_reports_the_playhead_without_waiting() -> None:
    """A swap detaches the sink before its playout wait runs, so the clear is its last word."""
    async with _Harness() as h:
        await h.push(0.5)
        pushed = h.sink._source_pushed_duration
        h.now += 0.3
        h.source.queued_duration = 0.2  # queued, so not yet played

        h.sink.clear_buffer()

    assert len(h.progress) == 1
    ev = h.progress[0]
    assert ev.offset == 0.0
    assert ev.duration == pytest.approx(pushed - 0.2)


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


async def test_no_tail_timer_runs_while_paused_and_resume_arms_one() -> None:
    """A paused output needs no frames; resume arms the timer from what survived the pause."""
    async with _Harness() as h:
        loop = asyncio.get_running_loop()
        await h.push(0.08)  # 50 ms out, 30 ms tail with a timer
        assert h.sink._bstream_flush_timer is not None
        h.sink.pause()
        assert h.sink._bstream_flush_timer is None

        await h.sink.capture_frame(_frame(0.08))  # 30 + 80 ms: two frames out, 10 ms tail
        assert h.sink._bstream_flush_timer is None
        await h.settle()  # the forwarder sees the pause and drops the source queue
        assert h.source.queued_duration == 0.0

        h.sink.resume()  # only the two held frames stand ahead of the tail
        assert h.sink._bstream_flush_timer.when() - loop.time() == pytest.approx(0.08, abs=0.01)
        await h.settle()
        assert h.source.queued_duration == pytest.approx(0.1)
        h.sink._flush_bstream()  # the deadline arrives
        await h.settle()
        assert h.source.queued_duration == pytest.approx(0.11)  # the tail follows, no gap


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
        await h.settle()
        h.source.queued_duration = 0.0
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


async def test_the_playout_wait_reports_what_plays_after_a_clear() -> None:
    """The clear reports the playhead, and the queue plays on until the wait drops it."""
    async with _Harness() as h:
        await h.push(0.5)
        pushed = h.sink._source_pushed_duration
        h.source.queued_duration = 0.2

        h.sink.flush()
        h.sink.clear_buffer()
        h.now += 0.05
        h.source.queued_duration = 0.15  # the device played on while the wait was scheduled

        for _ in range(50):
            await asyncio.sleep(0)

    played = round(pushed - 0.2, 3)
    assert [(round(ev.offset, 3), round(ev.duration, 3)) for ev in h.progress] == [
        (0.0, played),
        (played, 0.05),
    ]


async def test_offsets_restart_with_each_segment() -> None:
    async with _Harness() as h:
        for _ in range(2):
            h.source.queued_duration = 0.5
            await h.push(0.2)
            h.sink.flush()
            await h.settle()
            h.source.queued_duration = 0.0
            for _ in range(50):
                await asyncio.sleep(0)
            h.now += 1.0

    assert len(h.progress) == 2
    assert [ev.offset for ev in h.progress] == [0.0, 0.0]
    assert all(ev.duration == pytest.approx(0.2) for ev in h.progress)


# --- first-frame attribution in audio forwarding ---


class _ForwardFirstWrapper(io.AudioOutput):
    """Count each segment after forwarding its first frame."""

    def __init__(self, next_in_chain: io.AudioOutput) -> None:
        super().__init__(
            label="ForwardFirstWrapper",
            next_in_chain=next_in_chain,
            capabilities=next_in_chain._capabilities,
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        assert self.next_in_chain is not None
        await self.next_in_chain.capture_frame(frame)
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        assert self.next_in_chain is not None
        self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        assert self.next_in_chain is not None
        self.next_in_chain.clear_buffer()


class _NoStartNotifyOutput(FakeAudioOutput):
    """Play audio without emitting ``playback_started``."""

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await io.AudioOutput.capture_frame(self, frame)
        self._pushed_duration += frame.duration
        if self._started_at is None:
            self._started_at = time.time()


class _BlockingBeforeCaptureOutput(FakeAudioOutput):
    def __init__(self) -> None:
        super().__init__()
        self.capture_entered = asyncio.Event()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.capture_entered.set()
        await asyncio.Event().wait()
        await super().capture_frame(frame)


async def _drive_forwarding(
    audio_output: io.AudioOutput, frames_ch: asyncio.Queue[rtc.AudioFrame | None]
):
    async def _source() -> AsyncIterable[rtc.AudioFrame]:
        while True:
            frame = await frames_ch.get()
            if frame is None:
                return
            yield frame

    return perform_audio_forwarding(
        audio_output=audio_output,
        tts_output=_source(),
        reconcile_playout_pause=lambda: None,
    )


@pytest.mark.parametrize("output_sample_rate", [None, 16000])
async def test_forwarding_result_does_not_retain_consumed_audio(
    output_sample_rate: int | None,
) -> None:
    audio_output = FakeAudioOutput(sample_rate=output_sample_rate)
    frames: weakref.WeakSet[rtc.AudioFrame] = weakref.WeakSet()

    async def _source() -> AsyncIterable[rtc.AudioFrame]:
        for _ in range(300):
            frame = rtc.AudioFrame.create(
                sample_rate=48000, num_channels=1, samples_per_channel=960
            )
            frames.add(frame)
            yield frame

    task, out = perform_audio_forwarding(
        audio_output=audio_output,
        tts_output=_source(),
        reconcile_playout_pause=lambda: None,
    )
    try:
        await task
        await asyncio.sleep(0)
        gc.collect()

        assert audio_output._pushed_duration == pytest.approx(6.0)
        assert not frames
        assert out.first_frame_fut.done()
    finally:
        audio_output.clear_buffer()


async def test_own_playback_started_resolves_first_frame_fut() -> None:
    audio_output = FakeAudioOutput()
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)

    frames_ch.put_nowait(_frame(0.1))
    frames_ch.put_nowait(None)
    await task

    assert out.first_frame_fut.done()
    assert audio_output.captured_playout_segments > out.captured_segments_before
    audio_output.clear_buffer()


async def test_own_event_forwarded_before_wrapper_counts_still_resolves() -> None:
    audio_output = _ForwardFirstWrapper(FakeAudioOutput())
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)

    frames_ch.put_nowait(_frame(0.1))
    frames_ch.put_nowait(None)
    await task

    assert out.first_frame_fut.done()
    audio_output.clear_buffer()


async def test_stale_event_before_own_capture_is_ignored() -> None:
    audio_output = FakeAudioOutput()
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)
    await asyncio.sleep(0)  # let the forwarding task attach its listener

    # a delayed RPC from an interrupted segment
    audio_output.on_playback_started(created_at=time.time())

    assert not out.first_frame_fut.done()

    frames_ch.put_nowait(None)
    await task
    out.first_frame_fut.cancel()


async def test_interrupted_commit_uses_position_evidence_without_started_event() -> None:
    # a remote avatar can report playback progress before its start RPC arrives
    audio_output = _NoStartNotifyOutput()
    speech_handle = SpeechHandle.create()

    frame_captured = asyncio.Event()

    async def _audio_source() -> AsyncIterable[rtc.AudioFrame]:
        yield _frame(0.1)
        frame_captured.set()
        await asyncio.Event().wait()  # keep the stream open until interruption

    forward_task = asyncio.create_task(
        forward_generation(
            speech_handle=speech_handle,
            audio_output=audio_output,
            text_output=None,
            audio_source=_audio_source(),
            text_source=None,
            on_first_frame=lambda _fut, _out: None,
            reconcile_playout_pause=lambda: None,
        )
    )

    await asyncio.wait_for(frame_captured.wait(), timeout=5)
    await asyncio.sleep(0.05)  # accrue playback progress
    speech_handle.interrupt()
    out = await asyncio.wait_for(forward_task, timeout=5)

    assert out.audio_out is not None
    assert not out.audio_out.first_frame_fut.done() or out.audio_out.first_frame_fut.cancelled()
    assert audio_output.captured_playout_segments > out.audio_out.captured_segments_before
    assert out.played == "partial"
    assert out.playback_position > 0


async def test_stale_event_during_unaccepted_capture_stays_skipped() -> None:
    audio_output = _BlockingBeforeCaptureOutput()
    speech_handle = SpeechHandle.create()

    async def _audio_source() -> AsyncIterable[rtc.AudioFrame]:
        yield _frame(0.1)

    forward_task = asyncio.create_task(
        forward_generation(
            speech_handle=speech_handle,
            audio_output=audio_output,
            text_output=None,
            audio_source=_audio_source(),
            text_source=None,
            on_first_frame=lambda _fut, _out: None,
            reconcile_playout_pause=lambda: None,
        )
    )

    await asyncio.wait_for(audio_output.capture_entered.wait(), timeout=5)
    audio_output.on_playback_started(created_at=time.time())
    speech_handle.interrupt()
    out = await asyncio.wait_for(forward_task, timeout=5)

    assert out.audio_out is not None
    assert out.audio_out.first_frame_fut.done()
    assert audio_output.captured_playout_segments == out.audio_out.captured_segments_before
    assert out.played == "skipped"


async def test_interrupted_commit_stays_skipped_without_any_capture() -> None:
    audio_output = FakeAudioOutput()
    speech_handle = SpeechHandle.create()

    async def _audio_source() -> AsyncIterable[rtc.AudioFrame]:
        await asyncio.Event().wait()
        yield _frame(0.1)  # pragma: no cover

    forward_task = asyncio.create_task(
        forward_generation(
            speech_handle=speech_handle,
            audio_output=audio_output,
            text_output=None,
            audio_source=_audio_source(),
            text_source=None,
            on_first_frame=lambda _fut, _out: None,
            reconcile_playout_pause=lambda: None,
        )
    )

    await asyncio.sleep(0.05)
    speech_handle.interrupt()
    out = await asyncio.wait_for(forward_task, timeout=5)

    assert out.audio_out is not None
    assert audio_output.captured_playout_segments == out.audio_out.captured_segments_before
    assert out.played == "skipped"


# --- playout pause reconciled at launch ---


@pytest.fixture
async def activity() -> AsyncIterator[tuple[AgentActivity, FakeAudioOutput]]:
    session = AgentSession(
        turn_handling={
            "interruption": {
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            }
        }
    )
    audio_output = FakeAudioOutput(can_pause=True)
    session.output.audio = audio_output
    agent_activity = AgentActivity(Agent(instructions="test"), session)
    session._update_agent_state("thinking")

    yield agent_activity, audio_output

    await session.aclose()


async def test_playout_launch_preserves_existing_pause(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    paused_at = audio_output._paused_at

    agent_activity._reconcile_playout_pause(speech_handle)

    assert paused_at is not None
    assert audio_output._paused_at == paused_at
    assert agent_activity._paused_speech is not None
    assert agent_activity._paused_speech.handle is speech_handle
    assert agent_activity._paused_speech.timeout == 2.0


async def test_playout_launch_pauses_when_sos_precedes_current_speech(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._user_silence_event.clear()

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is not None
    assert agent_activity._paused_speech is not None
    assert agent_activity._paused_speech.handle is speech_handle
    assert agent_activity._paused_speech.timeout == 0


async def test_playout_launch_resumes_when_user_is_silent(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    assert agent_activity._user_silence_event.is_set()

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_playout_launch_releases_pause_when_interruptions_are_disabled(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    timer = MagicMock()
    agent_activity._false_interruption_timer = timer
    agent_activity._false_interruption_pending = True

    speech_handle.allow_interruptions = False
    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None
    timer.cancel.assert_called_once_with()
    assert agent_activity._false_interruption_timer is None
    assert agent_activity._false_interruption_pending is False

    speech_handle.allow_interruptions = True
    agent_activity._reconcile_playout_pause(speech_handle)
    await agent_activity._cancel_speech_pause()

    assert audio_output._paused_at is None
    assert speech_handle.interrupted is False


async def test_playout_launch_releases_pause_for_interrupted_speech(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()

    speech_handle.interrupt()
    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_playout_launch_releases_pause_when_pause_is_disabled(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    agent_activity._session.options.interruption["resume_false_interruption"] = False

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_playout_launch_releases_pause_when_audio_output_is_disabled(
    activity: tuple[AgentActivity, FakeAudioOutput],
) -> None:
    agent_activity, audio_output = activity
    speech_handle = SpeechHandle.create()
    agent_activity._update_paused_speech(speech_handle, timeout=2.0)
    audio_output.pause()
    agent_activity._session.output.set_audio_enabled(False)

    agent_activity._reconcile_playout_pause(speech_handle)

    assert audio_output._paused_at is None
    assert agent_activity._paused_speech is None


async def test_audio_forwarding_reconciles_playout_pause_before_first_frame() -> None:
    order: list[str] = []
    audio_output = MagicMock()
    audio_output.sample_rate = None
    audio_output.capture_frame = AsyncMock(side_effect=lambda _: order.append("frame"))
    audio_output.flush.side_effect = lambda: order.append("flush")

    frame = MagicMock()

    async def _frames():
        yield frame

    out = _AudioOutput(
        first_frame_fut=asyncio.Future(),
        captured_segments_before=0,
    )
    await _audio_forwarding_task(
        audio_output,
        _frames(),
        out,
        reconcile_playout_pause=lambda: order.append("reconcile"),
        tts_data=None,
    )

    assert order == ["reconcile", "frame", "flush"]
    audio_output.resume.assert_not_called()
