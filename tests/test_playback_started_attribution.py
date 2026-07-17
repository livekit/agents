"""Tests for cross-segment attribution of playback_started events (AGT-3147).

The shared AudioOutput emits playback_started with no segment identity, so the
per-segment listener in perform_audio_forwarding and the interrupted-commit gates
must decide whether an event/position belongs to their own segment. See
_AudioOutput.own_segment_index and the listener guard in perform_audio_forwarding.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable

import pytest

from livekit import rtc
from livekit.agents.voice.generation import forward_generation, perform_audio_forwarding
from livekit.agents.voice.io import AudioOutput
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000


def _make_frame(duration: float = 0.1) -> rtc.AudioFrame:
    num_samples = int(SAMPLE_RATE * duration + 0.5)
    return rtc.AudioFrame(
        data=b"\x00\x00" * num_samples,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=num_samples,
    )


class _ForwardFirstWrapper(AudioOutput):
    """Mimics _SyncedAudioOutput / recorder ordering.

    Forwards to the leaf (which emits playback_started synchronously inside the
    first capture) BEFORE counting its own segment, so the forwarded event fires
    while this output's counter still reads the pre-capture snapshot.
    """

    def __init__(self, next_in_chain: AudioOutput) -> None:
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
    """A sink whose playback-started notification is delayed/lost.

    Simulates a remote (avatar) output that delivers playback_started via RPC:
    frames are accepted and counted, playback genuinely progresses (so the
    playback-finished event reports a real position), but no playback_started
    is emitted locally.
    """

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await AudioOutput.capture_frame(self, frame)  # count the playout segment
        self._pushed_duration += frame.duration
        if self._started_at is None:
            self._started_at = time.time()  # playing, but never notifies


async def _drive_forwarding(
    audio_output: AudioOutput, frames_ch: asyncio.Queue[rtc.AudioFrame | None]
):
    async def _source() -> AsyncIterable[rtc.AudioFrame]:
        while True:
            frame = await frames_ch.get()
            if frame is None:
                return
            yield frame

    return perform_audio_forwarding(audio_output=audio_output, tts_output=_source())


async def test_own_playback_started_resolves_first_frame_fut() -> None:
    audio_output = FakeAudioOutput()
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)

    frames_ch.put_nowait(_make_frame())
    frames_ch.put_nowait(None)
    await task

    assert out.first_frame_fut.done()
    assert out.own_segment_index == 1
    audio_output.clear_buffer()


async def test_own_event_forwarded_before_wrapper_counts_still_resolves() -> None:
    # Chained outputs (transcript sync, recorder) forward the leaf's synchronous
    # playback_started before counting their own segment: the event must still be
    # attributed to the segment whose first capture is in flight.
    audio_output = _ForwardFirstWrapper(FakeAudioOutput())
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)

    frames_ch.put_nowait(_make_frame())
    frames_ch.put_nowait(None)
    await task

    assert out.first_frame_fut.done()
    assert out.own_segment_index == 1
    audio_output.clear_buffer()


async def test_stale_event_before_own_capture_is_ignored() -> None:
    audio_output = FakeAudioOutput()
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)
    await asyncio.sleep(0)  # let the forwarding task attach the listener

    # a stale event (e.g. an avatar RPC from a previous, interrupted segment)
    # arrives before this segment captured anything
    audio_output.on_playback_started(created_at=time.time())

    assert not out.first_frame_fut.done()

    frames_ch.put_nowait(None)
    await task
    out.first_frame_fut.cancel()


async def test_event_after_foreign_segment_bump_is_ignored() -> None:
    # Segment A captures into a paused output (counted, but nothing plays and no
    # playback_started fires). A foreign segment then bumps the counter. A later
    # event must NOT resolve A's future: the counter no longer points at A.
    audio_output = FakeAudioOutput(can_pause=True)
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)
    await asyncio.sleep(0)  # forwarding task calls resume(); pause after it
    audio_output.pause()

    frames_ch.put_nowait(_make_frame())
    frames_ch.put_nowait(None)
    await task

    assert out.own_segment_index == 1
    assert not out.first_frame_fut.done()

    # a foreign segment (next speech) opens on the shared output
    await audio_output.capture_frame(_make_frame())
    assert audio_output.captured_playout_segments == 2

    # the foreign segment's playback starts
    audio_output.on_playback_started(created_at=time.time())

    assert not out.first_frame_fut.done()
    out.first_frame_fut.cancel()


async def test_interrupted_commit_uses_position_evidence_without_started_event() -> None:
    # Avatar-style race: frames genuinely played, but the started notification
    # never arrived before the interruption. The playback position reported by
    # the finished event is proof of partial playback.
    audio_output = _NoStartNotifyOutput()
    speech_handle = SpeechHandle.create()

    frame_captured = asyncio.Event()

    async def _audio_source() -> AsyncIterable[rtc.AudioFrame]:
        yield _make_frame()
        frame_captured.set()
        await asyncio.Event().wait()  # keep the TTS stream open until interrupted

    forward_task = asyncio.create_task(
        forward_generation(
            speech_handle=speech_handle,
            audio_output=audio_output,
            text_output=None,
            audio_source=_audio_source(),
            text_source=None,
            on_first_frame=lambda _fut, _out: None,
        )
    )

    await asyncio.wait_for(frame_captured.wait(), timeout=5)
    await asyncio.sleep(0.05)  # accrue some playback position
    speech_handle.interrupt()
    out = await asyncio.wait_for(forward_task, timeout=5)

    assert out.audio_out is not None
    assert not out.audio_out.first_frame_fut.done() or out.audio_out.first_frame_fut.cancelled()
    assert out.audio_out.own_segment_index == 1
    assert out.played == "partial"
    assert out.playback_position > 0


async def test_interrupted_commit_stays_skipped_without_any_capture() -> None:
    # Interrupted before any frame was counted: no started event, no position
    # evidence — the segment must stay "skipped" (nothing reached the user).
    audio_output = FakeAudioOutput()
    speech_handle = SpeechHandle.create()

    async def _audio_source() -> AsyncIterable[rtc.AudioFrame]:
        await asyncio.Event().wait()  # never yields
        yield _make_frame()  # pragma: no cover

    forward_task = asyncio.create_task(
        forward_generation(
            speech_handle=speech_handle,
            audio_output=audio_output,
            text_output=None,
            audio_source=_audio_source(),
            text_source=None,
            on_first_frame=lambda _fut, _out: None,
        )
    )

    await asyncio.sleep(0.05)
    speech_handle.interrupt()
    out = await asyncio.wait_for(forward_task, timeout=5)

    assert out.audio_out is not None
    assert out.audio_out.own_segment_index is None
    assert out.played == "skipped"
