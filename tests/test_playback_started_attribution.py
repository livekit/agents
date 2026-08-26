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
    """Count each segment after forwarding its first frame."""

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
    """Play audio without emitting ``playback_started``."""

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await AudioOutput.capture_frame(self, frame)
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
    audio_output: AudioOutput, frames_ch: asyncio.Queue[rtc.AudioFrame | None]
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


async def test_own_playback_started_resolves_first_frame_fut() -> None:
    audio_output = FakeAudioOutput()
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)

    frames_ch.put_nowait(_make_frame())
    frames_ch.put_nowait(None)
    await task

    assert out.first_frame_fut.done()
    assert audio_output.captured_playout_segments > out.captured_segments_before
    audio_output.clear_buffer()


async def test_own_event_forwarded_before_wrapper_counts_still_resolves() -> None:
    audio_output = _ForwardFirstWrapper(FakeAudioOutput())
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)

    frames_ch.put_nowait(_make_frame())
    frames_ch.put_nowait(None)
    await task

    assert out.first_frame_fut.done()
    audio_output.clear_buffer()


async def test_stale_event_before_own_capture_is_ignored() -> None:
    audio_output = FakeAudioOutput()
    frames_ch: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
    task, out = await _drive_forwarding(audio_output, frames_ch)
    await asyncio.sleep(0)  # Allow the forwarding task to attach its listener.

    # Simulate a delayed RPC from an interrupted segment.
    audio_output.on_playback_started(created_at=time.time())

    assert not out.first_frame_fut.done()

    frames_ch.put_nowait(None)
    await task
    out.first_frame_fut.cancel()


async def test_interrupted_commit_uses_position_evidence_without_started_event() -> None:
    # A remote avatar can report playback progress before its start RPC arrives.
    audio_output = _NoStartNotifyOutput()
    speech_handle = SpeechHandle.create()

    frame_captured = asyncio.Event()

    async def _audio_source() -> AsyncIterable[rtc.AudioFrame]:
        yield _make_frame()
        frame_captured.set()
        await asyncio.Event().wait()  # Keep the stream open until interruption.

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
    await asyncio.sleep(0.05)  # Accrue playback progress.
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
        yield _make_frame()

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
        yield _make_frame()  # pragma: no cover

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
