from __future__ import annotations

import asyncio
import time

from livekit import rtc
from livekit.agents.voice import io
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer


class _ManualAudioOutput(io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="ManualAudioOutput",
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._pushed_duration = 0.0
        self._started = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self._pushed_duration += frame.duration
        if not self._started:
            self._started = True
            self.on_playback_started(created_at=time.time())

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        if self._pushed_duration:
            self.finish(interrupted=True)

    def finish(self, *, interrupted: bool = False) -> None:
        playback_position = self._pushed_duration
        self._pushed_duration = 0.0
        self._started = False
        self.on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
        )


class _CapturingTextOutput(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="CapturingTextOutput", next_in_chain=None)
        self._current_text = ""
        self.messages: list[str] = []

    async def capture_text(self, text: str) -> None:
        self._current_text += text

    def flush(self) -> None:
        self.messages.append(self._current_text)
        self._current_text = ""


def _audio_frame(duration: float = 0.02) -> rtc.AudioFrame:
    sample_rate = 16000
    samples = int(sample_rate * duration)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def test_empty_audio_flush_after_advance_does_not_end_fresh_segment() -> None:
    audio_output = _ManualAudioOutput()
    text_output = _CapturingTextOutput()
    synchronizer = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=text_output,
    )

    try:
        await synchronizer.audio_output.capture_frame(_audio_frame())
        await synchronizer.text_output.capture_text("hello")
        synchronizer.audio_output.flush()
        synchronizer.text_output.flush()

        assert len(synchronizer._pending_impls) == 1
        assert not synchronizer._impl.audio_input_ended

        synchronizer.audio_output.flush()
        await synchronizer.barrier()

        assert not synchronizer._impl.audio_input_ended
    finally:
        await synchronizer.aclose()


async def test_playback_finished_waits_for_text_input_before_emitting_transcript() -> None:
    audio_output = _ManualAudioOutput()
    text_output = _CapturingTextOutput()
    synchronizer = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=text_output,
        speed=100.0,
    )
    playback_events: list[io.PlaybackFinishedEvent] = []
    synchronizer.audio_output.on("playback_finished", playback_events.append)

    try:
        await synchronizer.audio_output.capture_frame(_audio_frame())
        await synchronizer.text_output.capture_text("hello")
        synchronizer.audio_output.flush()
        audio_output.finish()

        waiter = asyncio.create_task(synchronizer.audio_output.wait_for_playout())
        await asyncio.sleep(0)

        assert not waiter.done()
        assert playback_events == []

        synchronizer.text_output.flush()
        playback_ev = await asyncio.wait_for(waiter, timeout=1.0)

        assert playback_ev.synchronized_transcript == "hello"
        assert playback_events == [playback_ev]
    finally:
        await synchronizer.aclose()


async def test_pause_resume_applies_to_pending_segments() -> None:
    audio_output = _ManualAudioOutput()
    text_output = _CapturingTextOutput()
    synchronizer = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=text_output,
    )

    try:
        await synchronizer.audio_output.capture_frame(_audio_frame())
        await synchronizer.text_output.capture_text("hello")
        synchronizer.audio_output.flush()
        synchronizer.text_output.flush()

        pending = synchronizer._pending_impls[0]

        synchronizer.audio_output.pause()
        assert not pending._output_enabled_ev.is_set()

        synchronizer.audio_output.resume()
        assert pending._output_enabled_ev.is_set()
    finally:
        await synchronizer.aclose()
