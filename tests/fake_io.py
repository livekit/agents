from __future__ import annotations

import asyncio
import time

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.io import AudioInput, AudioOutput, AudioOutputCapabilities, TextOutput


class FakeAudioInput(AudioInput):
    def __init__(self) -> None:
        super().__init__(label="FakeIO")
        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._sample_rate = 16000

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._audio_ch.__anext__()

    def push(self, frame: rtc.AudioFrame | float) -> None:
        if not isinstance(frame, rtc.AudioFrame):
            num_samples = int(self._sample_rate * frame + 0.5)
            audio_frame = rtc.AudioFrame(
                data=b"\x00\x00" * num_samples,
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=num_samples,
            )
        else:
            audio_frame = frame

        self._audio_ch.send_nowait(audio_frame)


class FakeAudioOutput(AudioOutput):
    def __init__(
        self,
        *,
        next_in_chain: AudioOutput | None = None,
        sample_rate: int | None = None,
        can_pause: bool = False,
    ) -> None:
        super().__init__(
            label="FakeIO",
            next_in_chain=next_in_chain,
            sample_rate=sample_rate,
            capabilities=AudioOutputCapabilities(pause=can_pause),
        )
        self._pushed_duration = 0.0
        self._flushed = False
        self._flush_handle: asyncio.TimerHandle | None = None
        # virtual playout clock: started_at is set on first non-paused capture; paused_at
        # is set while paused; total_paused accumulates fully-closed pause intervals.
        self._started_at: float | None = None
        self._paused_at: float | None = None
        self._total_paused = 0.0

    def _played_duration(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._paused_at if self._paused_at is not None else time.time()
        return end - self._started_at - self._total_paused

    def _reset_playout(self) -> None:
        self._pushed_duration = 0.0
        self._flushed = False
        self._started_at = None
        self._paused_at = None
        self._total_paused = 0.0
        if self._flush_handle:
            self._flush_handle.cancel()
            self._flush_handle = None

    def _schedule_flush_completion(self) -> None:
        if self._flush_handle:
            self._flush_handle.cancel()
            self._flush_handle = None
        if self._paused_at is not None:
            return  # defer until resume()

        pushed_duration = self._pushed_duration
        delay = max(pushed_duration - self._played_duration(), 0.0)

        def _on_playback_finished() -> None:
            self._reset_playout()
            self.on_playback_finished(
                playback_position=pushed_duration,
                interrupted=False,
                synchronized_transcript=None,
            )

        self._flush_handle = asyncio.get_event_loop().call_later(delay, _on_playback_finished)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        self._pushed_duration += frame.duration
        if self._paused_at is not None:
            return  # frames arriving during pause are held implicitly (no side effects)

        if self._started_at is None:
            self._started_at = time.time()
            self.on_playback_started(created_at=self._started_at)

    def pause(self) -> None:
        super().pause()
        if self._paused_at is None:
            self._paused_at = time.time()
        if self._flush_handle:
            self._flush_handle.cancel()
            self._flush_handle = None

    def resume(self) -> None:
        super().resume()
        if self._paused_at is not None:
            self._total_paused += time.time() - self._paused_at
            self._paused_at = None
        # fire on_playback_started if frames arrived while paused (first frame now plays)
        if self._started_at is None and self._pushed_duration > 0:
            self._started_at = time.time()
            self.on_playback_started(created_at=self._started_at)
        if self._flushed:
            self._schedule_flush_completion()

    def flush(self) -> None:
        super().flush()
        if not self._pushed_duration:
            return
        self._flushed = True
        self._schedule_flush_completion()

    def clear_buffer(self) -> None:
        if not self._pushed_duration:
            return

        playback_position = min(self._played_duration(), self._pushed_duration)
        self._reset_playout()
        self.on_playback_finished(
            playback_position=playback_position,
            interrupted=True,
            synchronized_transcript=None,
        )


class FakeTextOutput(TextOutput):
    def __init__(self, *, next_in_chain: TextOutput | None = None) -> None:
        super().__init__(label="FakeIO", next_in_chain=next_in_chain)
        self._pushed_text = ""
        self._messages: list[str] = []

    async def capture_text(self, text: str) -> None:
        self._pushed_text += text

    def flush(self) -> None:
        self._messages.append(self._pushed_text)
        print(self._pushed_text)
        self._pushed_text = ""
