from __future__ import annotations

import asyncio
import time

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.io import AudioInput, AudioOutput, TextOutput


class FakeAudioInput(AudioInput):
    def __init__(self) -> None:
        super().__init__()
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
        self, *, next_in_chain: AudioOutput | None = None, sample_rate: int | None = None
    ) -> None:
        super().__init__(next_in_chain=next_in_chain, sample_rate=sample_rate)
        self._start_time = 0.0
        self._pushed_duration = 0.0
        self._flush_handle: asyncio.TimerHandle | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if not self._pushed_duration:
            self._start_time = time.time()
        self._pushed_duration += frame.duration

    def flush(self) -> None:
        super().flush()
        if not self._pushed_duration:
            return

        def _on_playback_finished() -> None:
            self.on_playback_finished(
                playback_position=self._pushed_duration,
                interrupted=False,
                synchronized_transcript=None,
            )
            self._pushed_duration = 0.0

        delay = self._pushed_duration - (time.time() - self._start_time)
        if self._flush_handle:
            self._flush_handle.cancel()
        self._flush_handle = asyncio.get_event_loop().call_later(delay, _on_playback_finished)

    def clear_buffer(self) -> None:
        if not self._pushed_duration:
            return

        if self._flush_handle:
            self._flush_handle.cancel()

        self._flush_handle = None
        self.on_playback_finished(
            playback_position=min(self._pushed_duration, time.time() - self._start_time),
            interrupted=True,
            synchronized_transcript=None,
        )
        self._pushed_duration = 0.0


class FakeTextOutput(TextOutput):
    def __init__(self, *, next_in_chain: TextOutput | None = None) -> None:
        super().__init__(next_in_chain=next_in_chain)
        self._pushed_text = ""
        self._messages: list[str] = []

    async def capture_text(self, text: str) -> None:
        self._pushed_text += text

    def flush(self) -> None:
        self._messages.append(self._pushed_text)
        print(self._pushed_text)
        self._pushed_text = ""
