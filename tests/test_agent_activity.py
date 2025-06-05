from __future__ import annotations

import asyncio
import time
from typing import Any

from fake_llm import FakeLLM, FakeLLMResponse
from fake_stt import FakeSTT, FakeUserSpeech
from fake_tts import FakeTTS, FakeTTSResponse
from fake_vad import FakeVAD

from livekit import rtc
from livekit.agents import Agent, AgentSession, MetricsCollectedEvent, utils
from livekit.agents.voice.io import AudioInput, AudioOutput, PlaybackFinishedEvent, TextOutput
from livekit.agents.voice.transcription import TranscriptSynchronizer

tracing = [
    {
        "type": "user_speech",
        "start_time": 0.5,
        "end_time": 2.5,
        "transcript": "Hello, how are you?",
        "stt_delay": 0.2,
    },  # +0.55s (min_silence_duration)
    {
        "type": "llm",
        "content": "I'm doing well, thank you!",
        "ttft": 0.1,
        "duration": 0.2,
    },  # +0.2s (duration, start tts after flush)
    {
        "type": "tts",
        "audio_duration": 2.0,
        "ttfb": 0.1,
        "duration": 0.2,
    },  # +0.1s (ttfb)
]


def parse_test_cases(
    tracing: list[dict[str, Any]], *, speed_factor: float = 1.0
) -> list[FakeUserSpeech | FakeLLMResponse | FakeTTSResponse]:
    items: list[FakeUserSpeech | FakeLLMResponse | FakeTTSResponse] = []
    prev_item: FakeUserSpeech | FakeLLMResponse | FakeTTSResponse | None = None
    for data in tracing:
        if not (type := data.get("type")):
            raise ValueError("type is required")

        if type == "llm" and "input" not in data and isinstance(prev_item, FakeUserSpeech):
            data["input"] = prev_item.transcript
        elif type == "tts" and "input" not in data and isinstance(prev_item, FakeLLMResponse):
            data["input"] = prev_item.content

        if type == "user_speech":
            item = FakeUserSpeech.model_validate(data)
        elif type == "llm":
            item = FakeLLMResponse.model_validate(data)
        elif type == "tts":
            item = FakeTTSResponse.model_validate(data)
        else:
            raise ValueError(f"unknown type: {type}")

        items.append(item)
        prev_item = item

    if speed_factor != 1.0:
        items = [item.speed_up(speed_factor) for item in items]

    return items


async def entrypoint() -> None:
    agent = Agent(instructions="You are a helpful assistant.")

    speed_factor = 1.0
    tracing_items = parse_test_cases(tracing, speed_factor=speed_factor)
    user_speeches = [item for item in tracing_items if isinstance(item, FakeUserSpeech)]
    llm_responses = [item for item in tracing_items if isinstance(item, FakeLLMResponse)]
    tts_responses = [item for item in tracing_items if isinstance(item, FakeTTSResponse)]

    session = AgentSession(
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.55 / speed_factor,
            min_speech_duration=0.05 / speed_factor,
        ),
        stt=FakeSTT(fake_user_speeches=user_speeches),
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        min_interruption_duration=0.5 / speed_factor,
        min_endpointing_delay=0.5 / speed_factor,
        max_endpointing_delay=6.0 / speed_factor,
    )

    # setup io with transcription sync
    audio_input = FakeAudioInput()
    audio_output = FakeAudioOutput()
    transcription_output = FakeTextOutput()

    transcript_sync = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=transcription_output,
        speed=speed_factor,
    )
    session.input.audio = audio_input
    session.output.audio = transcript_sync.audio_output
    session.output.transcription = transcript_sync.text_output

    start_time = time.time()
    audio_input.push_empty_audio(0.1)

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        print(ev.metrics)

    @session.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        print(f"playback_finished: {time.time() - start_time}: {ev}")

    await session.start(agent)

    await asyncio.sleep(10)
    if session.current_speech:
        await session.current_speech


class FakeAudioInput(AudioInput):
    def __init__(self) -> None:
        super().__init__()
        self._audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._sample_rate = 16000

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._audio_ch.__anext__()

    def push(self, frame: rtc.AudioFrame) -> None:
        self._audio_ch.send_nowait(frame)

    def push_empty_audio(self, duration: float) -> None:
        num_samples = int(self._sample_rate * duration + 0.5)
        self._audio_ch.send_nowait(
            rtc.AudioFrame(
                data=b"\x00\x00" * num_samples,
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=num_samples,
            )
        )


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


if __name__ == "__main__":
    asyncio.run(entrypoint())
