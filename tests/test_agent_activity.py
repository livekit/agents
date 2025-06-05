from __future__ import annotations

import asyncio
import time

from fake_llm import FakeLLM, FakeLLMResponse
from fake_stt import FakeSTT, FakeUserSpeech
from fake_tts import FakeTTS, FakeTTSResponse
from fake_vad import FakeVAD

from livekit import rtc
from livekit.agents import Agent, AgentSession, utils
from livekit.agents.voice.io import AudioInput, AudioOutput, TextOutput

fake_user_speeches = [
    {
        "start_time": 0.5,
        "end_time": 2.5,
        "transcript": "Hello, how are you?",
        "stt_delay": 0.2,
    },
    {
        "start_time": 5.0,
        "end_time": 6.0,
        "transcript": "Tell me a joke",
        "stt_delay": 0.2,
    }
]

fake_llm_responses = [
    {
        "input": "Hello, how are you?",
        "content": "I'm doing well, thank you!",
        "ttft": 0.1,
        "duration": 0.2,
    },
    {
        "input": "Tell me a joke",
        "content": "Why did the chicken cross the road? To get to the other side!",
        "ttft": 0.1,
        "duration": 0.2,
    },
]

fake_tts_responses = [
    {
        "input": "I'm doing well, thank you!",
        "audio_duration": 2.0,
        "ttfb": 0.1,
        "duration": 0.2,
    },
    {
        "input": "Why did the chicken cross the road? To get to the other side!",
        "audio_duration": 5.0,
        "ttfb": 0.2,
        "duration": 0.3,
    },
]


async def entrypoint() -> None:
    agent = Agent(instructions="You are a helpful assistant.")

    speed_factor = 5.0
    user_speeches = [
        FakeUserSpeech.model_validate(r).speed_up(speed_factor) for r in fake_user_speeches
    ]
    llm_responses = [
        FakeLLMResponse.model_validate(r).speed_up(speed_factor) for r in fake_llm_responses
    ]
    tts_responses = [
        FakeTTSResponse.model_validate(r).speed_up(speed_factor) for r in fake_tts_responses
    ]

    session = AgentSession(
        vad=FakeVAD(fake_user_speeches=user_speeches),
        stt=FakeSTT(fake_user_speeches=user_speeches),
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
    )

    audio_input = FakeAudioInput()
    audio_output = FakeAudioOutput()
    transcription_output = FakeTextOutput()
    session.input.audio = audio_input
    session.output.audio = audio_output
    session.output.transcription = transcription_output

    await session.start(agent)

    audio_input.push_empty_audio(0.1)

    await asyncio.sleep(5)


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
