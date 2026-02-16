from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    NotGivenOr,
    utils,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.transcription.synchronizer import (
    TranscriptSynchronizer,
    _SyncedAudioOutput,
)

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_tts import FakeTTS, FakeTTSResponse
from .fake_vad import FakeVAD


def create_session(
    actions: FakeActions,
    *,
    speed_factor: float = 1.0,
    extra_kwargs: dict[str, Any] | None = None,
) -> AgentSession:
    user_speeches = actions.get_user_speeches(speed_factor=speed_factor)
    llm_responses = actions.get_llm_responses(speed_factor=speed_factor)
    tts_responses = actions.get_tts_responses(speed_factor=speed_factor)

    stt = FakeSTT(fake_user_speeches=user_speeches)
    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=user_speeches,
            min_silence_duration=0.5 / speed_factor,
            min_speech_duration=0.05 / speed_factor,
        ),
        stt=stt,
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        min_interruption_duration=0.5 / speed_factor,
        min_endpointing_delay=0.5 / speed_factor,
        max_endpointing_delay=6.0 / speed_factor,
        false_interruption_timeout=2.0 / speed_factor,
        **(extra_kwargs or {}),
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
    return session


async def run_session(session: AgentSession, agent: Agent, *, drain_delay: float = 0.2) -> float:
    stt = session.stt
    audio_input = session.input.audio
    assert isinstance(stt, FakeSTT)
    assert isinstance(audio_input, FakeAudioInput)

    transcription_sync: TranscriptSynchronizer | None = None
    if isinstance(session.output.audio, _SyncedAudioOutput):
        transcription_sync = session.output.audio._synchronizer

    await session.start(agent)

    # start the fake vad and stt
    t_origin = time.time()
    audio_input.push(0.1)

    # wait for the user speeches to be processed
    await stt.fake_user_speeches_done

    await asyncio.sleep(drain_delay)
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()

    if transcription_sync is not None:
        await transcription_sync.aclose()

    return t_origin


class FakeActions:
    def __init__(self) -> None:
        self._items: list[FakeUserSpeech | FakeLLMResponse | FakeTTSResponse] = []

    def add_user_speech(
        self, start_time: float, end_time: float, transcript: str, *, stt_delay: float = 0.2
    ) -> None:
        self._items.append(
            FakeUserSpeech(
                start_time=start_time,
                end_time=end_time,
                transcript=transcript,
                stt_delay=stt_delay,
            )
        )

    def add_llm(
        self,
        content: str,
        tool_calls: list[FunctionToolCall] | None = None,
        *,
        input: NotGivenOr[str] = NOT_GIVEN,
        ttft: float = 0.1,
        duration: float = 0.3,
    ) -> None:
        if (
            not utils.is_given(input)
            and self._items
            and isinstance(self._items[-1], FakeUserSpeech)
        ):
            # use the last user speech as input
            input = self._items[-1].transcript

        if not utils.is_given(input):
            raise ValueError("input is required or previous item needs to be a user speech")

        self._items.append(
            FakeLLMResponse(
                content=content,
                input=input,
                ttft=ttft,
                duration=duration,
                tool_calls=tool_calls or [],
            )
        )

    def add_tts(
        self,
        audio_duration: float,
        *,
        input: NotGivenOr[str] = NOT_GIVEN,
        ttfb: float = 0.2,
        duration: float = 0.3,
    ) -> None:
        if (
            not utils.is_given(input)
            and self._items
            and isinstance(self._items[-1], FakeLLMResponse)
        ):
            input = self._items[-1].content

        if not utils.is_given(input):
            raise ValueError("input is required or previous item needs to be a llm response")

        self._items.append(
            FakeTTSResponse(
                audio_duration=audio_duration,
                input=input,
                ttfb=ttfb,
                duration=duration,
            )
        )

    def get_user_speeches(self, *, speed_factor: float = 1.0) -> list[FakeUserSpeech]:
        return [item.speed_up(speed_factor) for item in self._items if item.type == "user_speech"]

    def get_llm_responses(self, *, speed_factor: float = 1.0) -> list[FakeLLMResponse]:
        return [item.speed_up(speed_factor) for item in self._items if item.type == "llm"]

    def get_tts_responses(self, *, speed_factor: float = 1.0) -> list[FakeTTSResponse]:
        return [item.speed_up(speed_factor) for item in self._items if item.type == "tts"]
