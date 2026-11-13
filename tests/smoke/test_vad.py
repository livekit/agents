"""A VAD written in Python, as a plugin is, drives the native session: its speech opens and ends
the person's turns and interrupts the agent. The shipped Silero VAD hears a real voice."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents import Agent, VoiceSession
from livekit.agents.utils import aio
from livekit.agents.utils.media import audio_frames_from_file
from livekit.agents.vad import VAD
from livekit.plugins import silero
from tests.fakes import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    audio_frame,
    until,
)

VOICE = Path(__file__).parents[1] / "fixtures" / "change-sophie.opus"


def listening_session(
    microphone: FakeAudioInput,
    vad: VAD | None = None,
    stt: FakeSTT | None = None,
    llm: FakeLLM | None = None,
    tts: FakeTTS | None = None,
) -> VoiceSession:
    """A session hearing ``microphone``, whose turns the VAD (``FakeVAD`` by default) detects."""
    session = VoiceSession(
        vad=vad or FakeVAD(),
        stt=stt or FakeSTT(),
        llm=llm or FakeLLM(),
        tts=tts or FakeTTS(),
        turn_handling={"turn_detection": "vad", "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
        # the speech these tests hear has no words left to transcribe once the session closes
        session_close_transcript_timeout=0,
    )
    session.input.audio = microphone
    session.output.audio = FakeAudioOutput()
    return session


async def talk(microphone: FakeAudioInput, voice: AsyncIterable[rtc.AudioFrame]) -> None:
    """Plays ``voice`` into the microphone, as a person speaks, then pauses."""
    async for frame in voice:
        microphone.push(frame)
    microphone.speak(0)


async def loud(seconds: float) -> AsyncIterable[rtc.AudioFrame]:
    for _ in range(round(seconds / 0.1)):
        yield audio_frame(0.1)


async def test_python_vad_speech_opens_and_ends_the_users_turn() -> None:
    """Speech the VAD hears makes the person speaking; the quiet after it hands the floor back."""
    microphone = FakeAudioInput()
    session = listening_session(microphone)
    states: list[str] = []
    session.on("user_state_changed", lambda ev: states.append(ev.new_state))

    async with session:
        await session.start(Agent(instructions="Listen."))
        await talk(microphone, loud(0.5))
        await until(lambda: states[-1:] == ["listening"], f"user states: {states}")

    assert states == ["speaking", "listening"]


async def test_python_vad_speech_interrupts_the_agent() -> None:
    """The person talking over the agent, as the VAD hears it, interrupts its speech."""
    microphone = FakeAudioInput()
    session = listening_session(microphone, tts=FakeTTS(seconds=10))

    async with session:
        await session.start(Agent(instructions="Talk."))
        speech = session.say("A very long story.")
        await until(lambda: session.agent_state == "speaking", "the agent never spoke")

        await talk(microphone, loud(1.0))

        await speech.wait_for_playout()

    assert speech.interrupted


async def test_silero_hears_a_recorded_voice_and_drives_its_turn() -> None:
    """Silero finds the speech in a real recording; the turn it delimits is transcribed once
    and answered."""
    microphone = FakeAudioInput()
    session = listening_session(
        microphone,
        vad=silero.VAD.load(),
        stt=FakeSTT(["change sophie"], streaming=False),
        llm=FakeLLM({"change sophie": "Done."}),
    )

    def said() -> list[tuple[str, str | None]]:
        return [(i.role, i.text_content) for i in session.history.items if i.type == "message"]

    async with session:
        await session.start(Agent(instructions="Answer."))
        await talk(microphone, audio_frames_from_file(str(VOICE), sample_rate=24000))
        await until(lambda: ("assistant", "Done.") in said(), f"no answer: {said()}")

    assert said() == [("user", "change sophie"), ("assistant", "Done.")]


async def test_closing_a_stream_whose_reader_stopped_reading_completes() -> None:
    """A VAD stream nobody reads stops taking audio once its input is full, and closing it then
    still returns instead of waiting for room in that input."""
    stream = FakeVAD().stream()
    with pytest.raises(aio.ChanFull):
        while True:
            stream.push_frame(audio_frame(0.01))
            await asyncio.sleep(0)

    async with asyncio.timeout(1):
        await stream.aclose()

    assert stream.closed
