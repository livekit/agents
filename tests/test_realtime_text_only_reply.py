"""A realtime model that answers an audio turn in text.

Gemini Live sometimes closes an audio-modality turn with a text part and no audio. The
caller hears nothing. With a TTS on the session the framework already speaks the text;
without one, the agent asks the model once to say it aloud, and never asks twice.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, utils
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_SAMPLE_RATE = 24000


def _generation(*, response_id: str, text: str, audio_duration: float) -> GenerationCreatedEvent:
    """One message; text only when audio_duration is 0, audio otherwise."""
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"] if audio_duration > 0 else ["text"])

    message_ch.send_nowait(
        MessageGeneration(
            message_id=f"{response_id}-message",
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    text_ch.send_nowait(text)
    text_ch.close()
    if audio_duration > 0:
        samples = int(_SAMPLE_RATE * audio_duration)
        audio_ch.send_nowait(
            rtc.AudioFrame(
                data=b"\x00\x01" * samples,
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=samples,
            )
        )
    audio_ch.close()
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=True,
        response_id=response_id,
    )


async def _wait_for_reply_futs(rt: FakeRealtimeSession, count: int) -> None:
    for _ in range(500):
        if len(rt._reply_futs) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} generate_reply calls, got {len(rt._reply_futs)}")


def _record_states(session: AgentSession) -> list[str]:
    states: list[str] = []
    session.on("agent_state_changed", lambda ev: states.append(ev.new_state))
    return states


async def _settle(rt: FakeRealtimeSession, *, ticks: int = 50) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0.01)


async def test_text_only_reply_is_asked_aloud_once_without_tts() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        states = _record_states(session)
        await session.start(Agent(instructions="test"))
        rt = model.active_session

        handle = session.generate_reply()
        await _wait_for_reply_futs(rt, 1)
        rt._reply_futs[0].set_result(_generation(response_id="r1", text="Okay.", audio_duration=0))

        await _wait_for_reply_futs(rt, 2)
        assert rt.generate_reply_instructions[1], "the respeak request carries instructions"
        rt._reply_futs[1].set_result(
            _generation(response_id="r2", text="Okay.", audio_duration=0.2)
        )

        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)
        await _settle(rt)
        assert rt.generate_reply_calls == 2
        assert "speaking" in states


async def test_text_only_retry_is_not_retried_again() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="test"))
        rt = model.active_session

        handle = session.generate_reply()
        await _wait_for_reply_futs(rt, 1)
        rt._reply_futs[0].set_result(_generation(response_id="r1", text="Okay.", audio_duration=0))
        await _wait_for_reply_futs(rt, 2)
        rt._reply_futs[1].set_result(_generation(response_id="r2", text="Okay.", audio_duration=0))

        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)
        await _settle(rt)
        assert rt.generate_reply_calls == 2


async def test_text_only_reply_is_spoken_by_tts_without_asking_again() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    async with AgentSession(llm=model, tts=FakeTTS(fake_audio_duration=0.3)) as session:
        session.output.audio = FakeAudioOutput()
        states = _record_states(session)
        await session.start(Agent(instructions="test"))
        rt = model.active_session

        handle = session.generate_reply()
        await _wait_for_reply_futs(rt, 1)
        rt._reply_futs[0].set_result(_generation(response_id="r1", text="Okay.", audio_duration=0))

        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)
        await _settle(rt)
        assert "speaking" in states
        assert rt.generate_reply_calls == 1
