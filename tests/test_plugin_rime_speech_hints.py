"""Speech-hint attenuation must preserve newer application gain settings."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, llm
from livekit.plugins.rime._realtime_io import AudioOutput as RimeAudioOutput
from livekit.plugins.rime._realtime_session import AgentSession
from livekit.plugins.rime._realtime_types import InputSpeechHintEvent

from .fake_io import FakeAudioOutput
from .fake_rime_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


class GainOutput(RimeAudioOutput):
    def __init__(self):
        super().__init__(FakeAudioOutput())


def hint(model):
    model.active_session.emit(
        "input_speech_hint", InputSpeechHintEvent(audio_start_ms=0, gain=0.25, timeout=0.02)
    )


def stop_hint(model):
    model.active_session.emit(
        "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
    )


@pytest.mark.parametrize("application_gain", [0.6, 0.2])
async def test_speech_hint_restore_preserves_new_application_gain(application_gain):
    model = FakeRealtimeModel()
    output = GainOutput()
    output.set_gain(0.8)
    async with AgentSession(llm=model) as session:
        session.output.audio = output
        await session.start(Agent(instructions="Help"))
        hint(model)
        assert output.gain == pytest.approx(0.2)
        output.set_gain(application_gain)
        stop_hint(model)
        assert output.gain == pytest.approx(application_gain)
    assert output.gain == pytest.approx(application_gain)


async def test_repeated_hint_rebases_on_new_application_gain():
    model = FakeRealtimeModel()
    output = GainOutput()
    output.set_gain(0.8)
    async with AgentSession(llm=model) as session:
        session.output.audio = output
        await session.start(Agent(instructions="Help"))
        hint(model)
        output.set_gain(0.6)
        hint(model)
        assert output.gain == pytest.approx(0.15)
        stop_hint(model)
        assert output.gain == pytest.approx(0.6)


async def test_hint_timeout_preserves_new_application_gain_through_wrapper():
    model = FakeRealtimeModel()
    leaf = GainOutput()
    leaf.set_gain(0.8)
    wrapper = RimeAudioOutput(FakeAudioOutput(next_in_chain=leaf))
    async with AgentSession(llm=model) as session:
        session.output.audio = wrapper
        await session.start(Agent(instructions="Help"))
        hint(model)
        assert leaf.gain == pytest.approx(0.2)
        # The application can reach the leaf directly, bypassing all wrappers.
        leaf.set_gain(0.6)
        await asyncio.sleep(0.04)
        assert wrapper.gain == pytest.approx(0.6)
        hint(model)
        assert wrapper.gain == pytest.approx(0.15)
        wrapper.set_gain(0.4)
        stop_hint(model)
        assert leaf.gain == pytest.approx(0.4)


async def test_hint_restoration_does_not_change_a_replacement_leaf():
    model = FakeRealtimeModel()
    old_leaf = GainOutput()
    old_leaf.set_gain(0.8)
    new_leaf = GainOutput()
    new_leaf.set_gain(0.2)
    wrapper = RimeAudioOutput(FakeAudioOutput(next_in_chain=old_leaf))
    async with AgentSession(llm=model) as session:
        session.output.audio = wrapper
        await session.start(Agent(instructions="Help"))
        hint(model)
        # Simulate the existing sink replacement seam below a stable wrapper.
        session.output.replace_audio_tail(new_leaf)
        stop_hint(model)
        assert new_leaf.gain == pytest.approx(0.2)
        hint(model)
        assert new_leaf.gain == pytest.approx(0.05)
        stop_hint(model)
        assert new_leaf.gain == pytest.approx(0.2)


async def test_speech_hints_leave_unsupported_output_unchanged():
    model = FakeRealtimeModel()
    output = RimeAudioOutput(FakeAudioOutput(), gain=False)
    async with AgentSession(llm=model) as session:
        session.output.audio = output
        await session.start(Agent(instructions="Help"))
        hint(model)
        stop_hint(model)
        assert output.gain == 1.0
