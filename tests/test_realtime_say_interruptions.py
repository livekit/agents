from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import Agent, AgentSession, llm, utils

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("supports_uninterruptible_say", [False, True])
@pytest.mark.parametrize("allow_interruptions", [False, True])
async def test_realtime_say_preserves_supported_interruptibility(
    supports_uninterruptible_say: bool,
    allow_interruptions: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(supports_uninterruptible_say=supports_uninterruptible_say)
    )
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))
        speech = session.say(
            "This call is being recorded.", allow_interruptions=allow_interruptions
        )
        expected = allow_interruptions or not supports_uninterruptible_say
        assert speech.allow_interruptions is expected

        for _ in range(500):
            if model.active_session.say_futs:
                break
            await asyncio.sleep(0)
        assert model.active_session.say_allow_interruptions == [expected]

        activity = session._activity
        assert activity is not None
        assert activity._current_speech is speech
        with caplog.at_level(logging.ERROR, logger="livekit.agents"):
            activity._on_input_speech_started(llm.InputSpeechStartedEvent())
        assert speech.interrupted is expected
        assert model.active_session.interrupted is expected
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]

        messages = utils.aio.Chan[llm.MessageGeneration]()
        functions = utils.aio.Chan[llm.FunctionCall]()
        messages.close()
        functions.close()
        model.active_session.say_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=messages, function_stream=functions, user_initiated=True
            )
        )
        await asyncio.wait_for(speech.wait_for_playout(), timeout=5)


async def test_uninterruptible_say_capability_does_not_apply_to_supplied_audio() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_uninterruptible_say=True))
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))
        audio = utils.aio.Chan()
        audio.close()
        speech = session.say("This call is being recorded.", audio=audio, allow_interruptions=False)
        assert speech.allow_interruptions is True
        await asyncio.wait_for(speech.wait_for_playout(), timeout=5)
        assert not model.active_session.say_calls


async def test_uninterruptible_say_capability_does_not_apply_to_generated_replies() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_uninterruptible_say=True))
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))
        speech = session.generate_reply(allow_interruptions=False)
        assert speech.allow_interruptions is True
