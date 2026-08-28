"""A realtime model advertising `supports_say` serves `say()` from the generation path.

The line is still the application speaking, not the model ending a turn, so it must be
recorded as such — a delegate reads that to tell an expert's answer from its announcements.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, utils
from livekit.agents.llm import FunctionCall, GenerationCreatedEvent, MessageGeneration
from livekit.agents.voice.delegation import MESSAGE_SOURCE_KEY

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.mark.asyncio
async def test_a_said_line_is_not_the_model_ending_a_turn() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_say=True))
    assert model.capabilities.supports_say

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="t"))

        handle = session.say("your balance is 50 dollars")
        while not model.active_session.say_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[MessageGeneration]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        function_ch = utils.aio.Chan[FunctionCall]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["text"])
        message_ch.send_nowait(
            MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("your balance is 50 dollars")
        text_ch.close()
        audio_ch.close()
        function_ch.close()
        model.active_session.say_futs[0].set_result(
            GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        await asyncio.wait_for(handle, timeout=10)
        said = [
            item
            for item in session.history.items
            if item.type == "message" and item.role == "assistant"
        ]
        assert [(i.extra.get(MESSAGE_SOURCE_KEY), i.text_content) for i in said] == [
            ("say", "your balance is 50 dollars")
        ]
