"""A requested realtime reply waits for the model off the floor, so a generation the model
produces meanwhile plays first, and the reply takes its turn once it arrives."""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, utils
from livekit.agents.llm import GenerationCreatedEvent, MessageGeneration

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = pytest.mark.unit


def _generation(text: str, *, user_initiated: bool) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])
    message_ch.send_nowait(
        MessageGeneration(
            message_id=utils.shortuuid("message-"),
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    text_ch.send_nowait(text)
    text_ch.close()
    audio_ch.close()
    function_ch = utils.aio.Chan()
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=user_initiated,
        response_id=utils.shortuuid("response-"),
    )


async def _assistant_texts(session: AgentSession) -> list[str]:
    return [item.text_content or "" for item in session.history.items if item.type == "message"]


async def test_a_generation_arriving_while_the_reply_is_awaited_plays_first() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="test"))
        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        model.active_session.emit("generation_created", _generation("late", user_initiated=False))
        for _ in range(200):
            if "late" in await _assistant_texts(session):
                break
            await asyncio.sleep(0.01)
        assert "late" in await _assistant_texts(session)
        assert not handle.done()
        assert session.agent_state == "thinking"

        model.active_session._reply_futs[0].set_result(_generation("reply", user_initiated=True))
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)
        assert await _assistant_texts(session) == ["late", "reply"]


async def test_user_activity_cancels_a_reply_the_model_is_still_working_on() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=False))
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="test"))
        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)
        activity = session._activity
        assert activity is not None
        activity.interruption_by_audio_activity_enabled = True

        activity._interrupt_by_audio_activity()
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)
        assert handle.interrupted
        assert model.active_session._reply_futs[0].cancelled()
