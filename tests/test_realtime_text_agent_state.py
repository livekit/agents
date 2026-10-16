import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils

from .fake_io import FakeAudioOutput, FakeTextOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit


async def _provide_text_reply(session: FakeRealtimeSession) -> None:
    """Deliver a text-only response through the fake provider's reply future."""
    while not session._reply_futs:
        await asyncio.sleep(0)

    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    text = utils.aio.Chan[str]()
    audio = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["text"])
    messages.send_nowait(
        llm.MessageGeneration(
            message_id="message-id",
            text_stream=text,
            audio_stream=audio,
            modalities=modalities,
        )
    )
    messages.close()
    functions.close()
    text.send_nowait("Hello")
    text.close()
    audio.close()
    session._reply_futs[0].set_result(
        llm.GenerationCreatedEvent(
            message_stream=messages,
            function_stream=functions,
            user_initiated=True,
            response_id="response-id",
        )
    )


async def test_realtime_text_reply_returns_to_listening_without_audio_output() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with AgentSession(llm=model) as session:
        session.output.transcription = FakeTextOutput()
        states: list[str] = []
        session.on("agent_state_changed", lambda ev: states.append(ev.new_state))
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        await asyncio.wait_for(_provide_text_reply(model.active_session), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

        assert reply.exception() is None
        assert states == ["listening", "speaking", "listening"]
        assert session.agent_state == "listening"


async def test_realtime_text_reply_returns_to_listening_with_audio_disabled() -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        session.output.set_audio_enabled(False)
        session.output.transcription = FakeTextOutput()
        await session.start(Agent(instructions="test"))

        reply = session.generate_reply()
        await asyncio.wait_for(_provide_text_reply(model.active_session), timeout=5)
        await asyncio.wait_for(reply.wait_for_playout(), timeout=5)

        assert reply.exception() is None
        assert session.agent_state == "listening"
