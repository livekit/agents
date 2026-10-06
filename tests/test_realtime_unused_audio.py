"""Realtime audio must be consumed even when the application has no audio output."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("disabled", [False, True])
async def test_unused_audio_is_drained(disabled: bool) -> None:
    model = FakeRealtimeModel()
    async with AgentSession(llm=model) as session:
        if disabled:
            session.output.audio = FakeAudioOutput()
            session.output.set_audio_enabled(False)
        await session.start(Agent(instructions="test"))
        speech = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        messages = utils.aio.Chan[llm.MessageGeneration]()
        functions = utils.aio.Chan[llm.FunctionCall]()
        text = utils.aio.Chan[str]()
        audio = utils.aio.Chan[rtc.AudioFrame](maxsize=1)
        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(["text", "audio"])
        messages.send_nowait(
            llm.MessageGeneration(
                message_id="message", text_stream=text, audio_stream=audio, modalities=modalities
            )
        )
        messages.close()
        functions.close()
        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                response_id="response",
                message_stream=messages,
                function_stream=functions,
                user_initiated=True,
            )
        )
        frame = rtc.AudioFrame(
            data=b"\0\0" * 480, sample_rate=24000, num_channels=1, samples_per_channel=480
        )

        async def produce() -> None:
            for _ in range(300):
                await audio.send(frame)
            audio.close()
            text.send_nowait("Complete reply")
            text.close()

        try:
            await asyncio.wait_for(produce(), 1)
            await asyncio.wait_for(speech.wait_for_playout(), 1)
        finally:
            audio.close()
            text.close()
        assert any(
            item.type == "message" and item.text_content == "Complete reply"
            for item in session.history.items
        )
