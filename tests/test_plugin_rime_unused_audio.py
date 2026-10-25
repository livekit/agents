"""Realtime audio must be consumed even when the application has no audio output."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, llm, utils
from livekit.plugins.rime._realtime_session import AgentSession
from livekit.plugins.rime._realtime_types import GenerationCreatedEvent

from .fake_io import FakeAudioOutput
from .fake_rime_realtime import FakeRealtimeModel

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
            GenerationCreatedEvent(
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


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("open_functions", [False, True])
@pytest.mark.parametrize("open_text", [False, True])
async def test_audio_drain_failure_records_error_and_cleans_up(
    requested, open_functions, open_text
):
    model = FakeRealtimeModel()
    failure = llm.RealtimeError("audio drain failed")
    function_stream_closed = asyncio.Event()
    functions = utils.aio.Chan[llm.FunctionCall]()

    async def function_stream():
        try:
            async for item in functions:
                yield item
        finally:
            function_stream_closed.set()

    async def broken_audio():
        yield rtc.AudioFrame(
            data=b"\0\0" * 240, sample_rate=24000, num_channels=1, samples_per_channel=240
        )
        raise failure

    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))
        speeches = []
        session.on("speech_created", speeches.append)
        if requested:
            speech = session.generate_reply()
            while not model.active_session._reply_futs:
                await asyncio.sleep(0)
        messages = utils.aio.Chan[llm.MessageGeneration]()
        text = utils.aio.Chan[str]()
        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(["text", "audio"])
        messages.send_nowait(
            llm.MessageGeneration(
                message_id="message",
                text_stream=text,
                audio_stream=broken_audio(),
                modalities=modalities,
            )
        )
        messages.close()
        if not open_functions:
            functions.close()
        text.send_nowait("Reply")
        if not open_text:
            text.close()
        event = GenerationCreatedEvent(
            response_id="response",
            message_stream=messages,
            function_stream=function_stream(),
            user_initiated=requested,
        )
        if requested:
            model.active_session._reply_futs[-1].set_result(event)
        else:
            model.active_session.emit("generation_created", event)
            speech = speeches[-1].speech_handle
        try:
            await asyncio.wait_for(speech.wait_for_playout(), 1)
            assert speech.exception() is failure
            await asyncio.wait_for(function_stream_closed.wait(), 1)
            assert session.agent_state == "listening"
        finally:
            functions.close()
            text.close()
