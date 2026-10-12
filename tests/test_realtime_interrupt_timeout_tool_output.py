"""A tool still running when an interrupted speech times out must still answer its call.

A realtime model holds the call open until it gets an output, so a call left unanswered
blocks every later response on models such as GPT-Live.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm, utils
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, _audio_frame, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def test_interrupt_timeout_answers_the_running_call() -> None:
    tool_started = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            tool_started.set()
            await asyncio.sleep(INTERRUPTION_TIMEOUT + 3)
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        await session.start(ToolAgent())
        rt_session = model.active_session

        handle = session.generate_reply()
        while not rt_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("let me check")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()
        function_ch.send_nowait(llm.FunctionCall(call_id="1", name="lookup", arguments="{}"))
        function_ch.close()

        rt_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        await asyncio.wait_for(tool_started.wait(), timeout=5)
        handle.interrupt(force=True)
        await asyncio.sleep(INTERRUPTION_TIMEOUT + 0.5)

        assert handle.done()
        outputs = [
            item
            for item in rt_session.chat_ctx.items
            if item.type == "function_call_output" and item.call_id == "1"
        ]
        assert len(outputs) == 1
        assert outputs[0].is_error
        assert "result is unknown" in outputs[0].output
