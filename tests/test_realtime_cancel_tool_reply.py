"""`cancel_tool_reply()` must reach the realtime session, not just the pipeline.

The result is still delivered, since the call stays open until it is; the request to stay
quiet travels with it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm, utils

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = pytest.mark.unit


async def test_cancelled_tool_reply_marks_the_synced_output() -> None:
    executed = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            executed.set()
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async with AgentSession(llm=model) as session:
        session.on("function_tools_executed", lambda ev: ev.cancel_tool_reply())
        await session.start(ToolAgent())

        session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("Let me look that up.")
        text_ch.close()
        audio_ch.close()
        function_ch.send_nowait(llm.FunctionCall(call_id="call_1", name="lookup", arguments="{}"))
        function_ch.close()

        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
            )
        )

        await asyncio.wait_for(executed.wait(), timeout=5)

        async def _synced_output() -> llm.FunctionCallOutput:
            while True:
                for item in model.active_session.chat_ctx.items:
                    if item.type == "function_call_output":
                        return item
                await asyncio.sleep(0)

        output = await asyncio.wait_for(_synced_output(), timeout=5)

    assert output.call_id == "call_1"
    assert not output.reply_required
