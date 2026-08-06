import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm, utils

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = pytest.mark.unit


async def test_realtime_tool_call_created_at_is_stamped_at_execution_start() -> None:
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

        # a call the model emitted well before this generation was authorized to speak
        fnc_call = llm.FunctionCall(
            call_id="call_1", name="lookup", arguments="{}", created_at=time.time() - 60
        )
        function_ch.send_nowait(fnc_call)
        function_ch.close()

        authorized_at = time.time()
        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
            )
        )

        await asyncio.wait_for(executed.wait(), timeout=5)

    assert fnc_call.created_at >= authorized_at
