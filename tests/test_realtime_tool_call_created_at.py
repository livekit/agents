import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, RunContext, function_tool, llm, utils

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

        speech_handle = session.generate_reply()
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
        await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=5)
        history = session.history

    assert fnc_call.created_at >= authorized_at
    # the call is recorded when it starts and again with its output, so it must not double up
    assert [item.call_id for item in history.items if item.type == "function_call"] == ["call_1"]


async def test_realtime_records_the_call_of_a_tool_that_never_ran() -> None:
    """A call rejected before execution still reaches the chat context with its output.

    The execution-started callback records the call, and a rejected call never reaches it, so
    the realtime path records the call next to its output as the pipeline task does.
    """

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async with AgentSession(llm=model) as session:
        await session.start(ToolAgent())

        speech_handle = session.generate_reply()
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

        # a non-object arguments payload that json_repair cannot turn into one
        function_ch.send_nowait(
            llm.FunctionCall(call_id="call_1", name="lookup", arguments="[1, 2, 3]")
        )
        function_ch.close()

        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
            )
        )

        await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=5)
        chat_ctx = session.current_agent.chat_ctx
        history = session.history

    for ctx in (chat_ctx, history):
        calls = [item for item in ctx.items if item.type == "function_call"]
        outputs = [item for item in ctx.items if item.type == "function_call_output"]
        assert [item.call_id for item in calls] == ["call_1"]
        assert [item.call_id for item in outputs] == ["call_1"]
        assert outputs[0].is_error

    # the pair survives the filter the agent applies to its own history
    copied = chat_ctx.copy(tools=session.current_agent.tools)
    assert [item.type for item in copied.items if item.type.startswith("function_call")] == [
        "function_call",
        "function_call_output",
    ]


async def test_a_realtime_tool_update_leaves_its_recorded_call_as_it_was() -> None:
    """The call is recorded when the tool starts, so ``ctx.update()`` must not edit it."""
    released = asyncio.Event()
    waited = asyncio.Event()
    recorded: list[llm.FunctionCall] = []

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self, ctx: RunContext) -> str:
            recorded.extend(i for i in ctx.session.history.items if i.type == "function_call")
            extra = [dict(call.extra) for call in recorded]
            with pytest.raises(RuntimeError, match="circular wait"):
                await ctx.speech_handle.wait_for_playout()
            await ctx.update("looking it up")
            assert [dict(call.extra) for call in recorded] == extra
            released.set()
            # once released, the tool no longer holds its speech, so it may wait for it
            await ctx.speech_handle.wait_for_playout()
            waited.set()
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async with AgentSession(llm=model) as session:
        await session.start(ToolAgent())

        speech_handle = session.generate_reply()
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
                message_stream=message_ch, function_stream=function_ch, user_initiated=True
            )
        )

        await asyncio.wait_for(released.wait(), timeout=5)
        await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=5)
        await asyncio.wait_for(waited.wait(), timeout=5)

    assert [call.call_id for call in recorded] == ["call_1"]
