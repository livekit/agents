"""A tool still running when an interrupted speech times out is released like an async tool.

Its call answers with the running placeholder at the timeout, so a realtime model that holds
the call open, such as GPT-Live, is not blocked, and its result arrives later as a deferred
reply instead of being discarded.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm, utils
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, _audio_frame, fake_capabilities
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

TOOL_DURATION = INTERRUPTION_TIMEOUT + 3


def _outputs(chat_ctx: llm.ChatContext, call_id: str) -> list[llm.FunctionCallOutput]:
    return [
        item
        for item in chat_ctx.items
        if item.type == "function_call_output" and item.call_id == call_id
    ]


async def test_interrupt_timeout_releases_the_running_tool() -> None:
    tool_started = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            tool_started.set()
            await asyncio.sleep(TOOL_DURATION)
            return "the fee is 10 dollars"

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

        # the timeout answers the open call with the running placeholder
        assert handle.done()
        outputs = _outputs(rt_session.chat_ctx, "1")
        assert len(outputs) == 1
        assert not outputs[0].is_error
        assert "still in progress" in outputs[0].output

        # the tool's real result arrives later, as an async tool's final result does
        await asyncio.sleep(TOOL_DURATION)
        final = _outputs(rt_session.chat_ctx, "1_final")
        assert len(final) == 1
        assert "the fee is 10 dollars" in final[0].output


async def test_interrupt_timeout_releases_the_running_pipeline_tool() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Look up my order")
    actions.add_llm(
        content="Let me check.",
        tool_calls=[FunctionToolCall(name="lookup_order", arguments="{}", call_id="1")],
    )
    actions.add_tts(4.0)

    class SlowToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup_order(self) -> str:
            """Look up the order."""
            await asyncio.sleep(TOOL_DURATION)
            return "order 42 shipped"

    session = create_session(actions)
    agent = SlowToolAgent()

    def _interrupt_while_speaking(ev: object) -> None:
        # only the first reply, so the deferred reply that carries the result plays out
        if getattr(ev, "new_state", None) == "speaking":
            session.off("agent_state_changed", _interrupt_while_speaking)
            asyncio.get_running_loop().call_later(0.5, lambda: session.interrupt(force=True))

    session.on("agent_state_changed", _interrupt_while_speaking)
    await asyncio.wait_for(run_session(session, agent, drain_delay=TOOL_DURATION + 5), timeout=60)

    # the running placeholder answers the call in history, so the model does not issue it again
    outputs = _outputs(agent.chat_ctx, "1")
    assert len(outputs) == 1
    assert "still in progress" in outputs[0].output

    # and the tool's real result is kept instead of discarded
    final = _outputs(agent.chat_ctx, "1_final")
    assert len(final) == 1
    assert "order 42 shipped" in final[0].output
