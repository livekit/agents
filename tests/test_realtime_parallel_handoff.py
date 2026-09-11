from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool, utils
from livekit.agents.llm import ChatContext, FunctionCall, GenerationCreatedEvent, MessageGeneration

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class ReceivingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are the receiving agent.")
        self.entered = asyncio.Event()

    async def on_enter(self) -> None:
        self.entered.set()


class ParallelHandoffAgent(Agent):
    def __init__(self, receiver: ReceivingAgent) -> None:
        super().__init__(instructions="You are the routing agent.")
        self._receiver = receiver

    @function_tool
    async def transfer(self) -> Agent:
        """Transfer the conversation to the receiving agent."""
        return self._receiver

    @function_tool
    async def lookup(self) -> str:
        """Look up the account before transfer."""
        return "lookup complete"


def _tool_generation(tool_names: tuple[str, str]) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    message_ch.close()
    for name in tool_names:
        function_ch.send_nowait(FunctionCall(call_id=f"{name}-1", name=name, arguments="{}"))
    function_ch.close()

    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=True,
        response_id="parallel-handoff",
    )


@pytest.mark.parametrize(
    "tool_names",
    [("lookup", "transfer"), ("transfer", "lookup")],
    ids=["handoff_last", "handoff_first"],
)
@pytest.mark.parametrize("auto_tool_reply_generation", [True, False], ids=["auto", "manual"])
async def test_realtime_parallel_tool_reply_does_not_race_agent_handoff(
    tool_names: tuple[str, str],
    auto_tool_reply_generation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            auto_tool_reply_generation=auto_tool_reply_generation,
            audio_output=False,
        )
    )
    receiver = ReceivingAgent()
    routing_agent = ParallelHandoffAgent(receiver)
    tools_executed = asyncio.Event()

    async with AgentSession(llm=model) as session:
        session.on("function_tools_executed", lambda _: tools_executed.set())
        await session.start(routing_agent)

        outgoing_rt_session = model.active_session
        chat_ctx_updates: list[ChatContext] = []
        update_chat_ctx = outgoing_rt_session.update_chat_ctx

        async def _record_chat_ctx_update(chat_ctx: ChatContext) -> None:
            chat_ctx_updates.append(chat_ctx.copy())
            await update_chat_ctx(chat_ctx)

        monkeypatch.setattr(outgoing_rt_session, "update_chat_ctx", _record_chat_ctx_update)
        speech_handle = session.generate_reply()

        async def _wait_for_reply_request() -> None:
            while not outgoing_rt_session._reply_futs:
                await asyncio.sleep(0)

        await asyncio.wait_for(_wait_for_reply_request(), timeout=5.0)
        outgoing_rt_session._reply_futs[0].set_result(_tool_generation(tool_names))

        await asyncio.wait_for(tools_executed.wait(), timeout=5.0)
        await asyncio.wait_for(receiver.entered.wait(), timeout=5.0)
        await asyncio.wait_for(speech_handle.wait_for_playout(), timeout=5.0)

        assert session.current_agent is receiver
        assert outgoing_rt_session.generate_reply_calls == 1
        assert not any(
            item.type == "function_call_output"
            for chat_ctx in chat_ctx_updates
            for item in chat_ctx.items
        )

        for label, items in (
            ("routing agent chat_ctx", routing_agent.chat_ctx.items),
            ("session history", session.history.items),
        ):
            calls = [item for item in items if item.type == "function_call"]
            outputs = [item for item in items if item.type == "function_call_output"]
            assert {call.call_id for call in calls} == {"transfer-1", "lookup-1"}, label
            assert {output.call_id for output in outputs} == {"transfer-1", "lookup-1"}, label
