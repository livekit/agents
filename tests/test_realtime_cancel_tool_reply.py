"""`cancel_tool_reply()` must reach the realtime session, not just the pipeline.

The result is still delivered, since the call stays open until it is; the request to stay
quiet travels with it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, ToolResult, function_tool

from .fake_realtime import run_realtime_tool_turn

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.mark.parametrize("wrapped", [True, False])
async def test_cancelled_tool_reply_marks_the_synced_output(wrapped: bool) -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def lookup(self) -> str | ToolResult:
            """Look something up."""
            self.tool_executed.set()
            return ToolResult("ok", reply_required=True) if wrapped else "ok"

    agent = ToolAgent()
    _, model = await run_realtime_tool_turn(
        agent,
        tool_executed=agent.tool_executed,
        on_session=lambda s: s.on("function_tools_executed", lambda ev: ev.cancel_tool_reply()),
    )

    synced = [i for i in model.active_session.chat_ctx.items if i.type == "function_call_output"]
    assert len(synced) == 1
    assert synced[0].call_id == "1"
    assert not synced[0].reply_required


@pytest.mark.parametrize("reply_required", [True, False])
@pytest.mark.parametrize("interrupt", [True, False])
async def test_tool_result_reply_required_reaches_realtime_session(
    reply_required: bool, interrupt: bool
) -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def send_dtmf(self) -> ToolResult:
            """Send DTMF events."""
            self.tool_executed.set()
            return ToolResult("Successfully sent DTMF events: 1", reply_required=reply_required)

    agent = ToolAgent()
    session, model = await run_realtime_tool_turn(
        agent, tool_executed=agent.tool_executed, interrupt=interrupt
    )

    for items in (agent.chat_ctx.items, session.history.items, model.active_session.chat_ctx.items):
        outputs = [item for item in items if item.type == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0].call_id == "1"
        assert outputs[0].output == "Successfully sent DTMF events: 1"
        assert not outputs[0].is_error
        assert outputs[0].reply_required is (reply_required and not interrupt)
    assert model.active_session.generate_reply_calls == 1
