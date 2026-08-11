"""`cancel_tool_reply()` must reach the realtime session, not just the pipeline.

The result is still delivered, since the call stays open until it is; the request to stay
quiet travels with it.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, function_tool

from .fake_realtime import run_realtime_tool_turn

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def test_cancelled_tool_reply_marks_the_synced_output() -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            self.tool_executed.set()
            return "ok"

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
