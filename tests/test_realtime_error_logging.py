from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, ToolResult, function_tool, llm

from .fake_realtime import FakeRealtimeModel, run_realtime_tool_turn

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_PRIVATE_ERROR = "private provider response with credentials"


@pytest.mark.parametrize("context_error", [None, llm.RealtimeError, RuntimeError])
async def test_realtime_reply_failure_logs_only_error_type(
    caplog: pytest.LogCaptureFixture, context_error: type[Exception] | None
) -> None:
    model = FakeRealtimeModel()
    error = (context_error or llm.RealtimeError)(_PRIVATE_ERROR)
    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="Test"))
        rt = model.active_session
        if context_error is not None:
            rt.update_error = error
        handle = session.generate_reply(user_input="hello")

        async def wait_for_reply() -> None:
            while not handle.done() and not rt._reply_futs:
                await asyncio.sleep(0)
            if rt._reply_futs:
                rt._reply_futs[-1].set_exception(error)
            await handle

        await asyncio.wait_for(wait_for_reply(), 2)
        assert handle.exception() is error

    records = [record for record in caplog.records if "failed to" in record.getMessage()]
    assert len(records) == (2 if context_error is llm.RealtimeError else 1)
    for record in records:
        assert record.error_type == type(error).__name__
        assert record.exc_info is None
        assert _PRIVATE_ERROR not in str(record.__dict__)


@pytest.mark.parametrize("interrupted", [False, True])
async def test_realtime_tool_sync_failure_logs_only_error_type(
    caplog: pytest.LogCaptureFixture, interrupted: bool
) -> None:
    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="Test")
            self.tool_executed = asyncio.Event()

        @function_tool
        async def lookup(self) -> ToolResult:
            """Look something up."""
            self.session._activity._rt_session.update_error = llm.RealtimeError(_PRIVATE_ERROR)
            self.tool_executed.set()
            return ToolResult("ok", reply_required=False)

    agent = ToolAgent()
    await run_realtime_tool_turn(agent, tool_executed=agent.tool_executed, interrupt=interrupted)
    records = [record for record in caplog.records if "failed to" in record.getMessage()]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "failed to sync the tool results of an interrupted generation"
        if interrupted
        else "failed to update chat context before generating the function calls results"
    )
    assert records[0].error_type == "RealtimeError"
    assert records[0].exc_info is None
    assert _PRIVATE_ERROR not in str(records[0].__dict__)
