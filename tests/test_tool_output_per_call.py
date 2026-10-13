"""Every tool call is answered, whatever the tool did.

An LLM re-issues a call it never got back, running its side effects again, and a realtime
model waits on it. A tool with nothing to say says so with `reply_required` instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, LiveSession, ToolResult, function_tool
from livekit.agents.llm import FunctionCallOutput, FunctionToolCall, StopResponse

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0


class ToolAgent(Agent):
    """Runs whatever the test hands it, so each case differs only by that callable."""

    def __init__(self, behavior: Callable[[], Any]) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self._behavior = behavior

    @function_tool
    async def do_the_thing(self) -> Any:
        """Do the thing."""
        return self._behavior()


async def _run(
    behavior: Callable[[], Any], *, reply_to: str | None = None
) -> tuple[Agent, LiveSession, FunctionCallOutput]:
    """Run one turn whose single tool call is answered, and return that answer."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Do the thing.")
    actions.add_llm(
        content="Working on it.",
        tool_calls=[FunctionToolCall(name="do_the_thing", arguments="{}", call_id="1")],
    )
    actions.add_tts(1.0)
    if reply_to is not None:
        actions.add_llm(input=reply_to, content="Done.")
        actions.add_tts(1.0)

    session = create_session(actions)
    agent = ToolAgent(behavior)
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    for label, items in (
        ("agent chat_ctx", agent.chat_ctx.items),
        ("session history", session.history.items),
    ):
        calls = [i for i in items if i.type == "function_call"]
        outs = [i for i in items if i.type == "function_call_output"]
        assert len(calls) == 1, f"{label}: the call must be recorded once"
        assert len(outs) == 1, f"{label}: the call must be answered exactly once"
        assert outs[0].call_id == calls[0].call_id

    return agent, session, [i for i in agent.chat_ctx.items if i.type == "function_call_output"][0]


def _raise_stop_response() -> Any:
    raise StopResponse


async def test_tool_result_does_not_suppress_other_tools_replies() -> None:
    class ParallelToolAgent(Agent):
        @function_tool
        async def send_dtmf(self) -> ToolResult:
            """Send DTMF events."""
            return ToolResult("Sent", reply_required=False)

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            return "Found"

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Send DTMF and look it up.")
    actions.add_llm(
        content="Working on it.",
        tool_calls=[
            FunctionToolCall(name="send_dtmf", arguments="{}", call_id="1"),
            FunctionToolCall(name="lookup", arguments="{}", call_id="2"),
        ],
    )
    actions.add_tts(1.0)
    for output in ("Sent", "Found"):
        actions.add_llm(input=output, content="Done.")
        actions.add_tts(1.0)

    session = create_session(actions)
    agent = ParallelToolAgent(instructions="test")
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)

    outputs = {
        item.call_id: item for item in session.history.items if item.type == "function_call_output"
    }
    assert set(outputs) == {"1", "2"}
    assert outputs["1"].output == "Sent"
    assert not outputs["1"].reply_required
    assert outputs["2"].output == "Found"
    assert outputs["2"].reply_required
    replies = [
        item.text_content
        for item in session.history.items
        if item.type == "message" and item.role == "assistant"
    ]
    assert replies == ["Working on it.", "Done."]
