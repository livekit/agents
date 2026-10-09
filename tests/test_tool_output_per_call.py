"""Every tool call is answered, whatever the tool did.

An LLM re-issues a call it never got back, running its side effects again, and a realtime
model waits on it. A tool with nothing to say says so with `reply_required` instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, ToolResult, function_tool
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
) -> tuple[Agent, AgentSession, FunctionCallOutput]:
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


@pytest.mark.parametrize(
    "behavior, error_text",
    [
        pytest.param(lambda: object(), "invalid output", id="invalid_output"),
        pytest.param(
            lambda: [Agent(instructions="first"), Agent(instructions="second")],
            "more than one agent",
            id="two_agents",
        ),
    ],
)
async def test_a_tool_that_produced_nothing_usable_answers_as_an_error(
    behavior: Callable[[], Any], error_text: str
) -> None:
    """The model can recover from a failure; it cannot recover from silence."""
    agent, session, output = await _run(behavior)

    assert session.current_agent is agent, "an ambiguous handoff must not be applied"
    assert output.is_error
    assert error_text in output.output


async def test_stop_response_answers_the_call_and_asks_for_no_reply() -> None:
    """StopResponse means "say nothing", not "leave the call open"."""
    _, _, output = await _run(_raise_stop_response)

    assert output.output == ""
    assert not output.is_error
    assert not output.reply_required


async def test_bare_handoff_answers_the_call_and_asks_for_no_reply() -> None:
    """A handoff with nothing to say is answered, and the new agent speaks instead."""
    agent, session, output = await _run(lambda: Agent(instructions="You are the next agent."))

    assert session.current_agent is not agent, "the handoff must be applied"
    assert output.output == ""
    assert not output.reply_required


@pytest.mark.parametrize(
    "result, expected_output, reply_required",
    [
        (
            ToolResult("Successfully sent DTMF events: 1", reply_required=False),
            "Successfully sent DTMF events: 1",
            False,
        ),
        (ToolResult("ok", reply_required=True), "ok", True),
        (ToolResult("ok"), "ok", True),
        (ToolResult(None, reply_required=True), "", True),
        (ToolResult(None, reply_required=False), "", False),
        ("ok", "ok", True),
        (None, "", False),
    ],
)
async def test_tool_result_controls_follow_up_reply(
    result: Any, expected_output: str, reply_required: bool
) -> None:
    agent, session, output = await _run(lambda: result, reply_to=expected_output)

    assert not output.is_error
    for items in (agent.chat_ctx.items, session.history.items):
        outputs = [item for item in items if item.type == "function_call_output"]
        assert outputs[0].output == expected_output
        assert outputs[0].reply_required is reply_required
        replies = [
            item.text_content
            for item in items
            if item.type == "message" and item.role == "assistant"
        ]
        assert replies == (["Working on it.", "Done."] if reply_required else ["Working on it."])


async def test_handoff_with_tool_result_keeps_output_without_requesting_reply() -> None:
    next_agent = Agent(instructions="You are the next agent.")
    _, session, output = await _run(
        lambda: (next_agent, ToolResult("Transferred", reply_required=False)),
        reply_to="Transferred",
    )

    assert session.current_agent is next_agent
    assert output.output == "Transferred"
    assert not output.is_error
    assert not output.reply_required
    assert not any(
        item.type == "message" and item.text_content == "Done." for item in session.history.items
    )


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
