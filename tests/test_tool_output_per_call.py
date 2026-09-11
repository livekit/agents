"""Every tool call is answered, whatever the tool did.

An LLM re-issues a call it never got back, running its side effects again, and a realtime
model waits on it. A tool with nothing to say says so with `reply_required` instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, function_tool
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


class ReceivingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are the receiving agent.")
        self.entered = asyncio.Event()

    async def on_enter(self) -> None:
        self.entered.set()
        await self.session.generate_reply(instructions="receiver_greeting")


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


async def _run(behavior: Callable[[], Any]) -> tuple[Agent, AgentSession, FunctionCallOutput]:
    """Run one turn whose single tool call is answered, and return that answer."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Do the thing.")
    actions.add_llm(
        content="Working on it.",
        tool_calls=[FunctionToolCall(name="do_the_thing", arguments="{}", call_id="1")],
    )
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
    "tool_names",
    [("lookup", "transfer"), ("transfer", "lookup")],
    ids=["handoff_last", "handoff_first"],
)
async def test_parallel_tool_reply_does_not_race_agent_handoff(
    tool_names: tuple[str, str],
) -> None:
    """A sibling tool result must not make the old agent reply after a handoff."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Transfer and look up the account.")
    actions.add_llm(
        content="",
        tool_calls=[
            FunctionToolCall(name=name, arguments="{}", call_id=f"{name}-1") for name in tool_names
        ],
    )
    actions.add_llm(content="old agent reply", input="lookup complete")
    actions.add_tts(0.5, input="old agent reply")
    actions.add_llm(content="receiver reply", input="receiver_greeting")
    actions.add_tts(0.5, input="receiver reply")

    session = create_session(actions)
    receiver = ReceivingAgent()
    routing_agent = ParallelHandoffAgent(receiver)
    await asyncio.wait_for(run_session(session, routing_agent), timeout=SESSION_TIMEOUT)

    assert receiver.entered.is_set(), "the receiving agent must run on_enter"
    assistant_text = [
        item.text_content
        for item in session.history.items
        if item.type == "message" and item.role == "assistant"
    ]
    assert "receiver reply" in assistant_text
    assert "old agent reply" not in assistant_text

    for label, items in (
        ("routing agent chat_ctx", routing_agent.chat_ctx.items),
        ("session history", session.history.items),
    ):
        calls = [item for item in items if item.type == "function_call"]
        outputs = [item for item in items if item.type == "function_call_output"]
        assert {call.call_id for call in calls} == {"transfer-1", "lookup-1"}, label
        assert {output.call_id for output in outputs} == {"transfer-1", "lookup-1"}, label
