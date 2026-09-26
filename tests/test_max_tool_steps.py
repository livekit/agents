"""`max_tool_steps` bounds the consecutive tool rounds of one LLM turn.

A model that re-issues its tool after every result must be cut off with
tool_choice="none" once it has spent its budget, so a turn executes at most
`max_tool_steps` tool rounds (agent_session.py documents "Maximum consecutive
tool calls per LLM turn").
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0

# scripted re-issues of the tool, enough for every max_tool_steps under test
TOOL_ROUNDS = 7


class LoopingToolAgent(Agent):
    """Answers every call with a fresh result, so the model always wants another round."""

    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")
        self.executions: list[str] = []

    @function_tool
    async def do_the_thing(self) -> str:
        """Do the thing."""
        self.executions.append(f"result {len(self.executions) + 1}")
        return self.executions[-1]


def _tool_call(call_id: str) -> FunctionToolCall:
    return FunctionToolCall(name="do_the_thing", arguments="{}", call_id=call_id)


async def _run(max_tool_steps: int) -> int:
    """Run one turn against a model that re-issues the tool after every result."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Do the thing.")
    actions.add_llm(
        content="Working on it.",
        tool_calls=[_tool_call("call_0")],
    )
    actions.add_tts(0.4)
    for i in range(1, TOOL_ROUNDS + 1):
        actions.add_llm(
            content=f"Round {i + 1}.",
            tool_calls=[_tool_call(f"call_{i}")],
            input=f"result {i}",
        )
        actions.add_tts(0.4, input=f"Round {i + 1}.")
    actions.add_llm(content="All done.", input=f"result {TOOL_ROUNDS + 1}")
    actions.add_tts(0.4, input="All done.")

    session = create_session(actions, extra_kwargs={"max_tool_steps": max_tool_steps})
    agent = LoopingToolAgent()
    await asyncio.wait_for(run_session(session, agent), timeout=SESSION_TIMEOUT)
    return len(agent.executions)


@pytest.mark.parametrize("max_tool_steps", [1, 2, 3])
async def test_max_tool_steps_caps_consecutive_tool_rounds(max_tool_steps: int) -> None:
    """The cap applies after max_tool_steps rounds, not one round later."""
    executions = await _run(max_tool_steps)

    assert executions == max_tool_steps
