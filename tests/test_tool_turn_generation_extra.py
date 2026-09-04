"""A generation's provider extra (the inference gateway's deployment stamp) is kept
on the turn even when the LLM produced only tool calls and no text."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

STAMP = {"livekit": {"inference_deployment": "standard_openai"}}


class ToolAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool
    async def do_the_thing(self) -> str:
        """Do the thing."""
        return "ok"


async def test_tool_only_turn_keeps_the_generation_extra() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Do the thing.")
    actions.add_llm(
        content="",
        tool_calls=[FunctionToolCall(name="do_the_thing", arguments="{}", call_id="1")],
        extra=STAMP,
    )
    actions.add_llm(content="Done.", input="ok")
    actions.add_tts(1.0)

    session: AgentSession = create_session(actions)
    agent = ToolAgent()
    await asyncio.wait_for(run_session(session, agent), timeout=60.0)

    calls = [i for i in agent.chat_ctx.items if i.type == "function_call"]
    assert len(calls) == 1
    assert calls[0].extra["livekit"] == STAMP["livekit"]
