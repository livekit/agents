"""A run's tools, the agent's and the session's together, have unique names (tools.md)."""

import pytest

from livekit.agents import Agent, RunContext, VoiceSession
from livekit.agents.llm import ToolContext, function_tool

from .test_voice_run_context import _calls

pytestmark = pytest.mark.unit

REFUSED = "two tools are named `check`: a run's tools, the agent's and the session's together"


@function_tool(name="check")
async def session_check(ctx: RunContext) -> str:
    """The session's check."""
    return "session"


def _agent() -> Agent:
    agent = Agent(instructions="Check.", name="desk")

    @agent.tool
    async def check(ctx: RunContext) -> str:
        """The agent's check."""
        return "agent"

    return agent


async def test_a_session_tool_named_as_an_agents_tool_refuses_the_call() -> None:
    async with VoiceSession(
        llm=_calls("hello"),
        vad=None,
        turn_handling={"turn_detection": "manual"},
        tools=[session_check],
    ) as session:
        with pytest.raises(ValueError, match=REFUSED):
            await session.start(_agent())


def test_the_same_tool_listed_twice_is_one_tool() -> None:
    assert list(ToolContext([session_check, session_check]).function_tools) == ["check"]
