"""A child agent run as a tool reads and writes its parent's application context."""

from dataclasses import dataclass

import pytest

from livekit.agents import Agent, RunContext, TextSession

from .test_text_session import Reply, Script, call

pytestmark = pytest.mark.unit


@dataclass
class Booking:
    nights: int = 0


async def test_a_child_shares_the_parents_context() -> None:
    context = Booking()
    seen: list[Booking] = []

    child = Agent(
        name="child",
        instructions="Record the nights.",
        output_type=str,
        llm=Script(call("record", {"nights": 3}, "record-1"), Reply("Recorded.")),
    )

    @child.tool
    async def record(ctx: RunContext[Booking], nights: int) -> str:
        """Record the nights.

        Args:
            nights: how many nights
        """
        seen.append(ctx.context)
        ctx.context.nights = nights
        return "ok"

    parent = Agent(
        name="parent",
        instructions="Delegate.",
        tools=[child.as_tool(tool_name="child")],
        llm=Script(call("child", {}, "delegate"), Reply("Booked for three nights.")),
    )

    turn = await TextSession(agent=parent).run("Book three nights", context=context)
    assert turn.result.status == "completed"
    assert seen == [context] and seen[0] is context
    assert context.nights == 3
