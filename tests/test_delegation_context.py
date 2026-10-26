"""What an expert of this project can read of the caller (Flow B): the delegation names the
caller's lane, and the person's own words are on it before the expert runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate

from .a2a_driver import (
    AGENT,
    Door,
    Reply,
    ScriptedLLM,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
)

pytestmark = [pytest.mark.unit]

PERSON = "Book me a cleaning on Monday, I am Ana Ruiz."
RESTATED = "book a dental cleaning on Monday"


def front_desk(delegate: A2ADelegate) -> tuple[Agent, ScriptedLLM]:
    model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "c1", request=RESTATED)]),
        Reply("Done.", when=output_of("c1")),
    )
    return Agent(instructions="front desk", tools=[delegate], llm=model), model


def tool_output(model: ScriptedLLM) -> str:
    (out,) = [i for i in model.requests[-1].items if i.type == "function_call_output"]
    return str(out.output)


@pytest.mark.rule("R18")
async def test_a_delegation_names_the_callers_lane_and_injects_nothing(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    """The expert's handler reads which lane delegated; its own model reads the request alone."""
    expert_model = ScriptedLLM(Reply("Booked for Monday."))
    expert_agent = Agent(instructions="scheduling", llm=expert_model)

    async def scheduling(ctx: A2ASessionContext[Any]) -> str:
        result = await expert_agent.run(ctx.message, message_history=ctx.message_history)
        return f"{result.output} (asked by {ctx.caller_lane})"

    front, model = front_desk(expert(scheduling))

    async def desk(ctx: A2ASessionContext[Any]) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    sent = await door(desk).send(PERSON)
    assert sent.state == "completed"
    assert tool_output(model) == f"Booked for Monday. (asked by {AGENT})"
    (request,) = expert_model.requests
    said = [i.text_content for i in request.items if i.type == "message" and i.role == "user"]
    assert said == [RESTATED]


@pytest.mark.rule("R17")
@pytest.mark.xfail(
    strict=True,
    reason="R17: Rust ask `Boundary.record` (ThreadDb.write_items under the run's fence): the "
    "caller's message reaches its lane only at the commit",
)
async def test_the_expert_reads_the_persons_words_on_the_callers_lane(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    """The front desk's lane holds the person's message before the delegation goes out."""

    async def scheduling(ctx: A2ASessionContext[Any]) -> str:
        assert ctx.caller_lane is not None
        history = await ctx.thread.chat_history(ctx.caller_lane)
        said = [i.text_content for i in history.items if i.type == "message" and i.role == "user"]
        return f"the person said: {said[-1] if said else 'nothing'}"

    front, model = front_desk(expert(scheduling))

    async def desk(ctx: A2ASessionContext[Any]) -> str | None:
        return (await front.run(ctx.message, message_history=ctx.message_history)).output

    sent = await door(desk).send(PERSON)
    assert sent.state == "completed"
    assert tool_output(model) == f"the person said: {PERSON}"
