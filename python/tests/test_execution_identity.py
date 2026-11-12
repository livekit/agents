"""Execution identity survives reconstruction and follows its A2A task."""

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.llm import ChatContext
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .fake_llm import FakeLLM
from .test_agent_run import call, respond

pytestmark = pytest.mark.unit


async def test_direct_execution_identity_survives_serialized_resume() -> None:
    identities: list[str] = []
    agent = Agent(
        instructions="Book after collecting a date.",
        llm=FakeLLM(
            fake_responses=[
                respond("book", "", call("book")),
                respond("monday", "", call("book", call_id="c2")),
                respond("reserved", "Booked."),
                respond("another booking", "", call("book", call_id="c3")),
            ]
        ),
    )

    @agent.tool
    def book(ctx: RunContext) -> str | InputRequired:
        """Book once a date is supplied."""
        identities.append(ctx.execution_id)
        if len(identities) != 2:
            return InputRequired("Which date?", verbatim=True)
        return "reserved"

    first = await agent.run("book")
    assert first.status == "input_required"
    saved = ChatContext.from_dict(ChatContext(items=first.all_messages()).to_dict())
    second = await agent.run("monday", message_history=saved)
    assert second.status == "completed" and second.output == "Booked."
    assert identities[0] and identities[1] == identities[0]
    await agent.run("another booking", message_history=second.all_messages())
    assert identities[2] != identities[0]


async def test_direct_execution_identity_uses_the_a2a_task_id() -> None:
    seen: list[tuple[str, str | None]] = []
    agent = Agent(instructions="Help.", llm=FakeLLM(fake_responses=[respond("hi", "Hello.")]))

    @agent.instructions
    def identity(ctx: RunContext) -> str:
        seen.append((ctx.execution_id, ctx.task_id))
        return "Continue."

    token = set_run_hooks(RunHooks(task_id="task-from-transport"))
    try:
        result = await agent.run("hi")
    finally:
        reset_run_hooks(token)
    assert result.status == "completed"
    assert seen == [("task-from-transport", "task-from-transport")]
