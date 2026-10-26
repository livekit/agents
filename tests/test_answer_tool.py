"""The answer to a tool's question in text (docs/human-in-the-loop.md): the next message goes to
the task's model, which acts on it; a question names no tool to take it and has no timeout."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext, ChatMessage

from .a2a_driver import (
    Door,
    Reply,
    ScriptedLLM,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
)

pytestmark = [pytest.mark.unit]


def said(text: str) -> Callable[[ChatContext], bool]:
    """A request whose last item is the person's message ``text``."""

    def matches(ctx: ChatContext) -> bool:
        last = ctx.items[-1] if ctx.items else None
        return isinstance(last, ChatMessage) and last.role == "user" and last.text_content == text

    return matches


@pytest.mark.rule("R44", "R41", "B8", "B28")
def test_a_question_names_no_tool_and_no_timeout() -> None:
    from datetime import timedelta

    with pytest.raises(TypeError):
        InputRequired("Name and email?", tool=None)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        InputRequired("Name and email?", timeout=timedelta(seconds=5))  # type: ignore[call-arg]


@pytest.mark.rule("R44", "B28")
@pytest.mark.xfail(
    strict=True,
    reason="R44: Rust ask, continuation by taskId: the routed answer leaves a working row and "
    "the question open",
)
async def test_the_answer_goes_to_the_tasks_model(door: Callable[..., Door]) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1", request="")], when=said("book me in")),
        Reply(
            tool_calls=[call("book", "b2", request="Jane <j.o@example.co.uk>")],
            when=said("Jane <j.o@example.co.uk>"),
        ),
        Reply("All set.", when=output_of("b2")),
    )
    agent = Agent(instructions="front desk", llm=model)

    @agent.tool
    async def book(ctx: RunContext, request: str) -> str | InputRequired:
        """Book a visit.

        Args:
            request: the person's name and email, once they gave them
        """
        if not request:
            return InputRequired("Name and email?")
        return f"booked for {request}"

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler)
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)
    assert asked.state == "input_required"

    done = await desk.send("Jane <j.o@example.co.uk>", context_id=thread)
    assert done.state == "completed" and done.answer == "All set."
    # the model read the answer as the person's message, and called the tool with it
    answered = model.requests[2]
    last_user = [i for i in answered.items if isinstance(i, ChatMessage) and i.role == "user"][-1]
    assert last_user.text_content == "Jane <j.o@example.co.uk>"
