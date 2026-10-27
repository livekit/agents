"""The answer to a tool's question in text (docs/human-in-the-loop.md): the next message goes to
the task's model, which acts on it; a question names no tool to take it and has no timeout."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext, ChatMessage, FunctionToolCall
from livekit.agents.storage import AgentDBStorage

from .a2a_driver import (
    Door,
    Reply,
    ScriptedLLM,
    call,
    http_client,
    make_server,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    payloads,
    send,
    stock_client,
    user_message,
)
from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit]


def said(text: str) -> Callable[[ChatContext], bool]:
    """A request whose last item is the person's message ``text``."""

    def matches(ctx: ChatContext) -> bool:
        last = ctx.items[-1] if ctx.items else None
        return isinstance(last, ChatMessage) and last.role == "user" and last.text_content == text

    return matches


front = Agent(
    instructions="front desk",
    llm=FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book me in",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="ask", arguments='{"request": "a booking"}', call_id="c1")
                ],
            ),
            # no response is scripted for the answer: the model is not asked about it
            FakeLLMResponse(
                input="booked for Jane O'Neil-Smith <j.o@example.co.uk>",
                content="All set.",
                ttft=0,
                duration=0,
            ),
        ]
    ),
)


@front.tool
async def ask(ctx: RunContext, request: str) -> str | InputRequired:
    """The back office.

    Args:
        request: the request, in full
    """
    if request == "a booking":
        return InputRequired("Name and email?", tool=ask)
    return f"booked for {request}"


async def desk(ctx: A2ASessionContext[Any]) -> str:
    result = await front.run(ctx.message, message_history=ctx.message_history)
    return result.input_required or str(result.output)


async def test_the_answer_goes_to_the_tool_that_asked(agentdb_storage: AgentDBStorage) -> None:
    server, _ = make_server(agentdb_storage, desk)
    async with http_client(server) as http:
        client = await stock_client(http)
        first = payloads(await send(client, user_message("book me in", "m1")))
        ctx_id = first[0].context_id
        asked = [p for p in first if hasattr(p, "status")][-1].status
        assert asked.state == pb.TaskState.TASK_STATE_INPUT_REQUIRED
        assert asked.message.parts[0].text == "Name and email?"

        answer = "Jane O'Neil-Smith <j.o@example.co.uk>"
        second = payloads(await send(client, user_message(answer, "m2", context_id=ctx_id)))
        done = [p for p in second if hasattr(p, "status")][-1].status
        await client.close()

    assert done.state == pb.TaskState.TASK_STATE_COMPLETED
    artifacts = [p for p in second if hasattr(p, "artifact")]
    assert artifacts and artifacts[-1].artifact.parts[0].text == "All set."

    db = await agentdb_storage.open(ctx_id)
    try:
        history = [row.payload for row in await db.history("front-desk")]
    finally:
        db.close()
    calls = [i for i in history if i["type"] == "function_call" and i["name"] == "ask"]
    outputs = {i["call_id"]: i["output"] for i in history if i["type"] == "function_call_output"}
    assert outputs["c1"] == answer, "the answer is the pending call's output"
    assert len(calls) == 2 and calls[1]["arguments"] == f'{{"request": "{answer}"}}'
    assert outputs[calls[1]["call_id"]] == f"booked for {answer}"


@pytest.mark.rule("R44", "R41", "B8", "B28")
@pytest.mark.xfail(strict=True, reason="R44, B8: InputRequired takes tool= and timeout=")
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
