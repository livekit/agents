"""``InputRequired(tool=)`` from a conversation tool in text (docs/tools.md): the next message
is handed to the named tool as its argument before any model request, as the person wrote it,
and the model goes on from that tool's output."""

from __future__ import annotations

from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import FunctionToolCall
from livekit.agents.storage import AgentDBStorage

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_a2a_server import http_client, make_server, payloads, send, stock_client, user_message

pytestmark = [pytest.mark.unit]

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
