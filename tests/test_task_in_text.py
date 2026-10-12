"""An ``AgentTask`` awaited in text (docs/agent-tasks.md): the call stays pending, the task
rides the lane's agent stack with its arguments and state, each message is a turn of the task,
and ``Completed`` becomes the pending call's output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import Agent, AgentTask, Completed, RunContext, function_tool
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import FunctionToolCall
from livekit.agents.storage import AgentDBStorage

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_a2a_server import http_client, make_server, payloads, send, stock_client, user_message

pytestmark = [pytest.mark.unit]


@dataclass
class Draft:
    nights: int | None = None


@dataclass
class BookRoom(AgentTask[str, Draft]):
    guests: int

    def __post_init__(self) -> None:
        super().__init__(
            instructions="Book the room.",
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="book a room", content="How many nights?", ttft=0, duration=0
                    ),
                    FakeLLMResponse(
                        input="two nights",
                        content="",
                        ttft=0,
                        duration=0,
                        tool_calls=[
                            FunctionToolCall(
                                name="set_nights", arguments='{"nights": 2}', call_id="c2"
                            )
                        ],
                    ),
                    FakeLLMResponse(
                        input="nights recorded", content="Shall I book it?", ttft=0, duration=0
                    ),
                    FakeLLMResponse(
                        input="yes",
                        content="",
                        ttft=0,
                        duration=0,
                        tool_calls=[FunctionToolCall(name="confirm", arguments="{}", call_id="c3")],
                    ),
                ]
            ),
        )

    @function_tool
    async def set_nights(self, ctx: RunContext, nights: int) -> str:
        """Record the stay length.

        Args:
            nights: how many nights
        """
        self.state.nights = nights
        return "nights recorded"

    @function_tool
    async def confirm(self, ctx: RunContext) -> Completed[str]:
        """Write the booking."""
        return Completed(f"HTL-{self.guests}{self.state.nights}")


front = Agent(
    instructions="front desk",
    llm=FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book a room",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="start_booking", arguments='{"guests": 2}', call_id="c1")
                ],
            ),
            FakeLLMResponse(input="HTL-22", content="Booked, HTL-22.", ttft=0, duration=0),
        ]
    ),
)


@front.tool
async def start_booking(ctx: RunContext, guests: int) -> str:
    """Book a room.

    Args:
        guests: the party size
    """
    code = await BookRoom(guests=guests)  # turns pass here
    return code


async def desk(ctx: A2ASessionContext[Any]) -> str:
    result = await front.run(ctx.message, message_history=ctx.message_history)
    return result.input_required or str(result.output)


async def test_a_task_awaited_in_text_holds_the_conversation_until_completed(
    agentdb_storage: AgentDBStorage,
) -> None:
    server, _ = make_server(agentdb_storage, desk)
    async with http_client(server) as http:
        client = await stock_client(http)
        first = payloads(await send(client, user_message("book a room", "m1")))
        ctx_id = first[0].context_id
        asked = [p for p in first if hasattr(p, "status")][-1].status
        assert asked.state == pb.TaskState.TASK_STATE_INPUT_REQUIRED
        assert asked.message.parts[0].text == "How many nights?"

        second = payloads(await send(client, user_message("two nights", "m2", context_id=ctx_id)))
        again = [p for p in second if hasattr(p, "status")][-1].status
        assert again.message.parts[0].text == "Shall I book it?"

        third = payloads(await send(client, user_message("yes", "m3", context_id=ctx_id)))
        done = [p for p in third if hasattr(p, "status")][-1].status
        await client.close()

    assert done.state == pb.TaskState.TASK_STATE_COMPLETED  # the root phrased the task's value
    artifacts = [p for p in third if hasattr(p, "artifact")]
    assert artifacts and artifacts[-1].artifact.parts[0].text == "Booked, HTL-22."

    db = await agentdb_storage.open(ctx_id)
    try:
        lane = await db.lane("front-desk")
        assert lane is not None and not lane.agent_stack
        history = [row.payload for row in await db.history("front-desk")]
    finally:
        db.close()
    outputs = {i["call_id"]: i["output"] for i in history if i["type"] == "function_call_output"}
    assert outputs["c1"] == "HTL-22", "the task's value is the pending call's output"
    assert outputs["c2"] == "nights recorded"
    handoffs = [
        (i["old_agent_id"], i["new_agent_id"]) for i in history if i["type"] == "agent_handoff"
    ]
    assert handoffs == [("front", "book_room"), ("book_room", "front")]
