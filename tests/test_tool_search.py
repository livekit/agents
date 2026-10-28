"""Discover and invoke beta tools through the public agent execution boundary."""

from collections import deque
from typing import Annotated, Any

import pytest
from pydantic import Field

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.beta import ToolProxyToolset, ToolSearchToolset
from livekit.agents.llm import ChatContext, FunctionToolCall, LLMStream, Tool
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


class ScriptedLLM(FakeLLM):
    def __init__(self, *steps: FunctionToolCall | str) -> None:
        super().__init__()
        self.steps = deque(steps)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        incoming = chat_ctx.items[-1]
        text = incoming.text_content if incoming.type == "message" else incoming.output
        step = self.steps.popleft()
        self.fake_response_map[text] = FakeLLMResponse(
            input=text,
            content=step if isinstance(step, str) else "",
            tool_calls=[step] if isinstance(step, FunctionToolCall) else [],
            ttft=0,
            duration=0,
        )
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)


async def test_discovered_tool_executes_on_the_next_model_step() -> None:
    booked: list[int] = []

    @function_tool
    async def reserve_seats(ctx: RunContext, count: int) -> str:
        """Reserve seats for the customer."""
        booked.append(count)
        return f"reserved {count} seats"

    agent = Agent(
        instructions="Find and use the reservations tool.",
        tools=[ToolSearchToolset(id="reservations", tools=[reserve_seats])],
        llm=ScriptedLLM(
            FunctionToolCall(
                name="tool_search", arguments='{"query":"reserve seats"}', call_id="s"
            ),
            FunctionToolCall(name="reserve_seats", arguments='{"count":2}', call_id="r"),
            "Your two seats are reserved.",
        ),
    )
    result = await agent.run("Reserve two seats.")
    assert booked == [2], [
        i.output for i in result.new_messages() if i.type == "function_call_output"
    ]
    assert result.status == "completed" and result.output == "Your two seats are reserved."
    output = next(
        i for i in result.new_messages() if i.type == "function_call_output" and i.call_id == "r"
    )
    assert output.output == "reserved 2 seats" and not output.is_error


@pytest.mark.parametrize(
    "count", [2, 0], ids=["executes-valid-arguments", "refuses-invalid-arguments"]
)
async def test_proxy_invokes_a_discovered_tool_only_with_valid_arguments(count: int) -> None:
    booked: list[int] = []

    @function_tool
    async def reserve_seats(ctx: RunContext, count: Annotated[int, Field(ge=1)]) -> str:
        """Reserve seats for the customer."""
        booked.append(count)
        return f"reserved {count} seats"

    agent = Agent(
        instructions="Search for reservations, then use call_tool.",
        tools=[ToolProxyToolset(id="reservations", tools=[reserve_seats])],
        llm=ScriptedLLM(
            FunctionToolCall(
                name="tool_search", arguments='{"query":"reserve seats"}', call_id="s"
            ),
            FunctionToolCall(
                name="call_tool",
                arguments='{"name":"reserve_seats","parameters":{"count":' + str(count) + "}}",
                call_id="r",
            ),
            "Reservation processed.",
        ),
    )
    result = await agent.run("Reserve seats.")
    output = next(
        i for i in result.new_messages() if i.type == "function_call_output" and i.call_id == "r"
    )
    if count:
        assert booked == [2]
        assert output.output == "reserved 2 seats" and not output.is_error
    else:
        assert booked == []
        assert output.is_error and "count" in output.output
    assert result.status == "completed" and result.output == "Reservation processed."
