"""Tool contracts through the public agent execution path."""

import json
from dataclasses import dataclass
from typing import Annotated

import pytest
from pydantic import AfterValidator, BaseModel

from livekit.agents import Agent, RunContext, ToolResult, function_tool
from livekit.agents.llm import FunctionToolCall, Toolset

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def model_call(name: str, arguments: str) -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="do it",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name=name, arguments=arguments, call_id="call-1")],
            )
        ]
    )


async def test_application_validator_changes_the_value_received_by_the_tool() -> None:
    seen: list[str] = []
    agent = Agent(instructions="test", llm=model_call("label", '{"value": "hello"}'))

    @agent.tool
    async def label(
        ctx: RunContext, value: Annotated[str, AfterValidator(str.upper)]
    ) -> ToolResult:
        """Label an item."""
        seen.append(value)
        return ToolResult(value, reply_required=False)

    await agent.run("do it")
    assert seen == ["HELLO"]


async def test_toolset_methods_serialize_nested_application_results_as_json() -> None:
    class Seat(BaseModel):
        row: int

    @dataclass
    class Flight:
        number: str
        seats: list[Seat]

    class Flights(Toolset):
        @function_tool
        async def list_flights(self, ctx: RunContext) -> ToolResult:
            """List flights."""
            return ToolResult([Flight("SK204", [Seat(row=3)])], reply_required=False)

    agent = Agent(
        instructions="test", tools=[Flights(id="flights")], llm=model_call("list_flights", "{}")
    )
    result = await agent.run("do it")
    output = next(i for i in result.new_messages() if i.type == "function_call_output")
    assert not output.is_error
    assert json.loads(output.output) == [{"number": "SK204", "seats": [{"row": 3}]}]


async def test_unsupported_tool_result_is_an_error_in_model_history() -> None:
    agent = Agent(instructions="test", llm=model_call("invalid", "{}"))

    @agent.tool
    async def invalid(ctx: RunContext) -> ToolResult:
        """Return an unsupported value."""
        return ToolResult(object(), reply_required=False)

    result = await agent.run("do it")
    output = next(i for i in result.new_messages() if i.type == "function_call_output")
    assert output.is_error


async def test_agent_history_is_read_only_inside_a_tool_and_unavailable_after_the_run() -> None:
    from livekit.agents.llm import ChatMessage

    agent = Agent(instructions="test", llm=model_call("inspect_history", "{}"))

    @agent.tool
    async def inspect_history(ctx: RunContext) -> ToolResult:
        """Inspect the conversation."""
        history = agent.chat_ctx
        before = list(history.items)
        message = ChatMessage(role="user", content=["injected"])
        with pytest.raises(RuntimeError):
            history.items.append(message)
        with pytest.raises(RuntimeError):
            history.insert(message)
        with pytest.raises(RuntimeError):
            history.items = [message]
        assert history.items == before
        return ToolResult("checked", reply_required=False)

    result = await agent.run("do it")
    assert not any(
        i.type == "message" and i.text_content == "injected" for i in result.all_messages()
    )
    output = next(i for i in result.new_messages() if i.type == "function_call_output")
    assert output.output == "checked" and not output.is_error
    with pytest.raises(RuntimeError):
        _ = agent.chat_ctx
