"""A tool's no-reply outcome: it does not suppress a sibling's reply, and a turn it ends has
no output."""

import pytest

from livekit.agents import Agent, RunContext, StopResponse, ToolResult
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


async def test_tool_result_does_not_suppress_other_tools_replies() -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="send and lookup",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="send", arguments="{}", call_id="send-1"),
                    FunctionToolCall(name="lookup", arguments="{}", call_id="lookup-1"),
                ],
            ),
            FakeLLMResponse(input="Found", content="Done.", ttft=0, duration=0),
        ]
    )
    agent = Agent(instructions="test", llm=model)

    @agent.tool
    async def send(ctx: RunContext) -> ToolResult:
        """Send digits."""
        return ToolResult("Sent", reply_required=False)

    @agent.tool
    async def lookup(ctx: RunContext) -> str:
        """Look something up."""
        return "Found"

    result = await agent.run("send and lookup")
    assert result.status == "completed" and result.output == "Done."
    outputs = {i.call_id: i for i in result.new_messages() if i.type == "function_call_output"}
    assert set(outputs) == {"send-1", "lookup-1"}
    assert outputs["send-1"].output == "Sent" and not outputs["send-1"].reply_required
    assert outputs["lookup-1"].output == "Found" and outputs["lookup-1"].reply_required
    assert result.usage.requests == 2


@pytest.mark.parametrize("output_type", [str, int])
async def test_a_turn_stopped_by_a_tool_has_no_output_not_a_completed_none(
    output_type: type,
) -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hang up",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="hang_up", arguments="{}", call_id="h-1")],
            )
        ]
    )
    agent = Agent(instructions="test", llm=model, output_type=output_type)

    @agent.tool
    async def hang_up(ctx: RunContext) -> None:
        """End the call."""
        raise StopResponse()

    result = await agent.run("hang up")
    assert result.status == "no_output"
    assert not hasattr(result, "output")
    assert [i.type for i in result.new_messages()][-1] == "function_call_output"
