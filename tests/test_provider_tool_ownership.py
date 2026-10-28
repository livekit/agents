"""Custom host-provider tools do not need a native wire declaration."""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
)

from livekit.agents import Agent, llm
from livekit.plugins.anthropic.tools import ComputerUse
from livekit.plugins.openai.responses import LLM as ResponsesLLM
from livekit.plugins.openai.tools import WebSearch
from tests.fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


async def test_custom_provider_tool_reaches_its_plugin_llm_by_identity() -> None:
    class CustomProviderTool(llm.ProviderTool):
        pass

    class RecordingLLM(FakeLLM):
        offered: list[llm.Tool] | None = None

        def chat(self, **kwargs: Any) -> llm.LLMStream:
            self.offered = kwargs.get("tools")
            return super().chat(**kwargs)

    provider = CustomProviderTool(id="custom_provider_tool")
    model = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hello",
                content="done",
                ttft=0,
                duration=0,
            )
        ]
    )
    agent = Agent(instructions="Help.", llm=model, tools=[provider])
    result = await agent.run("hello")
    assert result.status == "completed"
    assert model.offered == [provider]
    assert model.offered[0] is provider


@pytest.mark.parametrize("reverse", [False, True])
def test_provider_id_and_function_name_collision_is_rejected_atomically(reverse: bool) -> None:
    @llm.function_tool
    async def computer() -> str:
        return "done"

    @llm.function_tool
    async def lookup() -> str:
        return "found"

    context = llm.ToolContext([lookup])
    tools: list[llm.Tool] = [ComputerUse(), computer]
    if reverse:
        tools.reverse()
    with pytest.raises(ValueError, match="provider.*computer"):
        context.update_tools(tools)
    assert list(context.function_tools) == ["lookup"]


@dataclass
class _Answer:
    city: str


async def test_host_sdk_receives_and_core_executes_its_virtual_output_tool() -> None:
    received: list[dict[str, Any]] = []
    call = ResponseFunctionToolCall(
        type="function_call", name="final_result", call_id="final", arguments='{"city":"Paris"}'
    )
    response = Response(
        id="response",
        created_at=0,
        model="gpt-4.1",
        object="response",
        output=[call],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
    )
    events = [
        ResponseCreatedEvent(type="response.created", sequence_number=0, response=response),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done", sequence_number=1, output_index=0, item=call
        ),
        ResponseCompletedEvent(type="response.completed", sequence_number=2, response=response),
    ]

    async def respond(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.Response(
            content_type="text/event-stream",
            text="".join("data: " + event.model_dump_json() + "\n\n" for event in events),
        )

    @llm.function_tool
    async def lookup() -> str:
        return "Paris"

    app = web.Application()
    app.router.add_post("/responses", respond)
    async with TestServer(app) as server:
        model = ResponsesLLM(
            api_key="test", base_url=str(server.make_url("/")), use_websocket=False
        )
        agent = Agent(
            instructions="Find a city.", output_type=_Answer, llm=model, tools=[lookup, WebSearch()]
        )
        try:
            async with asyncio.timeout(5):
                result = await agent.run("Find a city.")
            assert result.status == "completed"
            assert result.output == _Answer("Paris")
        finally:
            await model.aclose()
    tools = received[0]["tools"]
    assert {tool["name"] for tool in tools if tool["type"] == "function"} == {
        "lookup",
        "final_result",
    }
    assert any(tool["type"] == "web_search" for tool in tools)
    final = next(tool for tool in tools if tool.get("name") == "final_result")
    assert final["parameters"]["properties"]["city"]["type"] == "string"
