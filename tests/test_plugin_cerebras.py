from __future__ import annotations

import os

import pytest

from livekit.agents import Agent, RunContext
from livekit.plugins.cerebras import LLM

pytestmark = pytest.mark.plugin("cerebras")

if not os.environ.get("CEREBRAS_API_KEY"):
    pytest.skip("needs CEREBRAS_API_KEY", allow_module_level=True)

# gpt-oss-120b supports both basic chat and function calling.
CHAT_MODEL = "gpt-oss-120b"
TOOL_MODEL = "gpt-oss-120b"


def _cerebras_llm(**kwargs) -> LLM:
    return LLM(model=CHAT_MODEL, **kwargs)


def _assistant(model: LLM) -> Agent:
    return Agent(instructions="You are a helpful assistant.", llm=model)


def _weather_agent(model: LLM) -> Agent:
    agent = _assistant(model)

    @agent.tool
    async def get_weather(ctx: RunContext, location: str) -> str:
        """Get the current weather for a location.
        Args:
            location: The city name
        """
        return f"The weather in {location} is sunny, 72°F."

    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gzip", "msgpack"), [(True, True), (True, False), (False, True), (False, False)]
)
async def test_chat(gzip: bool, msgpack: bool):
    """Basic chat completion returns an assistant message, with each request body encoding."""
    async with _cerebras_llm(gzip_compression=gzip, msgpack_encoding=msgpack) as model:
        result = await _assistant(model).run("Say hello in exactly one word.")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_function_call():
    """LLM can invoke a tool and the result is returned."""
    async with LLM(model=TOOL_MODEL) as model:
        result = await _weather_agent(model).run("What is the weather in Tokyo?")
        result.expect.next_event().is_function_call(
            name="get_weather", arguments={"location": "Tokyo"}
        )
        result.expect.next_event().is_function_call_output(
            output="The weather in Tokyo is sunny, 72°F."
        )
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()
