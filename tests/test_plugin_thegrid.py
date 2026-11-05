from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool, llm
from livekit.plugins.thegrid import LLM

pytestmark = pytest.mark.plugin("thegrid")

# text-standard is the general conversation tier; agent-standard is tuned for
# multi-step tool use. Both accept the OpenAI tool-calling fields.
CHAT_MODEL = "text-standard"
TOOL_MODEL = "agent-standard"


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")

    @function_tool
    async def get_weather(self, ctx: RunContext, location: str) -> str:
        """Get the current weather for a location.
        Args:
            location: The city name
        """
        return f"The weather in {location} is sunny, 72°F."


@pytest.mark.asyncio
async def test_chat():
    """Basic chat completion returns a non-empty assistant message."""
    async with LLM(model=CHAT_MODEL) as model, AgentSession(llm=model) as sess:
        await sess.start(Agent(instructions="You are a helpful assistant."))
        result = await sess.run(user_input="Say hello in exactly one word.")
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_function_call():
    """LLM can invoke a tool and the result is returned."""
    async with LLM(model=TOOL_MODEL) as model, AgentSession(llm=model) as sess:
        await sess.start(WeatherAgent())
        result = await sess.run(user_input="What is the weather in Tokyo?")
        result.expect.next_event().is_function_call(
            name="get_weather", arguments={"location": "Tokyo"}
        )
        result.expect.next_event().is_function_call_output(
            output="The weather in Tokyo is sunny, 72°F."
        )
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_streaming():
    """Streaming chat returns content via the LLM directly."""
    async with LLM(model=CHAT_MODEL) as model:
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="You are a helpful assistant.")
        chat_ctx.add_message(role="user", content="Count from 1 to 5.")

        stream = model.chat(chat_ctx=chat_ctx)
        text = ""
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
        await stream.aclose()

        assert len(text) > 0, "Expected non-empty streaming response"
        assert "3" in text, "Expected the count to include '3'"
