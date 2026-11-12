"""Programmatic run assertions against a live inference model."""

import pytest

from livekit.agents import Agent, RunContext, inference

pytestmark = pytest.mark.evals


async def test_function_call_assertions_match_the_recorded_tool_effect() -> None:
    looked_up: list[str] = []
    model = inference.LLM(model="openai/gpt-4.1-mini")
    agent = Agent(instructions="Use lookup_weather to answer weather questions.", llm=model)

    @agent.tool
    async def lookup_weather(ctx: RunContext, location: str) -> str:
        """Get weather for a location."""
        looked_up.append(location)
        return "sunny with a temperature of 70 degrees."

    async with model:
        result = await agent.run("What is the weather in San Francisco?")
        result.expect.contains_function_call(
            name="lookup_weather", arguments={"location": "San Francisco"}
        )
        result.expect.contains_function_call_output(
            output="sunny with a temperature of 70 degrees."
        )
        result.expect.contains_message(role="assistant")
        assert looked_up == ["San Francisco"]
