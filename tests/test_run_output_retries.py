from __future__ import annotations

import pytest
from pydantic import BaseModel

from livekit.agents import AgentSession, RunOutputError
from livekit.agents.llm import FunctionToolCall, function_tool
from livekit.agents.voice.agent import AgentTask
from livekit.agents.voice.run_result import _OUTPUT_RETRY_PROMPT

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


class _Out(BaseModel):
    value: str


class _Task(AgentTask[_Out]):
    def __init__(self, output_retry_instructions: str | None = None) -> None:
        super().__init__(
            instructions="test", output_retry_instructions=output_retry_instructions
        )

    @function_tool
    async def submit_result(self, value: str) -> None:
        if not self.done():
            self.complete(_Out(value=value))


@pytest.mark.asyncio
async def test_output_retry_recovers() -> None:
    """A run that ends in prose is re-prompted and recovers the typed output."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="chatting instead", ttft=0.01, duration=0.02),
            FakeLLMResponse(
                input=_OUTPUT_RETRY_PROMPT,
                content="",
                ttft=0.01,
                duration=0.02,
                tool_calls=[
                    FunctionToolCall(
                        name="submit_result", arguments='{"value": "ok"}', call_id="call_1"
                    )
                ],
            ),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(_Task())
        result = await sess.run(user_input="hello", output_type=_Out)

    assert result.final_output.value == "ok"


@pytest.mark.asyncio
async def test_output_retry_custom_instructions() -> None:
    """The retry prompt is overridable, matching the keyterm-detection pattern."""
    custom = "Submit your analysis via the tool, nothing else."
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="chatting instead", ttft=0.01, duration=0.02),
            FakeLLMResponse(
                input=custom,
                content="",
                ttft=0.01,
                duration=0.02,
                tool_calls=[
                    FunctionToolCall(
                        name="submit_result", arguments='{"value": "custom"}', call_id="call_2"
                    )
                ],
            ),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(_Task(output_retry_instructions=custom))
        result = await sess.run(user_input="hello", output_type=_Out)

    assert result.final_output.value == "custom"


@pytest.mark.asyncio
async def test_output_retry_exhausted() -> None:
    """With no retry budget, a prose ending raises RunOutputError."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="chatting instead", ttft=0.01, duration=0.02),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(_Task())
        with pytest.raises(RunOutputError):
            await sess.run(user_input="hello", output_type=_Out, output_retries=0)
