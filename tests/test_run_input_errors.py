"""Model failures propagate through the public run boundary; silence remains a valid turn."""

from typing import Any

import pytest

from livekit.agents import Agent, APIStatusError
from livekit.agents.llm import LLMStream

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class FailingLLM(FakeLLM):
    def chat(self, **kwargs: Any) -> LLMStream:
        raise APIStatusError("object cannot be found", status_code=401, retryable=False)


async def test_run_preserves_the_provider_error_status() -> None:
    agent = Agent(instructions="test", llm=FailingLLM())
    with pytest.raises(APIStatusError) as error:
        await agent.run("hello")
    assert error.value.status_code == 401


async def test_empty_model_completion_is_not_an_error() -> None:
    agent = Agent(instructions="test", llm=FakeLLM())
    result = await agent.run("okay, thank you")
    assert result.status == "completed"
    assert not any(i.type == "message" and i.role == "assistant" for i in result.new_messages())
