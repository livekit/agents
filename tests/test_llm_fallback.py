import asyncio
import contextlib

import pytest

from livekit.agents.llm import ChatContext
from livekit.agents.llm.fallback_adapter import FallbackAdapter

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_fallback_adapter_attempt_timeout_bounds_time_to_first_chunk() -> None:
    slow = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="slow", ttft=1, duration=1)]
    )
    fast = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="fast", ttft=0, duration=0)]
    )
    fallback = FallbackAdapter([slow, fast], attempt_timeout=0.01)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hello")

    response = await fallback.chat(chat_ctx=chat_ctx).collect()

    assert response.text == "fast"
    if recovery_task := fallback._status[0].recovering_task:
        recovery_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recovery_task
    await fallback.aclose()
