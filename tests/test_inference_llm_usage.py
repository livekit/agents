from __future__ import annotations

import httpx2 as httpx
import pytest

from livekit.agents import TokenUsage, llm
from livekit.plugins import openai

from .utils.chat_completions import serve

pytestmark = pytest.mark.unit


# Some OpenAI-compatible proxies send a final chunk whose ``usage`` object is present
# but has null numeric fields (rather than omitting ``usage`` entirely). Usage is
# built leniently, so those nulls reach our ``TokenUsage``
# construction as ``None``. See issue #6595.
_STREAM_WITH_NULL_USAGE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[],"created":0,"model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":null,"prompt_tokens":7,"total_tokens":null}}

data: [DONE]

"""


async def _usage(stream: bytes) -> list[TokenUsage]:
    """The usage chunks of one completion of `stream`, through the native client."""
    async with serve(stream) as (base_url, _):
        model = openai.LLM(model="m", api_key="test-key", base_url=base_url)
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content="hi")
        usage: list[TokenUsage] = []
        try:
            async with model.chat(chat_ctx=chat_ctx) as completion:
                async for chunk in completion:
                    if chunk.usage is not None:
                        usage.append(chunk.usage)
        finally:
            await model.aclose()
        return usage


async def test_null_usage_fields_do_not_crash_stream() -> None:
    # A null field inside a non-null ``usage`` object must not raise a
    # ValidationError (which the stream would wrap as a non-retryable
    # APIConnectionError and terminate the session). Missing counts default to 0.
    usage_chunks = await _usage(_STREAM_WITH_NULL_USAGE)
    assert len(usage_chunks) == 1
    usage = usage_chunks[0]
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (7, 0, 7)


# The gateway reports Gemini's thinking tokens as ``completion_tokens_details.reasoning_tokens``
# on the OpenAI-compatible stream, and counts them inside ``completion_tokens`` (which it derives
# as total - prompt). Reasoning is therefore a subset of the completion tokens, never an addition.
_STREAM_WITH_REASONING_USAGE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[],"created":0,"model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":40,"prompt_tokens":7,"total_tokens":47,"completion_tokens_details":{"reasoning_tokens":32}}}

data: [DONE]

"""


async def test_reasoning_tokens_are_reported_from_completion_tokens_details() -> None:
    usage_chunks = await _usage(_STREAM_WITH_REASONING_USAGE)
    assert len(usage_chunks) == 1
    usage = usage_chunks[0]
    assert usage.output_reasoning_tokens == 32
    assert usage.output_tokens == 40
    assert usage.total_tokens == 47


async def test_an_unreported_reasoning_breakdown_stays_absent() -> None:
    # Providers that don't break reasoning out omit ``completion_tokens_details`` entirely.
    usage_chunks = await _usage(_STREAM_WITH_NULL_USAGE)
    assert usage_chunks[0].output_reasoning_tokens is None


# Catches: the quota fields returned as pairs, so the 429 handler's `**` raised TypeError
# instead of the APIStatusError.
def test_a_rate_limit_logs_the_quota_the_gateway_stamped() -> None:
    from livekit.agents.inference import extract_quota_usage

    headers = httpx.Headers({"x-livekit-inference-rpm-limit": "60", "content-type": "text/plain"})
    assert {**extract_quota_usage(headers)} == {"rpm_limit": "60"}
