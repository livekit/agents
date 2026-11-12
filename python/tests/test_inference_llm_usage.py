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


# Catches: the quota fields returned as pairs, so the 429 handler's `**` raised TypeError
# instead of the APIStatusError.
def test_a_rate_limit_logs_the_quota_the_gateway_stamped() -> None:
    from livekit.agents.inference import extract_quota_usage

    headers = httpx.Headers({"x-livekit-inference-rpm-limit": "60", "content-type": "text/plain"})
    assert {**extract_quota_usage(headers)} == {"rpm_limit": "60"}


# Catches: the generated gateway stream dropping the context's items, the pydantic
# ``response_format`` conversion, or the call's ``inference_class`` (which overrides the client's).
async def test_a_gateway_completion_sends_its_context_format_and_class() -> None:
    import json

    from pydantic import BaseModel

    from livekit.agents import inference

    class Answer(BaseModel):
        value: int

    async with serve(_STREAM_WITH_NULL_USAGE) as (base_url, received):
        model = inference.LLM(
            "openai/gpt-4.1-mini",
            base_url=base_url,
            api_key="key",
            api_secret="secret",
            inference_class="high",
        )
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content="hi")
        try:
            async with model.chat(
                chat_ctx=chat_ctx, response_format=Answer, inference_class="low"
            ) as completion:
                async for _ in completion:
                    pass
        finally:
            await model.aclose()
    (request,) = received
    body = json.loads(request.body)
    assert body["messages"][-1]["content"] == "hi"
    assert body["response_format"]["json_schema"]["name"] == "Answer"
    assert request.headers["x-livekit-inference-priority"] == "low"
