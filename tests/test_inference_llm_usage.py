from __future__ import annotations

import httpx
import openai as openai_sdk
import pytest

from livekit.agents import llm
from livekit.plugins import openai

pytestmark = pytest.mark.unit


# Some OpenAI-compatible proxies send a final chunk whose ``usage`` object is present
# but has null numeric fields (rather than omitting ``usage`` entirely). The SDK-side
# usage model is built leniently, so those nulls reach our ``CompletionUsage``
# construction as ``None``. See issue #6595.
_STREAM_WITH_NULL_USAGE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[],"created":0,"model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":null,"prompt_tokens":7,"total_tokens":null}}

data: [DONE]

"""


class _NullUsageTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_STREAM_WITH_NULL_USAGE,
            request=request,
        )


async def test_null_usage_fields_do_not_crash_stream() -> None:
    # A null field inside a non-null ``usage`` object must not raise a
    # ValidationError (which the stream would wrap as a non-retryable
    # APIConnectionError and terminate the session). Missing counts default to 0.
    client = openai_sdk.AsyncClient(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=_NullUsageTransport()),
    )
    model = openai.LLM(model="m", client=client)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    usage_chunks: list[llm.CompletionUsage] = []
    stream = model.chat(chat_ctx=chat_ctx)
    try:
        async for chunk in stream:
            if chunk.usage is not None:
                usage_chunks.append(chunk.usage)
    finally:
        await stream.aclose()
        await model.aclose()

    assert len(usage_chunks) == 1
    usage = usage_chunks[0]
    assert usage.completion_tokens == 0
    assert usage.prompt_tokens == 7
    assert usage.total_tokens == 0


# The gateway reports Gemini's thinking tokens as ``completion_tokens_details.reasoning_tokens``
# on the OpenAI-compatible stream, and counts them inside ``completion_tokens`` (which it derives
# as total - prompt). Reasoning is therefore a subset of the completion tokens, never an addition.
_STREAM_WITH_REASONING_USAGE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"m","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[],"created":0,"model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":40,"prompt_tokens":7,"total_tokens":47,"completion_tokens_details":{"reasoning_tokens":32}}}

data: [DONE]

"""


class _ReasoningUsageTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_STREAM_WITH_REASONING_USAGE,
            request=request,
        )


async def test_reasoning_tokens_are_reported_from_completion_tokens_details() -> None:
    client = openai_sdk.AsyncClient(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=_ReasoningUsageTransport()),
    )
    model = openai.LLM(model="m", client=client)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    usage_chunks: list[llm.CompletionUsage] = []
    stream = model.chat(chat_ctx=chat_ctx)
    try:
        async for chunk in stream:
            if chunk.usage is not None:
                usage_chunks.append(chunk.usage)
    finally:
        await stream.aclose()
        await model.aclose()

    assert len(usage_chunks) == 1
    usage = usage_chunks[0]
    assert usage.reasoning_tokens == 32
    assert usage.completion_tokens == 40
    assert usage.total_tokens == 47


async def test_reasoning_tokens_default_to_zero_without_details() -> None:
    # Providers that don't break reasoning out omit ``completion_tokens_details`` entirely.
    client = openai_sdk.AsyncClient(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=_NullUsageTransport()),
    )
    model = openai.LLM(model="m", client=client)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    usage_chunks: list[llm.CompletionUsage] = []
    stream = model.chat(chat_ctx=chat_ctx)
    try:
        async for chunk in stream:
            if chunk.usage is not None:
                usage_chunks.append(chunk.usage)
    finally:
        await stream.aclose()
        await model.aclose()

    assert usage_chunks[0].reasoning_tokens == 0
