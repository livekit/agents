from __future__ import annotations

from pathlib import Path

import httpx
import httpx2
import openai as openai_sdk
import pytest

from livekit.agents import llm
from livekit.plugins import groq, openai

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


class _NullUsageTransport(httpx2.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_STREAM_WITH_NULL_USAGE,
            request=request,
        )


async def test_openai_llm_uses_httpx2_and_warns_for_legacy_timeout() -> None:
    with pytest.warns(DeprecationWarning, match="LiveKit Agents 2.0") as warnings:
        model = openai.LLM(api_key="test-key", timeout=httpx.Timeout(5))

    try:
        assert isinstance(model._client._client, httpx2.AsyncClient)
        assert Path(warnings[0].filename).resolve() == Path(__file__).resolve()
    finally:
        await model.aclose()


async def test_openai_wrapper_warning_points_to_application() -> None:
    with pytest.warns(DeprecationWarning, match="LiveKit Agents 2.0") as warnings:
        model = groq.LLM(api_key="test-key", timeout=httpx.Timeout(5))

    try:
        assert Path(warnings[0].filename).resolve() == Path(__file__).resolve()
    finally:
        await model.aclose()


async def test_null_usage_fields_do_not_crash_stream() -> None:
    # A null field inside a non-null ``usage`` object must not raise a
    # ValidationError (which the stream would wrap as a non-retryable
    # APIConnectionError and terminate the session). Missing counts default to 0.
    client = openai_sdk.AsyncClient(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=_NullUsageTransport()),
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
