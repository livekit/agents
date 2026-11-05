from __future__ import annotations

from typing import Any

import msgpack
import pytest

from livekit.agents import llm
from livekit.plugins.cerebras import LLM

from .utils.chat_completions import Received, serve

pytestmark = pytest.mark.unit


_STREAM_RESPONSE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"gpt-oss-120b","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"gpt-oss-120b","object":"chat.completion.chunk"}

data: [DONE]

"""


async def _capture_chat_request(**options: Any) -> Received:
    async with serve(_STREAM_RESPONSE) as (base_url, received):
        model = LLM(model="gpt-oss-120b", api_key="test-key", base_url=base_url, **options)
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content="hi")
        try:
            async with model.chat(chat_ctx=chat_ctx) as stream:
                async for _ in stream:
                    pass
        finally:
            await model.aclose()
    assert len(received) == 1
    return received[0]


def _payload(request: Received) -> dict[str, Any]:
    payload = msgpack.unpackb(request.body, raw=False)
    assert isinstance(payload, dict)
    return payload


# Catches: the Cerebras defaults (gzip and msgpack) not reaching the native request, or an
# option missing from the encoded body.
@pytest.mark.asyncio
async def test_max_completion_tokens_reaches_compressed_msgpack_request() -> None:
    request = await _capture_chat_request(max_completion_tokens=321)

    assert request.headers["content-type"] == "application/vnd.msgpack"
    assert request.headers["content-encoding"] == "gzip"
    assert request.headers["authorization"] == "Bearer test-key"
    assert _payload(request)["max_completion_tokens"] == 321


@pytest.mark.asyncio
async def test_omitted_max_completion_tokens_does_not_change_request() -> None:
    without_limit = _payload(await _capture_chat_request())
    with_limit = _payload(await _capture_chat_request(max_completion_tokens=321))

    assert "max_completion_tokens" not in without_limit
    assert with_limit.pop("max_completion_tokens") == 321
    assert with_limit == without_limit


@pytest.mark.asyncio
async def test_plain_json_when_both_encodings_are_off() -> None:
    request = await _capture_chat_request(gzip_compression=False, msgpack_encoding=False)

    assert request.headers["content-type"] == "application/json"
    assert "content-encoding" not in request.headers
