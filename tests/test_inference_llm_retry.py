"""Retry eligibility of a failed inference LLM stream.

A stream that dies having emitted nothing the caller can see must be retried:
provider metadata (a gateway deployment stamp, a thought signature) and token
counts are not generation.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import openai
import pytest

from livekit.agents import APIConnectOptions, APITimeoutError, llm
from livekit.agents.inference import LLM

pytestmark = pytest.mark.unit


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk(delta: dict) -> bytes:
    return _sse(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "google/gemma-4-31b-it",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


# The gateway stamps its deployment and billing tier onto the leading delta, which
# carries no content of its own.
_METADATA_ONLY = _chunk(
    {
        "role": "assistant",
        "extra_content": {
            "livekit": {"inference_deployment": "d", "inference_tier_billed": "standard"}
        },
    }
)
_TEXT = _chunk({"role": "assistant", "content": "hello"})


class _StallingStream(httpx.AsyncByteStream):
    """Yields the given SSE bytes, then stalls out like a provider going quiet."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        raise httpx.ReadTimeout("stalled mid-stream")


def _llm_over(chunks: list[bytes]) -> tuple[LLM, list[httpx.Request]]:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_StallingStream(chunks),
        )

    # Long enough to keep PyJWT's short-key warning out of the suite output.
    fake_secret = "f" * 32
    llm_model = LLM(model="google/gemma-4-31b-it", api_key=fake_secret, api_secret=fake_secret)
    llm_model._client = openai.AsyncClient(
        api_key=fake_secret,
        base_url="http://inference.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm_model, attempts


async def _run(chunks: list[bytes], *, max_retry: int) -> tuple[Exception, list[httpx.Request]]:
    llm_model, attempts = _llm_over(chunks)
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    with pytest.raises(Exception) as exc_info:  # noqa: PT011
        async with llm_model.chat(
            chat_ctx=chat_ctx,
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0, timeout=5.0),
        ) as stream:
            async for _ in stream:
                pass

    return exc_info.value, attempts


@pytest.mark.asyncio
async def test_metadata_only_chunk_stays_retryable() -> None:
    error, attempts = await _run([_METADATA_ONLY], max_retry=2)

    assert len(attempts) == 3, "a stall after metadata alone must exhaust the retries"
    assert "after 3 attempts" in str(error)


@pytest.mark.asyncio
async def test_generated_text_is_not_retried() -> None:
    error, attempts = await _run([_METADATA_ONLY, _TEXT], max_retry=2)

    assert len(attempts) == 1, "text already sent to the caller must not be regenerated"
    assert "after" not in str(error)


@pytest.mark.asyncio
async def test_stream_read_timeout_is_a_timeout_error() -> None:
    error, _ = await _run([_TEXT], max_retry=0)

    assert isinstance(error, APITimeoutError), "a stalled stream body is a timeout, not a connect"
    assert "timed out" in str(error).lower()
