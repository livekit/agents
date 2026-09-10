"""Retry eligibility, and reported latency, of a failed inference LLM stream.

A stream that dies having emitted nothing the caller can see must be retried:
provider metadata (a gateway deployment stamp, a thought signature) and token
counts are not generation. The same line divides the latency the caller actually
waited from the moment a contentless chunk happened to arrive.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import httpx
import openai
import pytest

import livekit.agents.inference.llm as inference_llm
from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError, llm
from livekit.agents.inference import LLM
from livekit.agents.inference._utils import HEADER_SESSION_ID
from livekit.agents.metrics import LLMMetrics

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
_TOOL_NAME = _chunk(
    {
        "role": "assistant",
        "tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup"}}
        ],
    }
)
_TOOL_ARGS = _chunk(
    {"role": "assistant", "tool_calls": [{"index": 0, "function": {"arguments": '{"q":"x"}'}}]}
)
_TOOL_DONE = _sse(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "google/gemma-4-31b-it",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
)
_USAGE = _sse(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "google/gemma-4-31b-it",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
)


class _StallingStream(httpx.AsyncByteStream):
    """Yields the given SSE bytes, then stalls out like a provider going quiet."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        raise httpx.ReadTimeout("stalled mid-stream")


class _CompletedStream(httpx.AsyncByteStream):
    """Yields the given SSE bytes, then ends the stream cleanly."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        yield b"data: [DONE]\n\n"


def _llm_for(
    responder: Callable[[int], httpx.AsyncByteStream],
) -> tuple[LLM, list[httpx.Request], list[LLMMetrics]]:
    attempts: list[httpx.Request] = []
    metrics: list[LLMMetrics] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=responder(len(attempts)),
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
    llm_model.on("metrics_collected", metrics.append)
    return llm_model, attempts, metrics


def _drain(llm_model: LLM, *, max_retry: int):
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    return llm_model.chat(
        chat_ctx=chat_ctx,
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0, timeout=5.0),
    )


async def _run(chunks: list[bytes], *, max_retry: int) -> tuple[Exception, list[httpx.Request]]:
    llm_model, attempts, _ = _llm_for(lambda _: _StallingStream(chunks))

    with pytest.raises(Exception) as exc_info:  # noqa: PT011
        async with _drain(llm_model, max_retry=max_retry) as stream:
            async for _ in stream:
                pass

    return exc_info.value, attempts


async def _run_to_completion(
    responder: Callable[[int], httpx.AsyncByteStream], *, max_retry: int
) -> tuple[list[LLMMetrics], list[httpx.Request]]:
    llm_model, attempts, metrics = _llm_for(responder)

    # aclose() awaits the metrics monitor, so metrics are settled once the block exits
    async with _drain(llm_model, max_retry=max_retry) as stream:
        async for _ in stream:
            pass

    return metrics, attempts


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
async def test_completed_tool_call_is_not_retried() -> None:
    error, attempts = await _run([_METADATA_ONLY, _TOOL_NAME, _TOOL_ARGS, _TOOL_DONE], max_retry=2)

    assert len(attempts) == 1, "a tool call already sent to the caller must not be re-issued"
    assert "after" not in str(error)


@pytest.mark.asyncio
async def test_partial_tool_arguments_stay_retryable() -> None:
    _, attempts = await _run([_METADATA_ONLY, _TOOL_NAME, _TOOL_ARGS], max_retry=2)

    assert len(attempts) == 3, "arguments still streaming have reached nobody"


@pytest.mark.asyncio
async def test_retries_get_fresh_sdk_session_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    session_ids = iter(("inference_first", "inference_second", "inference_third"))
    monkeypatch.setattr(
        inference_llm,
        "get_inference_headers",
        lambda **_kwargs: {HEADER_SESSION_ID: next(session_ids)},
    )
    llm_model, attempts, _ = _llm_for(lambda _: _StallingStream([_METADATA_ONLY]))
    llm_model._opts.extra_kwargs["extra_headers"] = {HEADER_SESSION_ID: "caller_value"}

    with pytest.raises(APIConnectionError):
        async with _drain(llm_model, max_retry=2) as stream:
            async for _ in stream:
                pass

    assert [request.headers[HEADER_SESSION_ID] for request in attempts] == [
        "inference_first",
        "inference_second",
        "inference_third",
    ]


@pytest.mark.asyncio
async def test_stream_read_timeout_is_a_timeout_error() -> None:
    error, _ = await _run([_TEXT], max_retry=0)

    assert isinstance(error, APITimeoutError), "a stalled stream body is a timeout, not a connect"
    assert "timed out" in str(error).lower()


@pytest.mark.asyncio
async def test_ttft_spans_the_retry() -> None:
    """The clock runs until generation, not until the failed attempt's metadata."""

    def responder(attempt: int) -> httpx.AsyncByteStream:
        if attempt == 1:
            return _StallingStream([_METADATA_ONLY])
        return _CompletedStream([_METADATA_ONLY, _TEXT, _USAGE])

    metrics, attempts = await _run_to_completion(responder, max_retry=2)

    assert len(attempts) == 2
    assert len(metrics) == 1
    # the first retry always waits APIConnectOptions._interval_for_retry(0) == 0.1s
    assert metrics[0].ttft > 0.09, "ttft latched on the failed attempt's metadata chunk"
    assert metrics[0].ttft <= metrics[0].duration


@pytest.mark.asyncio
async def test_token_counts_survive_a_response_that_generates_nothing() -> None:
    """Metadata and usage but no output: ttft is unmeasurable, the tokens still are."""
    metrics, attempts = await _run_to_completion(
        lambda _: _CompletedStream([_METADATA_ONLY, _USAGE]), max_retry=2
    )

    assert len(attempts) == 1
    assert len(metrics) == 1
    assert metrics[0].ttft == -1
    assert metrics[0].completion_tokens == 7
    assert metrics[0].total_tokens == 18
