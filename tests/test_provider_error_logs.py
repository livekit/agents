"""Structured fields on provider retry and failure logs.

Cross-session log search filters on these fields, so the tests check the fields on the
log record, not only the message text.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable

import httpx2 as httpx
import openai
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    llm,
)
from livekit.agents._exceptions import error_log_fields
from livekit.agents.inference import LLM

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit

_MODEL = "google/gemma-4-31b-it"


def test_error_log_fields_for_status_error() -> None:
    error = APIStatusError("rate limited", status_code=429, request_id="req_1")

    assert error_log_fields(error) == {
        "error_type": "APIStatusError",
        "retryable": True,
        "status_code": 429,
        "request_id": "req_1",
    }


def test_error_log_fields_without_status() -> None:
    assert error_log_fields(APIConnectionError(retryable=False)) == {
        "error_type": "APIConnectionError",
        "retryable": False,
    }
    assert error_log_fields(ValueError("boom")) == {"error_type": "ValueError"}


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    return _sse(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


_USAGE = _sse(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": _MODEL,
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
)


class _Stream(httpx.AsyncByteStream):
    """Yields the given SSE bytes, then ends cleanly or stalls out."""

    def __init__(self, chunks: list[bytes], *, stall: bool = False) -> None:
        self._chunks = chunks
        self._stall = stall

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._stall:
            raise httpx.ReadTimeout("stalled mid-stream")
        yield b"data: [DONE]\n\n"


def _llm_for(responder: Callable[[], httpx.AsyncByteStream]) -> LLM:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=responder()
        )

    # long enough to keep PyJWT's short-key warning out of the suite output
    fake_secret = "f" * 32
    llm_model = LLM(model=_MODEL, api_key=fake_secret, api_secret=fake_secret)
    llm_model._client = openai.AsyncClient(
        api_key=fake_secret,
        base_url="http://inference.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm_model


async def _drain(llm_model: LLM, *, max_retry: int = 0) -> None:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    async with llm_model.chat(
        chat_ctx=chat_ctx,
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0, timeout=5.0),
    ) as stream:
        async for _ in stream:
            pass


def _records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(message)]


@pytest.mark.asyncio
async def test_llm_retry_warning_has_fields(caplog: pytest.LogCaptureFixture) -> None:
    llm_model = _llm_for(lambda: _Stream([], stall=True))

    with caplog.at_level(logging.WARNING), pytest.raises(APIConnectionError):
        await _drain(llm_model, max_retry=1)

    [record] = _records(caplog, "failed to generate LLM completion")
    assert record.provider == "livekit"
    assert record.model == _MODEL
    assert record.attempt == 1
    assert record.max_retry == 1
    assert record.error_type == "APITimeoutError"
    assert record.retryable is True


@pytest.mark.asyncio
async def test_length_finish_reason_warns(caplog: pytest.LogCaptureFixture) -> None:
    chunks = [_chunk({"role": "assistant", "content": "hel"}, finish_reason="length"), _USAGE]
    llm_model = _llm_for(lambda: _Stream(chunks))

    with caplog.at_level(logging.WARNING):
        await _drain(llm_model)

    [record] = _records(caplog, "LLM response stopped early")
    assert record.finish_reason == "length"
    assert record.completion_tokens == 7
    assert record.model == _MODEL
    assert not _records(caplog, "LLM returned an empty completion")


@pytest.mark.asyncio
async def test_empty_completion_warns(caplog: pytest.LogCaptureFixture) -> None:
    chunks = [_chunk({"role": "assistant"}, finish_reason="stop"), _USAGE]
    llm_model = _llm_for(lambda: _Stream(chunks))

    with caplog.at_level(logging.WARNING):
        await _drain(llm_model)

    assert len(_records(caplog, "LLM returned an empty completion")) == 1
    assert not _records(caplog, "LLM response stopped early")


@pytest.mark.asyncio
async def test_normal_completion_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    chunks = [_chunk({"role": "assistant", "content": "hello"}, finish_reason="stop"), _USAGE]
    llm_model = _llm_for(lambda: _Stream(chunks))

    with caplog.at_level(logging.WARNING):
        await _drain(llm_model)

    assert not _records(caplog, "LLM response stopped early")
    assert not _records(caplog, "LLM returned an empty completion")


@pytest.mark.asyncio
async def test_tts_no_audio_error_omits_text(caplog: pytest.LogCaptureFixture) -> None:
    tts = FakeTTS(fake_audio_duration=0.0)
    secret_text = "my account number is 1234"

    try:
        with caplog.at_level(logging.WARNING), pytest.raises(APIError) as exc_info:
            async with tts.synthesize(
                secret_text,
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=0.5),
            ) as stream:
                async for _ in stream:
                    pass
    finally:
        await tts.aclose()

    assert "no audio frames were pushed" in str(exc_info.value)
    assert secret_text not in str(exc_info.value)
    [record] = _records(caplog, "failed to synthesize speech")
    assert secret_text not in record.getMessage()
    assert record.error_type == "APIError"
    assert record.max_retry == 1
