"""The inference LLM honors the gateway's X-Should-Retry on error responses.

The gateway answers terminal quota denials (out of credits, spent request allowance,
suspended project) with a 429 and ``X-Should-Retry: false``; retrying those only adds
latency. Transient 429s (RPM/TPM/concurrency) carry no such header and stay retryable.
"""

from __future__ import annotations

import httpx2 as httpx
import openai
import pytest

from livekit.agents import APIConnectOptions, APIError, APIStatusError, llm
from livekit.agents.inference import LLM

pytestmark = pytest.mark.unit

_QUOTA_BODY = {"error": "credits exhausted", "category": "credits"}


async def _run_429(headers: dict[str, str]) -> tuple[APIStatusError, list[httpx.Request]]:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429, headers=headers, json=_QUOTA_BODY)

    # Long enough to keep PyJWT's short-key warning out of the suite output.
    fake_secret = "f" * 32
    llm_model = LLM(model="google/gemma-4-31b-it", api_key=fake_secret, api_secret=fake_secret)
    llm_model._client = openai.AsyncClient(
        api_key=fake_secret,
        base_url="http://inference.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")
    conn_options = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0)

    with pytest.raises(APIError) as exc_info:
        async with llm_model.chat(chat_ctx=chat_ctx, conn_options=conn_options) as stream:
            async for _ in stream:
                pass

    # Exhausted retries surface as an APIConnectionError caused by the last 429.
    error = exc_info.value
    status_error = error if isinstance(error, APIStatusError) else error.__cause__
    assert isinstance(status_error, APIStatusError)
    return status_error, attempts


@pytest.mark.asyncio
async def test_should_retry_false_is_not_retried() -> None:
    error, attempts = await _run_429({"X-Should-Retry": "false"})

    assert error.status_code == 429
    assert error.retryable is False
    assert len(attempts) == 1, "a terminal quota denial must not be retried"


@pytest.mark.asyncio
async def test_should_retry_header_is_case_insensitive() -> None:
    error, attempts = await _run_429({"x-should-retry": "FALSE"})

    assert error.retryable is False
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_429_without_should_retry_stays_retryable() -> None:
    error, attempts = await _run_429({})

    assert error.status_code == 429
    assert error.retryable is True
    assert len(attempts) == 3, "a transient 429 must exhaust the retries"


@pytest.mark.asyncio
async def test_should_retry_true_stays_retryable() -> None:
    error, attempts = await _run_429({"X-Should-Retry": "true"})

    assert error.retryable is True
    assert len(attempts) == 3
