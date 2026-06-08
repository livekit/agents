"""Tests for typed API errors, in particular the inference quota-exceeded error.

Covers:
- APIQuotaExceededError field extraction / retryable semantics
- APIQuotaExceededError.from_response detection
- create_api_error_from_http returning the typed error for quota bodies
- the inference LLM plugin raising APIQuotaExceededError on a 429 quota body
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from livekit.agents import (
    APIQuotaExceededError,
    APIStatusError,
    create_api_error_from_http,
    inference,
)
from livekit.agents._exceptions import INFERENCE_QUOTA_EXCEEDED_TYPE
from livekit.agents.llm import ChatContext

pytestmark = pytest.mark.unit


def _quota_body(hint: str | None = "Wait for the next billing cycle or upgrade your plan.") -> dict:
    return {
        "type": INFERENCE_QUOTA_EXCEEDED_TYPE,
        "error": "LLM token credit quota exceeded, category: MaxGatewayCredits, remaining_limit: 0",
        "hint": hint,
        "quota_type": "llm",
        "category": "MaxGatewayCredits",
        "remaining_limit": "0",
        "free_tier": "false",
        "documentation_url": "https://livekit.com/pricing",
    }


def test_quota_error_extracts_fields_from_body() -> None:
    err = APIQuotaExceededError("quota exceeded", status_code=429, body=_quota_body())

    assert isinstance(err, APIStatusError)
    assert err.status_code == 429
    assert err.quota_type == "llm"
    assert err.category == "MaxGatewayCredits"
    assert err.hint == "Wait for the next billing cycle or upgrade your plan."
    assert err.remaining_limit == "0"


def test_quota_error_is_not_retryable_by_default() -> None:
    # quota exhaustion will fail identically on an immediate retry
    err = APIQuotaExceededError("quota exceeded", status_code=429, body=_quota_body())
    assert err.retryable is False


def test_quota_error_explicit_fields_take_precedence_over_body() -> None:
    err = APIQuotaExceededError(
        "quota exceeded",
        status_code=429,
        body=_quota_body(),
        quota_type="tts",
        hint="custom hint",
    )
    assert err.quota_type == "tts"
    assert err.hint == "custom hint"
    # untouched fields still come from the body
    assert err.category == "MaxGatewayCredits"


def test_quota_error_missing_body_fields_are_none() -> None:
    err = APIQuotaExceededError("quota exceeded", status_code=429, body={"type": "x"})
    assert err.quota_type is None
    assert err.category is None
    assert err.hint is None
    assert err.remaining_limit is None


def test_from_response_detects_quota_body() -> None:
    err = APIQuotaExceededError.from_response("quota exceeded", status_code=429, body=_quota_body())
    assert isinstance(err, APIQuotaExceededError)
    assert err.quota_type == "llm"


@pytest.mark.parametrize(
    "body",
    [
        None,
        "rate limited",  # non-dict body (e.g. plain text)
        {"type": "something_else"},
        {"error": "no type field"},
    ],
)
def test_from_response_returns_none_for_non_quota_body(body: object) -> None:
    assert APIQuotaExceededError.from_response("msg", status_code=429, body=body) is None


def test_create_api_error_from_http_returns_typed_quota_error() -> None:
    err = create_api_error_from_http("quota exceeded", status=429, body=_quota_body())
    assert isinstance(err, APIQuotaExceededError)
    assert err.hint == "Wait for the next billing cycle or upgrade your plan."


def test_create_api_error_from_http_returns_plain_status_error() -> None:
    err = create_api_error_from_http("boom", status=429, body={"type": "rate_limit"})
    assert isinstance(err, APIStatusError)
    assert not isinstance(err, APIQuotaExceededError)


async def test_inference_llm_raises_quota_error_on_429() -> None:
    """The inference LLM plugin surfaces a 429 inference_quota_exceeded body as a
    typed, non-retryable APIQuotaExceededError carrying the gateway hint."""
    inference_llm = inference.LLM("gpt-4o", api_key="devkey", api_secret="s" * 32)

    response = httpx.Response(429, request=httpx.Request("POST", "http://x/v1/chat/completions"))
    status_error = openai.APIStatusError(
        "LLM token credit quota exceeded", response=response, body=_quota_body()
    )

    # replace the client before chat() so the background stream task uses the mock
    real_client = inference_llm._client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=status_error)
    inference_llm._client = mock_client

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    stream = inference_llm.chat(chat_ctx=chat_ctx)

    with pytest.raises(APIQuotaExceededError) as exc_info:
        async for _ in stream:
            pass

    err = exc_info.value
    assert err.status_code == 429
    assert err.quota_type == "llm"
    assert err.hint == "Wait for the next billing cycle or upgrade your plan."
    assert err.retryable is False

    # quota errors are terminal: no retries, the endpoint is hit exactly once
    assert mock_client.chat.completions.create.await_count == 1

    await stream.aclose()
    await real_client.close()
