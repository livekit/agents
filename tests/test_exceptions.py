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
    APIConnectionError,
    APIQuotaExceededError,
    APIStatusError,
    create_api_error_from_http,
    inference,
)
from livekit.agents._exceptions import INFERENCE_QUOTA_EXCEEDED_TYPE
from livekit.agents.llm import ChatContext
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit


def _quota_body(hint: str | None = "Wait for the next billing cycle or upgrade your plan.") -> dict:
    """A credit-exhaustion (terminal) quota body."""
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


def _rate_limit_body() -> dict:
    """A per-minute rate-limit (transient) quota body — same `type`, different category."""
    return {
        "type": INFERENCE_QUOTA_EXCEEDED_TYPE,
        "error": "LLM request/min rate limit exceeded",
        "hint": "LLM request rate limit reached. Reduce request rate or upgrade your plan.",
        "quota_type": "llm",
        "category": "MaxConcurrentGatewayLLMRpm",
        "remaining_limit": "0",
    }


def test_quota_error_extracts_fields_from_body() -> None:
    err = APIQuotaExceededError("quota exceeded", status_code=429, body=_quota_body())

    assert isinstance(err, APIStatusError)
    assert err.status_code == 429
    assert err.quota_type == "llm"
    assert err.category == "MaxGatewayCredits"
    assert err.hint == "Wait for the next billing cycle or upgrade your plan."
    assert err.remaining_limit == "0"


def test_credit_quota_error_is_terminal_and_not_retryable() -> None:
    # credit exhaustion will fail identically on an immediate retry and every turn
    err = APIQuotaExceededError("quota exceeded", status_code=429, body=_quota_body())
    assert err.retryable is False
    assert err.terminal is True


def test_rate_limit_quota_error_is_retryable_and_not_terminal() -> None:
    # a per-minute rate limit recovers via backoff: keep it retryable + non-terminal so
    # it isn't misclassified as "out of credits" and doesn't kill the session on turn 1
    err = APIQuotaExceededError("rate limited", status_code=429, body=_rate_limit_body())
    assert err.category == "MaxConcurrentGatewayLLMRpm"
    assert err.retryable is True
    assert err.terminal is False


def test_unknown_quota_category_defaults_to_transient() -> None:
    # an unknown/missing category is treated as transient (no regression: it falls
    # through the existing tolerance instead of killing the session immediately)
    body = {**_quota_body(), "category": "SomeFutureCategory"}
    err = APIQuotaExceededError("quota exceeded", status_code=429, body=body)
    assert err.terminal is False
    assert err.retryable is True


def test_quota_error_explicit_terminal_and_retryable_override() -> None:
    err = APIQuotaExceededError(
        "rate limited", status_code=429, body=_rate_limit_body(), terminal=True, retryable=False
    )
    assert err.terminal is True
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


def test_quota_error_non_str_body_fields_are_dropped() -> None:
    # wire data from a user-pointable endpoint: non-str values must not crash the
    # category check (unhashable types) or leak into the `str | None` fields
    body = {
        "type": INFERENCE_QUOTA_EXCEEDED_TYPE,
        "quota_type": 3,
        "category": ["MaxGatewayCredits"],  # unhashable without coercion
        "hint": {"text": "x"},
        "remaining_limit": 0,
    }
    err = APIQuotaExceededError("quota exceeded", status_code=429, body=body)
    assert err.quota_type is None
    assert err.category is None
    assert err.hint is None
    assert err.remaining_limit is None
    # an invalid category cannot be credit exhaustion -> stays transient
    assert err.terminal is False
    assert err.retryable is True


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


def _inference_llm_raising(body: dict) -> tuple[inference.LLM, MagicMock, openai.AsyncClient]:
    """Build an inference LLM whose client raises a 429 with ``body``.

    The exception is constructed through the openai SDK's own response handling
    (``_make_status_error_from_response``) so tests exercise the body shape real
    gateway traffic produces: the SDK narrows a mapping body to its ``error`` value,
    a bare string for gateway payloads, so the plugin must re-parse the response.

    Returns (llm, mock_client, real_client). Swap the client *before* chat() so the
    background stream task uses the mock. Caller must close the real client.
    """
    inference_llm = inference.LLM("gpt-4o", api_key="devkey", api_secret="s" * 32)
    real_client = inference_llm._client

    response = httpx.Response(
        429, json=body, request=httpx.Request("POST", "http://x/v1/chat/completions")
    )
    status_error = real_client._make_status_error_from_response(response)
    # the SDK unwrapped body["error"] — keep this pinned so the fixture stays honest
    assert not isinstance(status_error.body, dict)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=status_error)
    inference_llm._client = mock_client
    return inference_llm, mock_client, real_client


async def test_inference_llm_raises_quota_error_on_429() -> None:
    """The inference LLM plugin surfaces a 429 credit-quota body as a typed,
    terminal, non-retryable APIQuotaExceededError carrying the gateway hint."""
    inference_llm, mock_client, real_client = _inference_llm_raising(_quota_body())

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
    assert err.terminal is True

    # credit exhaustion is terminal: no retries, the endpoint is hit exactly once
    assert mock_client.chat.completions.create.await_count == 1

    await stream.aclose()
    await real_client.close()


async def test_inference_llm_retries_rate_limit_429() -> None:
    """A 429 rate-limit body is retryable, so the stream retries it with backoff
    instead of giving up on the first attempt (contrast with credit exhaustion)."""
    inference_llm, mock_client, real_client = _inference_llm_raising(_rate_limit_body())

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    max_retry = 2
    stream = inference_llm.chat(
        chat_ctx=chat_ctx,
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0),
    )

    # after exhausting retries the stream raises APIConnectionError, not the 429
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass

    assert mock_client.chat.completions.create.await_count == max_retry + 1

    await stream.aclose()
    await real_client.close()
