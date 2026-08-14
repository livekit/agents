"""Unit tests for LiveKit Inference quota telemetry."""

from __future__ import annotations

import logging

import httpx
import openai
import pytest

from livekit.agents.inference._utils import (
    extract_quota_usage,
)
from livekit.agents.inference.llm import LLMStream

pytestmark = pytest.mark.unit


def test_extract_quota_usage_returns_all_stamped_dimensions() -> None:
    """Every quota header the gateway stamps maps to its log-friendly field."""
    headers = {
        "X-LiveKit-Inference-RPM-Limit": "100",
        "X-LiveKit-Inference-RPM-Used": "101",
        "X-LiveKit-Inference-TPM-Limit": "50000",
        "X-LiveKit-Inference-TPM-Used": "48000",
        "X-LiveKit-Inference-Credits-Limit": "1000000",
        "X-LiveKit-Inference-Credits-Used": "999999",
    }

    usage = extract_quota_usage(headers)

    assert usage == {
        "rpm_limit": "100",
        "rpm_used": "101",
        "tpm_limit": "50000",
        "tpm_used": "48000",
        "credits_limit": "1000000",
        "credits_used": "999999",
    }


def test_extract_quota_usage_omits_missing_dimensions() -> None:
    """A dimension the gateway didn't stamp (not enforced) is left out entirely."""
    headers = {
        "X-LiveKit-Inference-RPM-Limit": "100",
        "X-LiveKit-Inference-RPM-Used": "101",
    }

    usage = extract_quota_usage(headers)

    assert usage == {"rpm_limit": "100", "rpm_used": "101"}


def test_extract_quota_usage_empty_when_no_quota_headers() -> None:
    """A response with no quota telemetry yields an empty dict."""
    assert extract_quota_usage({}) == {}
    assert extract_quota_usage({"Content-Type": "application/json"}) == {}


def test_extract_quota_usage_case_insensitive_with_httpx_headers() -> None:
    """HTTP/2 lowercases field names on the wire; httpx.Headers lookup still matches."""
    headers = httpx.Headers(
        {
            "x-livekit-inference-rpm-limit": "100",
            "x-livekit-inference-rpm-used": "42",
        }
    )

    usage = extract_quota_usage(headers)

    assert usage == {"rpm_limit": "100", "rpm_used": "42"}


def test_llm_stream_logs_quota_on_429(caplog: pytest.LogCaptureFixture) -> None:
    """A 429 from the gateway logs a warning carrying the quota snapshot."""
    response = httpx.Response(
        429,
        headers={
            "x-request-id": "req_123",
            "X-LiveKit-Inference-RPM-Limit": "100",
            "X-LiveKit-Inference-RPM-Used": "101",
        },
        request=httpx.Request("POST", "https://agent-gateway.livekit.cloud/v1/chat/completions"),
    )
    err = openai.APIStatusError("rate limited", response=response, body=None)

    # bypass __init__: it spawns the request task; _log_rate_limited only needs _model
    stream = LLMStream.__new__(LLMStream)
    stream._model = "openai/gpt-4o"

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        stream._log_rate_limited(err)

    record = next(
        r for r in caplog.records if r.message == "LLM request rate limited by inference gateway"
    )
    assert record.model == "openai/gpt-4o"
    assert record.request_id == "req_123"
    assert record.rpm_limit == "100"
    assert record.rpm_used == "101"
    assert not hasattr(record, "tpm_limit")


def test_llm_stream_logs_429_without_quota_headers(caplog: pytest.LogCaptureFixture) -> None:
    """A 429 with no quota telemetry still logs, just without quota fields."""
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://agent-gateway.livekit.cloud/v1/chat/completions"),
    )
    err = openai.APIStatusError("rate limited", response=response, body=None)

    stream = LLMStream.__new__(LLMStream)
    stream._model = "openai/gpt-4o"

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        stream._log_rate_limited(err)

    record = next(
        r for r in caplog.records if r.message == "LLM request rate limited by inference gateway"
    )
    assert record.model == "openai/gpt-4o"
    assert not hasattr(record, "rpm_limit")
    assert not hasattr(record, "rpm_used")
