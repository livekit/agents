"""Unit tests for shared LiveKit Inference request metadata."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

import livekit.agents.inference._utils as inference_utils
import livekit.agents.job as job_module
from livekit.agents.inference._utils import (
    HEADER_SESSION_ID,
    create_inference_request_id,
    extract_quota_usage,
    get_inference_headers,
)
from livekit.agents.inference.llm import LLMStream

pytestmark = pytest.mark.unit


def test_inference_session_id_is_omitted_without_job_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_job_context() -> None:
        raise RuntimeError("no job context")

    monkeypatch.setattr(job_module, "get_job_context", _no_job_context)

    assert HEADER_SESSION_ID not in get_inference_headers()


def test_inference_session_id_is_fresh_and_sdk_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = SimpleNamespace(sid="RM_test", isconnected=lambda: False)
    ctx = SimpleNamespace(
        job=SimpleNamespace(id="job_test", room=room),
        room=room,
        inference_headers={HEADER_SESSION_ID: "caller_value"},
    )
    suffixes = iter(("first", "second"))

    monkeypatch.setattr(job_module, "get_job_context", lambda: ctx)
    monkeypatch.setattr(
        inference_utils,
        "shortuuid",
        lambda prefix="": prefix + next(suffixes),
    )

    first = get_inference_headers()
    second = get_inference_headers()

    assert first[HEADER_SESSION_ID] == "inference_first"
    assert second[HEADER_SESSION_ID] == "inference_second"


def test_inference_request_id_links_to_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inference_utils,
        "shortuuid",
        lambda prefix="": prefix + "suffix",
    )

    assert create_inference_request_id("inference_parent", "tts") == "inference_parent_tts_suffix"
    assert (
        create_inference_request_id(None, "eot", fallback_prefix="turn_request_")
        == "turn_request_suffix"
    )


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
