from __future__ import annotations

import json

import pytest

from livekit.plugins import openai

pytestmark = pytest.mark.unit

_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1"


@pytest.fixture(autouse=True)
def _clear_cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # keep construction deterministic regardless of the host environment
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)


def test_builds_rest_api_url_from_account() -> None:
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", account_id="acct", api_key="cf-tok")
    assert str(llm._client.base_url).rstrip("/") == _BASE_URL


def test_account_id_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", api_key="cf-tok")
    assert "/accounts/env-acct/ai/v1" in str(llm._client.base_url)


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "env-tok")
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", account_id="acct")
    assert llm._client.api_key == "env-tok"


def test_token_sent_as_bearer_authorization() -> None:
    # the Cloudflare API token is the OpenAI client's api_key -> Authorization: Bearer <token>
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", account_id="acct", api_key="cf-tok")
    assert llm._client.api_key == "cf-tok"
    # the deprecated cf-aig-authorization header is no longer used
    assert "cf-aig-authorization" not in (llm._opts.extra_headers or {})


def test_base_url_overrides_account_id() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o", account_id="ignored", base_url=_BASE_URL, api_key="cf-tok"
    )
    assert str(llm._client.base_url).rstrip("/") == _BASE_URL


def test_gateway_id_sets_header() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o", account_id="acct", api_key="cf-tok", gateway_id="prod"
    )
    assert llm._opts.extra_headers["cf-aig-gateway-id"] == "prod"


def test_no_gateway_id_omits_header() -> None:
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", account_id="acct", api_key="cf-tok")
    assert "cf-aig-gateway-id" not in (llm._opts.extra_headers or {})


def test_gateway_options_map_to_headers() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        account_id="acct",
        api_key="cf-tok",
        gateway_options={
            "cache_ttl": 3600,
            "cache_key": "k1",
            "request_timeout": 2000,
            "max_attempts": 3,
            "retry_delay": 500,
            "backoff": "exponential",
            "metadata": {"room": "r1", "turn": 4, "live": True},
        },
    )
    headers = llm._opts.extra_headers
    assert headers["cf-aig-cache-ttl"] == "3600"
    assert headers["cf-aig-cache-key"] == "k1"
    assert headers["cf-aig-request-timeout"] == "2000"
    assert headers["cf-aig-max-attempts"] == "3"
    assert headers["cf-aig-retry-delay"] == "500"
    assert headers["cf-aig-backoff"] == "exponential"
    assert json.loads(headers["cf-aig-metadata"]) == {"room": "r1", "turn": 4, "live": True}


def test_metadata_accepts_json_string() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        account_id="acct",
        api_key="cf-tok",
        gateway_options={"metadata": '{"room":"r1"}'},
    )
    # a pre-serialized JSON string is passed through unchanged
    assert llm._opts.extra_headers["cf-aig-metadata"] == '{"room":"r1"}'


def test_collect_log_emitted_true_or_false() -> None:
    on = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        account_id="acct",
        api_key="cf-tok",
        gateway_options={"collect_log": True},
    )
    assert on._opts.extra_headers["cf-aig-collect-log"] == "true"

    off = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        account_id="acct",
        api_key="cf-tok",
        gateway_options={"collect_log": False},
    )
    assert off._opts.extra_headers["cf-aig-collect-log"] == "false"


def test_skip_cache_header_only_emitted_when_true() -> None:
    enabled = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        account_id="acct",
        api_key="cf-tok",
        gateway_options={"skip_cache": True},
    )
    assert enabled._opts.extra_headers["cf-aig-skip-cache"] == "true"

    disabled = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        account_id="acct",
        api_key="cf-tok",
        gateway_options={"skip_cache": False},
    )
    assert "cf-aig-skip-cache" not in disabled._opts.extra_headers


def test_invalid_base_url_raises() -> None:
    with pytest.raises(ValueError):
        openai.LLM.with_cloudflare(model="openai/gpt-4o", base_url="not-a-url", api_key="cf-tok")


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match=r"API token"):
        openai.LLM.with_cloudflare(model="openai/gpt-4o", account_id="acct")


def test_missing_account_id_raises() -> None:
    with pytest.raises(ValueError, match=r"account_id"):
        openai.LLM.with_cloudflare(model="openai/gpt-4o", api_key="cf-tok")
