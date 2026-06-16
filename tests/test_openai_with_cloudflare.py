from __future__ import annotations

import json

import pytest

from livekit.plugins import openai

pytestmark = pytest.mark.unit

_BASE_URL = "https://gateway.ai.cloudflare.com/v1/acct/default/compat"


@pytest.fixture(autouse=True)
def _clear_cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # keep construction deterministic regardless of the host environment
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)


def test_builds_url_from_account_and_default_gateway() -> None:
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", account_id="acct", cf_aig_token="t")
    assert (
        str(llm._client.base_url).rstrip("/")
        == "https://gateway.ai.cloudflare.com/v1/acct/default/compat"
    )


def test_builds_url_with_custom_gateway_id() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o", account_id="acct", gateway_id="prod", cf_aig_token="t"
    )
    assert "/v1/acct/prod/compat" in str(llm._client.base_url)


def test_account_id_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    llm = openai.LLM.with_cloudflare(model="openai/gpt-4o", cf_aig_token="t")
    assert "/v1/env-acct/default/compat" in str(llm._client.base_url)


def test_base_url_overrides_account_id() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o", account_id="ignored", base_url=_BASE_URL, cf_aig_token="t"
    )
    assert str(llm._client.base_url).rstrip("/") == _BASE_URL


def test_missing_account_id_raises() -> None:
    with pytest.raises(ValueError):
        openai.LLM.with_cloudflare(model="openai/gpt-4o", cf_aig_token="t")


def test_byok_forwards_provider_key_and_base_url() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o", base_url=_BASE_URL, api_key="sk-provider"
    )
    assert str(llm._client.base_url).rstrip("/") == _BASE_URL
    assert llm._client.api_key == "sk-provider"
    # no gateway token -> no cf-aig-authorization header
    assert "cf-aig-authorization" not in (llm._opts.extra_headers or {})


def test_gateway_token_sets_header_and_placeholder_key() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o", base_url=_BASE_URL, cf_aig_token="cf-tok"
    )
    # no provider key -> SDK still needs a non-empty Authorization, so a placeholder is used
    assert llm._client.api_key == "cloudflare"
    assert llm._opts.extra_headers["cf-aig-authorization"] == "Bearer cf-tok"


def test_gateway_options_map_to_headers() -> None:
    llm = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        base_url=_BASE_URL,
        cf_aig_token="cf-tok",
        gateway_options={
            "cache_ttl": 3600,
            "cache_key": "k1",
            "request_timeout": 2000,
            "max_attempts": 3,
            "retry_delay": 500,
            "backoff": "exponential",
            "metadata": {"room": "r1", "turn": 4, "live": True},
            "custom_cost": {"per_token_in": 0.000001, "per_token_out": 0.000002},
        },
    )
    headers = llm._opts.extra_headers
    assert headers["cf-aig-cache-ttl"] == "3600"
    assert headers["cf-aig-cache-key"] == "k1"
    assert headers["cf-aig-request-timeout"] == "2000"
    assert headers["cf-aig-max-attempts"] == "3"
    assert headers["cf-aig-retry-delay"] == "500"
    assert headers["cf-aig-backoff"] == "exponential"
    # metadata and custom_cost are JSON-encoded
    assert json.loads(headers["cf-aig-metadata"]) == {"room": "r1", "turn": 4, "live": True}
    assert json.loads(headers["cf-aig-custom-cost"]) == {
        "per_token_in": 0.000001,
        "per_token_out": 0.000002,
    }


def test_skip_cache_header_only_emitted_when_true() -> None:
    enabled = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        base_url=_BASE_URL,
        cf_aig_token="t",
        gateway_options={"skip_cache": True},
    )
    assert enabled._opts.extra_headers["cf-aig-skip-cache"] == "true"

    disabled = openai.LLM.with_cloudflare(
        model="openai/gpt-4o",
        base_url=_BASE_URL,
        cf_aig_token="t",
        gateway_options={"skip_cache": False},
    )
    assert "cf-aig-skip-cache" not in disabled._opts.extra_headers


def test_invalid_base_url_raises() -> None:
    with pytest.raises(ValueError):
        openai.LLM.with_cloudflare(model="openai/gpt-4o", base_url="not-a-url", cf_aig_token="t")


def test_missing_auth_raises() -> None:
    with pytest.raises(ValueError):
        openai.LLM.with_cloudflare(model="openai/gpt-4o", base_url=_BASE_URL)
