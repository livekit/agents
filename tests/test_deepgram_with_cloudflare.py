from __future__ import annotations

import pytest

from livekit.plugins import deepgram

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_TOKEN", raising=False)


def test_stt_builds_gateway_url_and_auth() -> None:
    stt = deepgram.STT.with_cloudflare(account_id="acct", cf_aig_token="cf-tok")
    assert stt._opts.endpoint_url == "https://gateway.ai.cloudflare.com/v1/acct/default/workers-ai"
    assert stt._connect_headers == {"cf-aig-authorization": "cf-tok"}
    assert stt._opts.model == "@cf/deepgram/nova-3"
    assert stt.capabilities.streaming is True


def test_stt_account_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    stt = deepgram.STT.with_cloudflare(cf_aig_token="t")
    assert "/v1/env-acct/default/workers-ai" in stt._opts.endpoint_url


def test_stt_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_TOKEN", "env-tok")
    stt = deepgram.STT.with_cloudflare(account_id="acct")
    assert stt._connect_headers == {"cf-aig-authorization": "env-tok"}


def test_stt_base_url_override() -> None:
    stt = deepgram.STT.with_cloudflare(
        account_id="ignored", base_url="https://example.com/ws", cf_aig_token="t"
    )
    assert stt._opts.endpoint_url == "https://example.com/ws"


def test_stt_missing_account_id_raises() -> None:
    with pytest.raises(ValueError, match=r"account_id"):
        deepgram.STT.with_cloudflare(cf_aig_token="t")


def test_stt_missing_token_raises() -> None:
    with pytest.raises(ValueError, match=r"[Tt]oken"):
        deepgram.STT.with_cloudflare(account_id="acct")


def test_tts_builds_gateway_url_and_auth() -> None:
    tts = deepgram.TTS.with_cloudflare(account_id="acct", cf_aig_token="cf-tok")
    assert tts._opts.base_url == "https://gateway.ai.cloudflare.com/v1/acct/default/workers-ai"
    assert tts._connect_headers == {"cf-aig-authorization": "cf-tok"}
    assert tts._opts.model == "@cf/deepgram/aura-1"
    assert tts.capabilities.streaming is True


def test_tts_account_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    tts = deepgram.TTS.with_cloudflare(cf_aig_token="t")
    assert "/v1/env-acct/default/workers-ai" in tts._opts.base_url


def test_tts_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_TOKEN", "env-tok")
    tts = deepgram.TTS.with_cloudflare(account_id="acct")
    assert tts._connect_headers == {"cf-aig-authorization": "env-tok"}


def test_tts_base_url_override() -> None:
    tts = deepgram.TTS.with_cloudflare(
        account_id="ignored", base_url="https://example.com/ws", cf_aig_token="t"
    )
    assert tts._opts.base_url == "https://example.com/ws"


def test_tts_missing_account_id_raises() -> None:
    with pytest.raises(ValueError, match=r"account_id"):
        deepgram.TTS.with_cloudflare(cf_aig_token="t")


def test_tts_missing_token_raises() -> None:
    with pytest.raises(ValueError, match=r"[Tt]oken"):
        deepgram.TTS.with_cloudflare(account_id="acct")
