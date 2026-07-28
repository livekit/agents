from __future__ import annotations

import pytest

from livekit.plugins import deepgram

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)


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


def test_stt_model_accepts_bare_and_prefixed_names() -> None:
    assert deepgram.STT.with_cloudflare(account_id="a", cf_aig_token="t")._opts.model == (
        "@cf/deepgram/nova-3"
    )
    assert (
        deepgram.STT.with_cloudflare(
            account_id="a", cf_aig_token="t", model="nova-2-general"
        )._opts.model
        == "@cf/deepgram/nova-2-general"
    )
    # an already-prefixed value is passed through unchanged
    assert (
        deepgram.STT.with_cloudflare(
            account_id="a", cf_aig_token="t", model="@cf/deepgram/nova-3"
        )._opts.model
        == "@cf/deepgram/nova-3"
    )


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


# --- extra_headers seam (default Deepgram behavior preserved; additive; sole auth) ---


def test_default_token_auth_unchanged() -> None:
    assert deepgram.STT(api_key="k")._connect_headers == {"Authorization": "Token k"}
    assert deepgram.TTS(api_key="k")._connect_headers == {"Authorization": "Token k"}


def test_extra_headers_merge_over_token_auth() -> None:
    stt = deepgram.STT(api_key="k", extra_headers={"X-Trace": "1"})
    assert stt._connect_headers == {"Authorization": "Token k", "X-Trace": "1"}
    tts = deepgram.TTS(api_key="k", extra_headers={"X-Trace": "1"})
    assert tts._connect_headers == {"Authorization": "Token k", "X-Trace": "1"}


def test_extra_headers_are_sole_auth_without_key() -> None:
    # no Deepgram key -> no default Authorization header; extra_headers carry auth
    stt = deepgram.STT(extra_headers={"cf-aig-authorization": "Bearer x"})
    assert stt._connect_headers == {"cf-aig-authorization": "Bearer x"}
    tts = deepgram.TTS(extra_headers={"cf-aig-authorization": "Bearer x"})
    assert tts._connect_headers == {"cf-aig-authorization": "Bearer x"}


def test_no_key_and_no_extra_headers_raises() -> None:
    with pytest.raises(ValueError):
        deepgram.STT()
    with pytest.raises(ValueError):
        deepgram.TTS()


def test_with_cloudflare_ignores_deepgram_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # a DEEPGRAM_API_KEY in the environment must not leak into the gateway request
    monkeypatch.setenv("DEEPGRAM_API_KEY", "leaked-key")
    stt = deepgram.STT.with_cloudflare(account_id="a", cf_aig_token="cf")
    assert stt._connect_headers == {"cf-aig-authorization": "cf"}
    tts = deepgram.TTS.with_cloudflare(account_id="a", cf_aig_token="cf")
    assert tts._connect_headers == {"cf-aig-authorization": "cf"}


def test_keyterm_allowed_for_cf_prefixed_nova3() -> None:
    # @cf/deepgram/nova-3 is still nova-3; keyterm must not be rejected by the routing prefix
    stt = deepgram.STT(
        model="@cf/deepgram/nova-3",
        extra_headers={"cf-aig-authorization": "cf"},
        keyterm=["livekit"],
    )
    assert stt._opts.model == "@cf/deepgram/nova-3"


def test_en_only_fallback_preserves_cf_prefix() -> None:
    # an en-only model with a non-English language falls back to nova-2-general,
    # but the @cf/deepgram/ routing prefix must be preserved
    stt = deepgram.STT(
        model="@cf/deepgram/nova-2-meeting",
        language="fr",
        extra_headers={"cf-aig-authorization": "cf"},
    )
    assert stt._opts.model == "@cf/deepgram/nova-2-general"
    # the bare (non-prefixed) path is unchanged
    bare = deepgram.STT(model="nova-2-meeting", language="fr", api_key="k")
    assert bare._opts.model == "nova-2-general"


# --- non-streaming paths are not proxied by the gateway; fail fast with a clear error ---


async def test_stt_with_cloudflare_recognize_raises() -> None:
    stt = deepgram.STT.with_cloudflare(account_id="a", cf_aig_token="t")
    with pytest.raises(NotImplementedError, match="streaming"):
        await stt.recognize([])


def test_tts_with_cloudflare_synthesize_raises() -> None:
    tts = deepgram.TTS.with_cloudflare(account_id="a", cf_aig_token="t")
    with pytest.raises(NotImplementedError, match="streaming"):
        tts.synthesize("hello")


# --- STTv2 (Flux) ---


def test_stt_v2_builds_gateway_url_and_auth() -> None:
    stt = deepgram.STTv2.with_cloudflare(account_id="acct", cf_aig_token="cf-tok")
    assert stt._opts.endpoint_url == "https://gateway.ai.cloudflare.com/v1/acct/default/workers-ai"
    assert stt._connect_headers == {"cf-aig-authorization": "cf-tok"}
    assert stt._opts.model == "@cf/deepgram/flux"
    assert stt.capabilities.streaming is True


def test_stt_v2_account_id_and_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_TOKEN", "env-tok")
    stt = deepgram.STTv2.with_cloudflare()
    assert "/v1/env-acct/default/workers-ai" in stt._opts.endpoint_url
    assert stt._connect_headers == {"cf-aig-authorization": "env-tok"}


def test_stt_v2_model_accepts_bare_and_prefixed_names() -> None:
    # Cloudflare's catalog has a single flux model: @cf/deepgram/flux
    assert (
        deepgram.STTv2.with_cloudflare(account_id="a", cf_aig_token="t", model="flux")._opts.model
        == "@cf/deepgram/flux"
    )
    assert (
        deepgram.STTv2.with_cloudflare(
            account_id="a", cf_aig_token="t", model="@cf/deepgram/flux"
        )._opts.model
        == "@cf/deepgram/flux"
    )


def test_stt_v2_base_url_override() -> None:
    stt = deepgram.STTv2.with_cloudflare(
        account_id="ignored", base_url="https://example.com/ws", cf_aig_token="t"
    )
    assert stt._opts.endpoint_url == "https://example.com/ws"


def test_stt_v2_missing_account_id_raises() -> None:
    with pytest.raises(ValueError, match=r"account_id"):
        deepgram.STTv2.with_cloudflare(cf_aig_token="t")


def test_stt_v2_missing_token_raises() -> None:
    with pytest.raises(ValueError, match=r"[Tt]oken"):
        deepgram.STTv2.with_cloudflare(account_id="acct")


def test_stt_v2_forwards_flux_options() -> None:
    stt = deepgram.STTv2.with_cloudflare(
        account_id="a",
        cf_aig_token="t",
        eager_eot_threshold=0.5,
        eot_threshold=0.8,
        eot_timeout_ms=4000,
        keyterm=["livekit"],
    )
    assert stt._opts.eager_eot_threshold == 0.5
    assert stt._opts.eot_threshold == 0.8
    assert stt._opts.eot_timeout_ms == 4000
    assert stt._opts.keyterm == ["livekit"]


def test_stt_v2_default_token_auth_unchanged() -> None:
    assert deepgram.STTv2(api_key="k")._connect_headers == {"Authorization": "Token k"}


def test_stt_v2_extra_headers_merge_over_token_auth() -> None:
    stt = deepgram.STTv2(api_key="k", extra_headers={"X-Trace": "1"})
    assert stt._connect_headers == {"Authorization": "Token k", "X-Trace": "1"}


def test_stt_v2_extra_headers_are_sole_auth_without_key() -> None:
    stt = deepgram.STTv2(extra_headers={"cf-aig-authorization": "x"})
    assert stt._connect_headers == {"cf-aig-authorization": "x"}


def test_stt_v2_no_key_and_no_extra_headers_raises() -> None:
    with pytest.raises(ValueError):
        deepgram.STTv2()


def test_stt_v2_with_cloudflare_ignores_deepgram_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "leaked-key")
    stt = deepgram.STTv2.with_cloudflare(account_id="a", cf_aig_token="cf")
    assert stt._connect_headers == {"cf-aig-authorization": "cf"}


def test_stt_v2_language_hint_warning_sees_through_cf_prefix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    # a prefixed multi model is still flux-general-multi; language_hint must not warn
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.deepgram"):
        deepgram.STTv2(
            model="@cf/deepgram/flux-general-multi",
            language_hint=["fr"],
            extra_headers={"cf-aig-authorization": "cf"},
        )
    assert not any("language_hint" in r.message for r in caplog.records)

    # the bare non-multi path still warns
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.deepgram"):
        deepgram.STTv2(model="flux-general-en", language_hint=["fr"], api_key="k")
    assert any("language_hint" in r.message for r in caplog.records)
