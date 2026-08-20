from __future__ import annotations

from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import pytest

from livekit.plugins.inworld.realtime.realtime_model import (
    DEFAULT_LLM_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    DEFAULT_WS_URL,
    RealtimeModel,
    RealtimeSession,
    _build_ws_url,
)

pytestmark = pytest.mark.unit

# any base64-ish string; the plugin sends it verbatim as `Basic <key>`
_API_KEY = "dGVzdC1rZXk="


# -- URL building --


def test_build_ws_url_adds_required_query_params() -> None:
    url = _build_ws_url("wss://api.inworld.ai/api/v1/realtime/session")
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.inworld.ai"
    assert parsed.path == "/api/v1/realtime/session"
    assert query["protocol"] == ["realtime"]
    assert query["key"]  # a session id is generated


def test_build_ws_url_converts_http_to_ws() -> None:
    assert _build_ws_url("https://host/realtime").startswith("wss://host/realtime")
    assert _build_ws_url("http://host/realtime").startswith("ws://host/realtime")


def test_build_ws_url_preserves_explicit_key() -> None:
    url = _build_ws_url("wss://host/realtime?key=my-session")
    assert parse_qs(urlparse(url).query)["key"] == ["my-session"]


# -- Model init --


def test_defaults() -> None:
    model = RealtimeModel(api_key=_API_KEY)
    assert model._opts.model == DEFAULT_LLM_MODEL
    assert model._opts.voice == DEFAULT_VOICE
    assert model._tts_model == DEFAULT_TTS_MODEL
    assert model._opts.input_audio_transcription.model == DEFAULT_STT_MODEL
    assert model._opts.base_url == DEFAULT_WS_URL
    assert urlparse(model._opts.base_url).netloc == "api.inworld.ai"
    assert model._provider_data == {"auto_tool_response": False}
    assert not model.capabilities.auto_tool_reply_generation
    assert model.provider == "Inworld"
    assert model._provider_label == "Inworld Realtime API"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INWORLD_API_KEY", _API_KEY)
    assert RealtimeModel()._opts.api_key == _API_KEY


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)
    with pytest.raises(ValueError):
        RealtimeModel()


# -- Session update mapping --


def _session_update_payload(model: RealtimeModel) -> dict:
    # build a session without going through __init__ (no event loop / network needed)
    session = RealtimeSession.__new__(RealtimeSession)
    session._realtime_model = model  # type: ignore[attr-defined]
    session._opts = replace(model._opts)  # type: ignore[attr-defined]
    session._instructions = None  # type: ignore[attr-defined]

    event = session._create_session_update_event()
    data = event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)  # type: ignore[union-attr]
    return data["session"]


def test_session_update_maps_llm_tts_stt() -> None:
    model = RealtimeModel(api_key=_API_KEY, model="inworld/auto", voice="Clive")
    sess = _session_update_payload(model)

    assert sess["model"] == "inworld/auto"
    assert sess["audio"]["output"]["voice"] == "Clive"
    assert sess["audio"]["output"]["model"] == DEFAULT_TTS_MODEL
    assert sess["audio"]["input"]["transcription"]["model"] == DEFAULT_STT_MODEL


def test_session_update_includes_provider_data() -> None:
    provider_data = {
        "auto_tool_response": False,
        "stt": {"voice_profile": True},
        "memory": {"enabled": True},
    }
    model = RealtimeModel(api_key=_API_KEY, provider_data=provider_data)
    assert _session_update_payload(model)["providerData"] == provider_data


def test_session_update_preserves_nested_provider_data_branches() -> None:
    provider_data = {
        "stt": {"voice_profile": True, "language_hints": ["en-US"]},
        "tts": {"segmenter_strategy": "sentence", "delivery_mode": "CREATIVE"},
        "text_generation_config": {"reasoning": {"effort": "LOW"}},
        "user_id": "user_abc",
    }
    model = RealtimeModel(api_key=_API_KEY, provider_data=provider_data)
    pd = _session_update_payload(model)["providerData"]

    assert pd["stt"]["language_hints"] == ["en-US"]
    assert pd["tts"]["segmenter_strategy"] == "sentence"
    assert pd["text_generation_config"]["reasoning"]["effort"] == "LOW"
    assert pd["user_id"] == "user_abc"


def test_session_update_disables_automatic_tool_responses_by_default() -> None:
    provider_data = _session_update_payload(RealtimeModel(api_key=_API_KEY))["providerData"]
    assert provider_data == {"auto_tool_response": False}


def test_session_update_allows_automatic_tool_response_override() -> None:
    model = RealtimeModel(api_key=_API_KEY, provider_data={"auto_tool_response": True})
    assert _session_update_payload(model)["providerData"]["auto_tool_response"] is True
