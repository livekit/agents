"""Unit tests for the Avaz TTS plugin."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.unit

_TEST_WS = "ws://127.0.0.1:8893/tts/stream-input"
_TEST_UUID = "15658888-374f-4739-a0c5-4f1d1c128d2a"


def test_tts_init_with_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.avaz import TTS

    monkeypatch.delenv("AVAZ_AGENT_MODEL_ID", raising=False)
    monkeypatch.delenv("AVAZ_BASE_URL", raising=False)
    monkeypatch.delenv("AVAZ_API_KEY", raising=False)
    engine = TTS(ws_url=_TEST_WS)
    assert engine.provider == "avaz"
    assert engine.model == "avaz3"
    assert engine._opts.ws_url == _TEST_WS


def test_tts_base_url_derives_wss(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.avaz import TTS

    monkeypatch.delenv("AVAZ_AGENT_MODEL_ID", raising=False)
    engine = TTS(
        api_key="test-api-key",
        base_url="https://test.example.com/api",
        model_id=_TEST_UUID,
    )
    assert engine._opts.ws_url == "wss://test.example.com/api/tts/stream-input"
    assert engine.model == _TEST_UUID


def test_tts_base_url_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.avaz import TTS

    monkeypatch.delenv("AVAZ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        TTS(base_url="https://test.example.com/api")


def test_build_init_message_uses_stream_model() -> None:
    from livekit.plugins.avaz import TTS
    from livekit.plugins.avaz.tts import _build_init_message

    engine = TTS(
        api_key="test-api-key",
        base_url="https://test.example.com/api",
        model_id=_TEST_UUID,
        stream_model="avaz3",
    )
    msg = _build_init_message(engine._opts)
    assert msg["model_settings"]["model_id"] == "avaz3"
    assert json.dumps(msg)


def test_derive_ws_url_from_base() -> None:
    from livekit.plugins.avaz.tts import _derive_ws_url_from_base

    assert (
        _derive_ws_url_from_base("https://dashboard.example/api")
        == "wss://dashboard.example/api/tts/stream-input"
    )


def test_auth_headers() -> None:
    from livekit.plugins.avaz import build_auth_headers

    headers = build_auth_headers("test-api-key")
    assert headers["X-API-Key"] == "test-api-key"
    assert headers["Authorization"] == "Bearer test-api-key"


def test_set_voice_ids_uuid() -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS)
    engine.set_voice_ids(model_id=_TEST_UUID)
    assert engine._opts.agent_model_id == _TEST_UUID


def test_resolve_stream_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.avaz.tts import _resolve_stream_model

    monkeypatch.setenv("AVAZ_STREAM_MODEL", "avaz2")
    assert (
        _resolve_stream_model(
            stream_model=NOT_GIVEN,
            model_id=NOT_GIVEN,
            agent_model_id=_TEST_UUID,
        )
        == "avaz2"
    )


@pytest.mark.asyncio
async def test_warmup_passes_auth_headers() -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS, api_key="test-api-key")
    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)
    mock_ws.recv = AsyncMock(
        side_effect=[
            '{"status":"initialized"}',
            '{"audio":"' + "A" * 44 + '"}',
        ]
    )
    mock_ws.send = AsyncMock()

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws) as connect:
        await engine.warmup(timeout_s=5.0)
        connect.assert_called_once()
        _, kwargs = connect.call_args
        assert kwargs["additional_headers"]["X-API-Key"] == "test-api-key"


def test_parse_init_response_error() -> None:
    from livekit.plugins.avaz.tts import _parse_init_response

    with pytest.raises(APIConnectionError, match="init error"):
        _parse_init_response('{"error":"model not found"}')
