"""Cartesia TTS plugin tests (API key redaction + aligned-transcript capability)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import APIConnectionError, APIStatusError, LanguageCode
from livekit.plugins.cartesia import tts as cartesia_tts

pytestmark = pytest.mark.plugin("cartesia")

SECRET_API_KEY = "cartesia-secret-api-key-do-not-log"


def _handshake_error_with_api_key(api_key: str) -> WSServerHandshakeError:
    url = URL("wss://api.cartesia.ai/tts/websocket")
    headers = CIMultiDict(
        {
            "Host": "api.cartesia.ai",
            "User-Agent": "LiveKit Agents Cartesia Plugin/test",
            "X-API-Key": api_key,
        }
    )
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


@pytest.mark.asyncio
async def test_connect_ws_redacts_api_key_from_handshake_error():
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)

    async def _raise(*_args, **_kwargs):
        raise _handshake_error_with_api_key(SECRET_API_KEY)

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    tts._session = session

    with pytest.raises(APIStatusError) as exc_info:
        await tts._connect_ws(timeout=5.0)

    err = exc_info.value
    assert err.status_code == 401
    assert SECRET_API_KEY not in repr(err)
    assert err.__cause__ is None


@pytest.mark.asyncio
async def test_connect_ws_generic_error_does_not_chain_cause():
    from livekit.plugins.cartesia import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    leaky = ConnectionError(f"wss://api.cartesia.ai/tts/websocket?api_key={SECRET_API_KEY}")

    async def _raise(*_args, **_kwargs):
        raise leaky

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    tts._session = session

    with pytest.raises(APIConnectionError) as exc_info:
        await tts._connect_ws(timeout=5.0)

    err = exc_info.value
    assert err.__cause__ is None
    assert str(err) == "ConnectionError"
    assert SECRET_API_KEY not in repr(err)
    assert SECRET_API_KEY not in str(err)


def test_aligned_transcript_enabled_for_supported_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is True


def test_aligned_transcript_disabled_for_unsupported_language() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is False


def test_aligned_transcript_disabled_when_word_timestamps_off() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=False)
    assert tts.capabilities.aligned_transcript is False


def test_aligned_transcript_allowed_for_preview_model_any_language() -> None:
    tts = cartesia_tts.TTS(
        api_key="test-key",
        model="sonic-preview",
        language="ja",
        word_timestamps=True,
    )
    assert tts.capabilities.aligned_transcript is True


def test_unsupported_config_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.cartesia"):
        tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)

    assert tts.capabilities.aligned_transcript is False
    assert any("does not support aligned transcript" in record.message for record in caplog.records)


def test_update_options_narrows_capability_when_language_becomes_unsupported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is True

    tts.update_options(language="ja")
    assert tts.capabilities.aligned_transcript is False


def test_update_options_restores_capability_when_language_becomes_supported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)
    assert tts.capabilities.aligned_transcript is False

    tts.update_options(language="fr")
    assert tts.capabilities.aligned_transcript is True


def test_update_options_to_preview_model_enables_capability() -> None:
    tts = cartesia_tts.TTS(
        api_key="test-key",
        model="sonic-3",
        language="ja",
        word_timestamps=True,
    )
    assert tts.capabilities.aligned_transcript is False

    tts.update_options(model="sonic-preview")
    assert tts.capabilities.aligned_transcript is True


def test_add_timestamps_omitted_when_unsupported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="ja", word_timestamps=True)
    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is False


def test_add_timestamps_requested_when_supported() -> None:
    tts = cartesia_tts.TTS(api_key="test-key", language="en", word_timestamps=True)
    options = cartesia_tts._to_cartesia_options(tts._opts, streaming=True)
    assert options["add_timestamps"] is True


def test_supports_word_timestamps_helper() -> None:
    assert cartesia_tts._supports_word_timestamps("sonic-3", LanguageCode("en")) is True
    assert cartesia_tts._supports_word_timestamps("sonic-3", LanguageCode("ja")) is False
    assert cartesia_tts._supports_word_timestamps("sonic-preview", LanguageCode("ja")) is True
    assert cartesia_tts._supports_word_timestamps("sonic-3", None) is True
