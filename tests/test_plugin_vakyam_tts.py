"""Tests for Vakyam TTS plugin configuration and WebSocket protocol helpers."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from livekit.agents import APIStatusError

pytestmark = pytest.mark.unit


def test_tts_requires_api_key() -> None:
    from livekit.plugins.vakyam import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly() -> None:
    from livekit.plugins.vakyam import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env() -> None:
    from livekit.plugins.vakyam import TTS

    with patch.dict("os.environ", {"VAKYAM_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


def test_tts_defaults() -> None:
    from livekit.plugins.vakyam import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "raaga-v1"
    assert tts.provider == "Vakyam"
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1
    assert tts.capabilities.streaming is True
    assert tts._opts.voice == "Archana"
    assert tts._opts.language == "ta-IN"
    assert tts._opts.speed == 1.0


def test_tts_custom_options() -> None:
    from livekit.plugins.vakyam import TTS

    tts = TTS(
        api_key="test-key",
        voice="vc_01EXAMPLE",
        language="hi-IN",
        sample_rate=16000,
        speed=1.2,
    )
    assert tts._opts.voice == "vc_01EXAMPLE"
    assert tts._opts.language == "hi-IN"
    assert tts.sample_rate == 16000
    assert tts._opts.speed == 1.2


def test_tts_rejects_unsupported_language() -> None:
    from livekit.plugins.vakyam import TTS

    with pytest.raises(ValueError, match="language"):
        TTS(api_key="test-key", language="en-US")


def test_tts_rejects_unsupported_sample_rate() -> None:
    from livekit.plugins.vakyam import TTS

    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="test-key", sample_rate=22050)


def test_tts_rejects_invalid_speed() -> None:
    from livekit.plugins.vakyam import TTS

    with pytest.raises(ValueError, match="speed"):
        TTS(api_key="test-key", speed=3.0)


def test_tts_rejects_empty_voice() -> None:
    from livekit.plugins.vakyam import TTS

    with pytest.raises(ValueError, match="voice"):
        TTS(api_key="test-key", voice="   ")


def test_tts_rejects_bare_custom_voice_prefix() -> None:
    from livekit.plugins.vakyam import TTS

    with pytest.raises(ValueError, match="vc_"):
        TTS(api_key="test-key", voice="vc_")


def test_update_options() -> None:
    from livekit.plugins.vakyam import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(language="en-IN", voice="Archana", sample_rate=8000, speed=0.9)
    assert tts._opts.language == "en-IN"
    assert tts.sample_rate == 8000
    assert tts._opts.speed == 0.9
    assert tts._needs_reconnect is True


def test_websocket_url_from_https() -> None:
    from livekit.plugins.vakyam._utils import websocket_url

    assert websocket_url("https://api.vakyam.ai") == "wss://api.vakyam.ai/v1/tts/websocket"


def test_http_stream_url() -> None:
    from livekit.plugins.vakyam._utils import http_stream_url

    assert http_stream_url("https://api.vakyam.ai") == "https://api.vakyam.ai/v1/tts/stream"


def test_insecure_base_url_rejected() -> None:
    from livekit.plugins.vakyam._utils import normalize_base_url

    with pytest.raises(ValueError, match="HTTPS"):
        normalize_base_url("http://example.com")


def test_insecure_localhost_allowed() -> None:
    from livekit.plugins.vakyam._utils import normalize_base_url

    assert normalize_base_url("http://localhost:8080") == "http://localhost:8080"


def test_session_config_wire_message() -> None:
    from livekit.plugins.vakyam._websocket import TTSSessionConfig

    msg = TTSSessionConfig(
        model="raaga-v1",
        voice="Archana",
        language="ta-IN",
        sample_rate=24000,
        speed=1.0,
    ).to_wire_message()
    assert msg["type"] == "config"
    assert msg["model_id"] == "raaga-v1"
    assert msg["voice"] == "Archana"
    assert msg["output_format"] == "pcm"


def test_speech_payload() -> None:
    from livekit.plugins.vakyam._utils import speech_payload

    payload = speech_payload(
        text="வணக்கம்.",
        model="raaga-v1",
        voice="Archana",
        language="ta-IN",
        sample_rate=24000,
        speed=1.0,
    )
    assert payload["model_id"] == "raaga-v1"
    assert payload["output_format"] == "pcm"
    assert payload["text"] == "வணக்கம்."


def test_text_too_long() -> None:
    from livekit.plugins.vakyam._utils import validate_text

    with pytest.raises(ValueError, match="3000"):
        validate_text("a" * 3001)


@pytest.mark.asyncio
async def test_synthesize_stream_waits_for_end_of_utterance() -> None:
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession, TTSSessionConfig

    class FakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self._incoming: list[bytes | str] = [
                b"pcm-1",
                b"pcm-2",
                json.dumps(
                    {"type": "end_of_utterance", "characters_used": 4, "duration_seconds": 0.5}
                ),
            ]

        async def send(self, data: str) -> None:
            self.sent.append(data)

        async def recv(self) -> bytes | str:
            return self._incoming.pop(0)

        async def close(self) -> None:
            return None

    session = AsyncStreamingTTSSession(
        api_key="test-key",
        config=TTSSessionConfig(),
        allow_insecure_base_url=True,
        base_url="http://localhost",
    )
    fake = FakeWS()
    session._connection = fake

    chunks = [chunk async for chunk in session.synthesize_stream("Hello.")]
    assert chunks == [b"pcm-1", b"pcm-2"]
    assert json.loads(fake.sent[0]) == {"type": "text", "text": "Hello."}
    assert session.last_result is not None
    assert session.last_result.cancelled is False
    assert session.last_result.characters_used == 4
    assert session.utterance_active is False


@pytest.mark.asyncio
async def test_synthesize_stream_cancel_drains_until_cancellation() -> None:
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession, TTSSessionConfig

    class FakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self._incoming: list[bytes | str] = [
                b"pcm-1",
                json.dumps({"type": "cancellation", "characters_used": 2, "duration_seconds": 0.1}),
            ]

        async def send(self, data: str) -> None:
            self.sent.append(data)

        async def recv(self) -> bytes | str:
            return self._incoming.pop(0)

        async def close(self) -> None:
            return None

    session = AsyncStreamingTTSSession(
        api_key="test-key",
        config=TTSSessionConfig(),
        allow_insecure_base_url=True,
        base_url="http://localhost",
    )
    fake = FakeWS()
    session._connection = fake

    agen = session.synthesize_stream("Hello.")
    first = await agen.__anext__()
    assert first == b"pcm-1"
    with pytest.raises(asyncio.CancelledError):
        await agen.athrow(asyncio.CancelledError)

    assert session.last_result is not None
    assert session.last_result.cancelled is True
    sent_types = [json.loads(item)["type"] for item in fake.sent]
    assert "cancel" in sent_types


def test_raise_http_error_parses_envelope() -> None:
    from livekit.plugins.vakyam._utils import raise_http_error

    with pytest.raises(APIStatusError, match="Too many requests") as exc_info:
        raise_http_error(
            429,
            json.dumps({"error": {"code": "rate_limit_exceeded", "message": "Too many requests."}}),
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
