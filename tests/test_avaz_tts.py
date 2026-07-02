"""Unit tests for the Avaz TTS plugin."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.unit

_TEST_WS = "ws://127.0.0.1:8893/tts/stream-input"
_TEST_UUID = "15658888-374f-4739-a0c5-4f1d1c128d2a"


def _minimal_wav_b64() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes(b"\x00" * 960)
    return base64.b64encode(buf.getvalue()).decode()


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


def test_normalize_chunk_notation_replaces_question_mark() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    # Appending "?" before "." yields chunks_generated: 0 on Avaz dashboard builds.
    assert _normalize_text_for_chunk_notation("How are you?", ".") == "How are you."


def test_normalize_chunk_notation_replaces_exclamation() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    assert _normalize_text_for_chunk_notation("Harika!", ".") == "Harika."


def test_normalize_chunk_notation_appends_boundary() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    assert _normalize_text_for_chunk_notation("Merhaba", ".") == "Merhaba."


def test_normalize_chunk_notation_preserves_existing_boundary() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    assert _normalize_text_for_chunk_notation("Merhaba.", ".") == "Merhaba."


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
            agent_model_id=_TEST_UUID,
        )
        == "avaz2"
    )


def test_resolve_stream_model_from_agent_name() -> None:
    from livekit.plugins.avaz.tts import _resolve_stream_model

    assert _resolve_stream_model(stream_model=NOT_GIVEN, agent_model_id="Avaz3") == "avaz3"


def test_log_server_payload_truncates_audio(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from livekit.plugins.avaz.tts import _log_server_payload

    caplog.set_level(logging.DEBUG, logger="livekit.plugins.avaz")
    _log_server_payload({"audio": "A" * 1000, "status": "ok"}, phase="drain")
    assert "<base64 1000 chars>" in caplog.text
    assert "A" * 100 not in caplog.text


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


@pytest.mark.asyncio
async def test_ensure_warmed_retries_after_failed_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS)
    engine._warmed = False

    async def failed_prewarm() -> bool:
        engine._warmed = False
        return False

    prewarm_task = asyncio.create_task(failed_prewarm())
    await prewarm_task
    engine._prewarm_task = prewarm_task

    warmup_calls = 0

    async def retry_warmup(timeout_s: float = 10.0) -> bool:
        nonlocal warmup_calls
        warmup_calls += 1
        return True

    monkeypatch.setattr(engine, "warmup", retry_warmup)

    await engine._ensure_warmed()

    assert engine._warmed is True
    assert warmup_calls == 1


def test_parse_init_response_error() -> None:
    from livekit.plugins.avaz.tts import _parse_init_response

    with pytest.raises(APIConnectionError, match="init error"):
        _parse_init_response('{"error":"model not found"}')


@pytest.mark.asyncio
async def test_stream_run_drains_audio_with_ws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: _drain_audio must receive ws explicitly (not closure)."""
    from livekit.plugins.avaz import TTS, SynthesizeStream

    engine = TTS(
        ws_url=_TEST_WS,
        api_key="test-api-key",
        post_text_drain_s=0.01,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    assert isinstance(stream, SynthesizeStream)
    stream.push_text("Merhaba.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    recv_queue = [
        '{"status":"initialized"}',
        json.dumps({"audio": audio_b64}),
        '{"status":"closed","chunks_generated":1}',
    ]

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if recv_queue:
            return recv_queue.pop(0)
        raise asyncio.TimeoutError

    mock_ws.recv = AsyncMock(side_effect=recv_side_effect)
    mock_ws.send = AsyncMock()

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        frames = 0
        async for _ev in stream:
            frames += 1

    assert frames >= 1
    assert mock_ws.send.await_count >= 2


@pytest.mark.asyncio
async def test_stream_run_turn_avoids_fixed_pre_flush_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-flush path uses recv idle drains, not fixed asyncio.sleep padding."""
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        api_key="test-api-key",
        post_text_drain_s=0.01,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Merhaba.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    recv_queue = [
        '{"status":"initialized"}',
        json.dumps({"audio": audio_b64}),
        '{"status":"closed","chunks_generated":1}',
    ]

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if recv_queue:
            return recv_queue.pop(0)
        raise asyncio.TimeoutError

    mock_ws.recv = AsyncMock(side_effect=recv_side_effect)
    mock_ws.send = AsyncMock()

    sleep_calls: list[float] = []

    async def track_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", track_sleep)

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        async for _ev in stream:
            pass

    assert sleep_calls == []


@pytest.mark.asyncio
async def test_stream_connect_errors_raise_api_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS, api_key="test-api-key", turn_timeout_s=5.0)
    stream = engine.stream()
    stream.push_text("Merhaba.")
    stream.end_input()

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    def failing_connect(*_args: object, **_kwargs: object):
        class CM:
            async def __aenter__(self):
                raise ConnectionRefusedError("connection refused")

            async def __aexit__(self, *_exc: object):
                return None

        return CM()

    monkeypatch.setattr("livekit.plugins.avaz.tts.websockets.connect", failing_connect)

    with pytest.raises(APIConnectionError, match="connection failed"):
        async for _ in stream:
            pass
