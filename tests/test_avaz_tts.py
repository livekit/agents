"""Unit tests for the Avaz TTS plugin."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import wave
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.unit

_TEST_WS = "ws://127.0.0.1:8893/tts/stream-input"
_TEST_UUID = "15658888-374f-4739-a0c5-4f1d1c128d2a"

# Provider env vars used by resolvers; clear by default so unit tests stay hermetic.
_AVAZ_ENV_KEYS = (
    "AVAZ_AGENT_MODEL_ID",
    "AVAZ_BASE_URL",
    "AVAZ_API_KEY",
    "AVAZ_STREAM_MODEL",
    "TTS_WS_URI",
)


@pytest.fixture(autouse=True)
def _clear_avaz_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _AVAZ_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _minimal_wav_b64() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes(b"\x00" * 960)
    return base64.b64encode(buf.getvalue()).decode()


def test_tts_init_with_ws_url() -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS)
    assert engine.provider == "avaz"
    assert engine.model == "avaz3"
    assert engine._opts.ws_url == _TEST_WS


def test_tts_base_url_derives_wss() -> None:
    from livekit.plugins.avaz import TTS

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
    assert msg["model_settings"]["agent_model_id"] == _TEST_UUID
    assert json.dumps(msg)


def test_ctor_non_uuid_model_id_not_sent_as_agent_model_id() -> None:
    """Plain model names belong in stream_model only — match set_voice_ids."""
    from livekit.plugins.avaz import TTS
    from livekit.plugins.avaz.tts import _build_init_message

    engine = TTS(
        api_key="test-api-key",
        base_url="https://test.example.com/api",
        model_id="avaz2",
    )
    assert engine._opts.stream_model == "avaz2"
    assert engine._opts.agent_model_id == ""
    assert engine.model == "avaz2"
    msg = _build_init_message(engine._opts)
    assert msg["model_settings"]["model_id"] == "avaz2"
    assert "agent_model_id" not in msg["model_settings"]


def test_derive_ws_url_from_base() -> None:
    from livekit.plugins.avaz.tts import _derive_ws_url_from_base

    assert (
        _derive_ws_url_from_base("https://dashboard.example/api")
        == "wss://dashboard.example/api/tts/stream-input"
    )
    assert (
        _derive_ws_url_from_base("http://dashboard.example/api")
        == "ws://dashboard.example/api/tts/stream-input"
    )
    assert (
        _derive_ws_url_from_base("dashboard.example/api")
        == "wss://dashboard.example/api/tts/stream-input"
    )
    assert (
        _derive_ws_url_from_base("dashboard.example:8443/api")
        == "wss://dashboard.example:8443/api/tts/stream-input"
    )


def test_derive_ws_url_rejects_empty_host() -> None:
    from livekit.plugins.avaz.tts import _derive_ws_url_from_base

    with pytest.raises(ValueError, match="http:// or https://"):
        _derive_ws_url_from_base("/only/path")


def test_rejects_plaintext_ws_with_api_key() -> None:
    from livekit.plugins.avaz import TTS

    with pytest.raises(ValueError, match="unencrypted ws://"):
        TTS(ws_url="ws://remote.example/tts/stream-input", api_key="secret")


def test_rejects_loopback_plaintext_ws_with_api_key() -> None:
    from livekit.plugins.avaz import TTS

    with pytest.raises(ValueError, match="unencrypted ws://"):
        TTS(ws_url="ws://127.0.0.1:8080/tts/stream-input", api_key="secret")


def test_explicit_ws_url_skips_dashboard_api_key_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz import TTS

    monkeypatch.setenv("AVAZ_BASE_URL", "https://dashboard.example/api")
    monkeypatch.delenv("AVAZ_API_KEY", raising=False)
    engine = TTS(ws_url=_TEST_WS)
    assert engine._opts.ws_url == _TEST_WS
    assert engine._opts.base_url == ""
    assert engine._opts.api_key == ""


def test_normalize_chunk_notation_preserves_question_mark() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    # Keep ? for prosody; append chunk boundary without a space.
    assert _normalize_text_for_chunk_notation("How are you?", ".") == "How are you?."


def test_normalize_chunk_notation_preserves_exclamation() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    assert _normalize_text_for_chunk_notation("Harika!", ".") == "Harika!."


def test_normalize_chunk_notation_appends_boundary() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    assert _normalize_text_for_chunk_notation("Merhaba", ".") == "Merhaba."


def test_normalize_chunk_notation_preserves_existing_boundary() -> None:
    from livekit.plugins.avaz.tts import _normalize_text_for_chunk_notation

    assert _normalize_text_for_chunk_notation("Merhaba.", ".") == "Merhaba."


def test_chunk_boundary_to_append_ignores_surrounding_whitespace() -> None:
    from livekit.plugins.avaz.tts import _chunk_boundary_to_append

    assert _chunk_boundary_to_append("Hello world ", ".") == "."
    assert _chunk_boundary_to_append("  Merhaba", ".") == "."
    assert _chunk_boundary_to_append("Merhaba.", ".") == ""
    assert _chunk_boundary_to_append("How are you? ", ".") == "."
    assert _chunk_boundary_to_append("   ", ".") == ""


@pytest.mark.asyncio
async def test_stream_appends_boundary_when_text_ends_with_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing whitespace must not suppress the chunk-boundary text frame."""
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        post_text_drain_s=0.0,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Hello world ")
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
        async for _ev in stream:
            pass

    texts = [
        json.loads(c.args[0]).get("text")
        for c in mock_ws.send.call_args_list
        if "text" in json.loads(c.args[0])
    ]
    assert texts == ["Hello world ", "."]


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


def test_set_voice_ids_non_uuid_clears_agent_model_id() -> None:
    from livekit.plugins.avaz import TTS
    from livekit.plugins.avaz.tts import _build_init_message

    engine = TTS(ws_url=_TEST_WS)
    engine.set_voice_ids(model_id=_TEST_UUID)
    engine.set_voice_ids(model_id="avaz2")
    assert engine._opts.stream_model == "avaz2"
    assert engine._opts.agent_model_id == ""
    msg = _build_init_message(engine._opts)
    assert msg["model_settings"]["model_id"] == "avaz2"
    assert "agent_model_id" not in msg["model_settings"]


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


def test_resolve_stream_model_explicit_name_beats_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz.tts import _resolve_stream_model

    monkeypatch.setenv("AVAZ_STREAM_MODEL", "avaz1")
    assert (
        _resolve_stream_model(
            stream_model=NOT_GIVEN,
            agent_model_id="avaz2",
            model_id_explicit=True,
        )
        == "avaz2"
    )


def test_ctor_explicit_model_name_beats_stream_model_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz import TTS

    monkeypatch.setenv("AVAZ_STREAM_MODEL", "avaz1")
    engine = TTS(
        api_key="test-api-key",
        base_url="https://test.example.com/api",
        model_id="avaz2",
    )
    assert engine._opts.stream_model == "avaz2"


def test_resolve_stream_model_from_agent_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz.tts import _resolve_stream_model

    monkeypatch.delenv("AVAZ_STREAM_MODEL", raising=False)
    assert _resolve_stream_model(stream_model=NOT_GIVEN, agent_model_id="Avaz3") == "avaz3"


def test_parse_init_response_keeps_audio_payload() -> None:
    from livekit.plugins.avaz.tts import _parse_init_response

    payload = _parse_init_response(json.dumps({"audio": "AAAA", "status": "weird"}))
    assert payload["audio"] == "AAAA"


def test_log_server_payload_truncates_audio(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from livekit.plugins.avaz.tts import _log_server_payload

    caplog.set_level(logging.DEBUG, logger="livekit.plugins.avaz")
    _log_server_payload({"audio": "A" * 1000, "status": "ok"}, phase="drain")
    assert "<base64 1000 chars>" in caplog.text
    assert "A" * 100 not in caplog.text


def test_log_server_payload_redacts_sensitive_fields(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from livekit.plugins.avaz.tts import _log_server_payload, _summarize_server_payload

    payload = {
        "status": "ok",
        "api_key": "super-secret-key-value",
        "Authorization": "Bearer leaked-credential",
        "model_settings": {"access_token": "session-secret-xyz", "model_id": "avaz3"},
    }
    summary = _summarize_server_payload(payload)
    assert summary["api_key"] == "<redacted>"
    assert summary["Authorization"] == "<redacted>"
    assert summary["model_settings"]["access_token"] == "<redacted>"
    assert summary["model_settings"]["model_id"] == "avaz3"

    caplog.set_level(logging.DEBUG, logger="livekit.plugins.avaz")
    _log_server_payload(payload, phase="init")
    assert "super-secret-key-value" not in caplog.text
    assert "leaked-credential" not in caplog.text
    assert "session-secret-xyz" not in caplog.text
    assert "<redacted>" in caplog.text


@pytest.mark.asyncio
async def test_warmup_passes_auth_headers() -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url="wss://127.0.0.1:8893/tts/stream-input",
        api_key="test-api-key",
    )
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
        assert "ssl" in kwargs


@pytest.mark.asyncio
async def test_ensure_warmed_skips_after_failed_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS)
    engine._warmed = False

    async def failed_prewarm() -> bool:
        engine._warmed = False
        engine._warmup_attempted = True
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

    assert engine._warmed is False
    assert engine._warmup_attempted is True
    assert warmup_calls == 0


@pytest.mark.asyncio
async def test_warmup_exits_on_idle_without_audio() -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS, recv_idle_timeout_s=0.3)
    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)
    # Init ok, then no further frames — idle should end warm-up quickly.
    mock_ws.recv = AsyncMock(
        side_effect=[
            '{"status":"initialized"}',
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
        ]
    )
    mock_ws.send = AsyncMock()

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        t0 = time.monotonic()
        ok = await engine.warmup(timeout_s=15.0)
        elapsed = time.monotonic() - t0

    assert ok is False
    assert elapsed < 3.0
    sent_payloads = [json.loads(call.args[0]) for call in mock_ws.send.call_args_list]
    assert any(p.get("text") == "warmup." for p in sent_payloads)
    assert any(p.get("flush") is True for p in sent_payloads)


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
async def test_stream_empty_text_is_noop_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-only / empty turns must not raise APIConnectionError or open a WS."""
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS)
    stream = engine.stream()
    stream.end_input()

    async def fail_warmup(timeout_s: float = 10.0) -> bool:
        raise AssertionError("warmup should not run for empty text turns")

    monkeypatch.setattr(engine, "warmup", fail_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect") as connect:
        frames = 0
        async for _ev in stream:
            frames += 1

    assert frames == 0
    connect.assert_not_called()


@pytest.mark.asyncio
async def test_stream_run_turn_sends_flush_after_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protocol: init → text → flush; no fixed pre-flush sleeps in the plugin."""
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
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

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        async for _ev in stream:
            pass

    payloads = [json.loads(c.args[0]) for c in mock_ws.send.call_args_list]
    assert any("model_settings" in p for p in payloads)
    assert any(p.get("text") == "Merhaba." for p in payloads)
    assert any(p.get("flush") is True for p in payloads)


@pytest.mark.asyncio
async def test_stream_clean_ws_close_after_audio_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal end-of-turn ConnectionClosedOK must not fail a turn that already got audio."""
    import websockets.exceptions

    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        post_text_drain_s=0.01,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Merhaba.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    recv_queue: list[object] = [
        '{"status":"initialized"}',
        json.dumps({"audio": audio_b64}),
        '{"status":"closed","chunks_generated":1}',
        websockets.exceptions.ConnectionClosedOK(None, None),
    ]

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if not recv_queue:
            raise asyncio.TimeoutError
        item = recv_queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

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


@pytest.mark.asyncio
async def test_stream_skips_flush_when_ws_closed_after_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the server hangs up after audio, do not fail the turn on flush send."""
    import websockets.exceptions

    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        post_text_drain_s=0.01,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Merhaba.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    recv_queue: list[object] = [
        '{"status":"initialized"}',
        json.dumps({"audio": audio_b64}),
        websockets.exceptions.ConnectionClosedOK(None, None),
    ]

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if not recv_queue:
            raise asyncio.TimeoutError
        item = recv_queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    mock_ws.recv = AsyncMock(side_effect=recv_side_effect)

    async def send_side_effect(payload: str) -> None:
        data = json.loads(payload)
        if data.get("flush"):
            raise websockets.exceptions.ConnectionClosedOK(None, None)

    mock_ws.send = AsyncMock(side_effect=send_side_effect)

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        frames = 0
        async for _ev in stream:
            frames += 1

    assert frames >= 1


@pytest.mark.asyncio
async def test_stream_tolerates_closed_ws_while_sending_text_after_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hang-up while sending later text must not fail a turn that already got audio."""
    import websockets.exceptions

    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        post_text_drain_s=0.0,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Hello. ")
    stream.push_text("More text.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    recv_queue: list[object] = [
        '{"status":"initialized"}',
        json.dumps({"audio": audio_b64}),
        '{"status":"closed","chunks_generated":1}',
    ]
    audio_seen = asyncio.Event()

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if not recv_queue:
            raise asyncio.TimeoutError
        item = recv_queue.pop(0)
        if isinstance(item, str) and '"audio"' in item:
            audio_seen.set()
        return item  # type: ignore[return-value]

    async def send_side_effect(payload: str) -> None:
        data = json.loads(payload)
        if data.get("text") == "More text.":
            await audio_seen.wait()
            # Let _handle_audio_payload mark emitter_ready before we hang up.
            await asyncio.sleep(0.05)
            raise websockets.exceptions.ConnectionClosedOK(None, None)

    mock_ws.recv = AsyncMock(side_effect=recv_side_effect)
    mock_ws.send = AsyncMock(side_effect=send_side_effect)

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        frames = 0
        async for _ev in stream:
            frames += 1

    assert frames >= 1


@pytest.mark.asyncio
async def test_stream_tolerates_closed_ws_while_sending_boundary_after_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hang-up while sending the chunk-boundary frame must not fail after audio."""
    import websockets.exceptions

    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        post_text_drain_s=0.0,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Merhaba")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    recv_queue: list[object] = [
        '{"status":"initialized"}',
        json.dumps({"audio": audio_b64}),
        '{"status":"closed","chunks_generated":1}',
    ]
    audio_seen = asyncio.Event()

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if not recv_queue:
            raise asyncio.TimeoutError
        item = recv_queue.pop(0)
        if isinstance(item, str) and '"audio"' in item:
            audio_seen.set()
        return item  # type: ignore[return-value]

    async def send_side_effect(payload: str) -> None:
        data = json.loads(payload)
        if data.get("text") == ".":
            await audio_seen.wait()
            await asyncio.sleep(0.05)
            raise websockets.exceptions.ConnectionClosedOK(None, None)

    mock_ws.recv = AsyncMock(side_effect=recv_side_effect)
    mock_ws.send = AsyncMock(side_effect=send_side_effect)

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        frames = 0
        async for _ev in stream:
            frames += 1

    assert frames >= 1


@pytest.mark.asyncio
async def test_stream_drain_idle_timeout_extends_on_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recv idle wait must restart per frame so long replies are not truncated."""
    from livekit.plugins.avaz import TTS

    idle_timeout = 0.12
    engine = TTS(
        ws_url=_TEST_WS,
        post_text_drain_s=0.0,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=idle_timeout,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Merhaba.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    # Spaced just under the post-flush idle window; a single deadline from
    # drain start would time out before the last frame.
    gap = idle_timeout * 0.7
    recv_plan: list[tuple[float, object]] = [
        (0.0, '{"status":"initialized"}'),
        (0.0, json.dumps({"audio": audio_b64, "chunk_index": 0})),
        (gap, json.dumps({"audio": audio_b64, "chunk_index": 1})),
        (gap, json.dumps({"audio": audio_b64, "chunk_index": 2})),
        (0.0, '{"status":"closed","chunks_generated":3}'),
    ]
    audio_chunks_delivered: list[int] = []

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if not recv_plan:
            raise asyncio.TimeoutError
        delay, item = recv_plan.pop(0)
        if delay:
            await asyncio.sleep(delay)
        if isinstance(item, str) and '"audio"' in item:
            payload = json.loads(item)
            audio_chunks_delivered.append(int(payload.get("chunk_index", -1)))
        return item  # type: ignore[return-value]

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
    assert audio_chunks_delivered == [0, 1, 2]


@pytest.mark.asyncio
async def test_stream_ends_promptly_on_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal status must end the turn without waiting out flush idle."""
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        recv_idle_timeout_s=2.0,
        flush_recv_timeout_s=5.0,
        turn_timeout_s=10.0,
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

    async def fake_warmup(timeout_s: float = 10.0) -> bool:
        engine._warmed = True
        return True

    monkeypatch.setattr(engine, "warmup", fake_warmup)

    with patch("livekit.plugins.avaz.tts.websockets.connect", return_value=mock_ws):
        t0 = time.monotonic()
        async for _ev in stream:
            pass
        elapsed = time.monotonic() - t0

    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_stream_forwards_text_chunks_as_they_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each input token should be sent as its own WebSocket text frame."""
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.05,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Mer")
    stream.push_text("haba.")
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
        async for _ev in stream:
            pass

    texts = [
        json.loads(c.args[0]).get("text")
        for c in mock_ws.send.call_args_list
        if "text" in json.loads(c.args[0])
    ]
    assert texts == ["Mer", "haba."]


@pytest.mark.asyncio
async def test_ensure_warmed_bounds_prewarm_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow prewarm task must not block the first turn beyond the inline budget."""
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS)
    started = asyncio.Event()

    async def slow_prewarm() -> bool:
        started.set()
        await asyncio.sleep(30.0)
        engine._warmed = True
        engine._warmup_attempted = True
        return True

    engine._prewarm_task = asyncio.create_task(slow_prewarm())
    await started.wait()

    monkeypatch.setattr(
        "livekit.plugins.avaz.tts.DEFAULT_INLINE_WARMUP_TIMEOUT_S",
        0.05,
    )

    t0 = time.monotonic()
    await engine._ensure_warmed()
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0
    assert engine._warmup_attempted is True

    # Subsequent turns must not pay the inline budget again.
    t1 = time.monotonic()
    await engine._ensure_warmed()
    assert time.monotonic() - t1 < 0.2

    engine._prewarm_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await engine._prewarm_task


@pytest.mark.asyncio
async def test_stream_honours_post_flush_window_after_short_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flush during a short pre-flush recv wait must still get the post-flush budget."""
    from livekit.plugins.avaz import TTS

    engine = TTS(
        ws_url=_TEST_WS,
        recv_idle_timeout_s=0.05,
        flush_recv_timeout_s=0.25,
        post_text_drain_s=0.0,
        turn_timeout_s=5.0,
    )
    stream = engine.stream()
    stream.push_text("Merhaba.")
    stream.end_input()

    audio_b64 = _minimal_wav_b64()
    # Init, then a short idle (flush races in), then audio inside post-flush window.
    timeout_sentinel = object()
    recv_plan: list[tuple[float, object]] = [
        (0.0, '{"status":"initialized"}'),
        (0.0, timeout_sentinel),
        (0.12, json.dumps({"audio": audio_b64})),
        (0.0, '{"status":"closed","chunks_generated":1}'),
    ]

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    async def recv_side_effect() -> str:
        if not recv_plan:
            raise asyncio.TimeoutError
        delay, item = recv_plan.pop(0)
        if delay:
            await asyncio.sleep(delay)
        if item is timeout_sentinel:
            raise asyncio.TimeoutError
        return item  # type: ignore[return-value]

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


@pytest.mark.asyncio
async def test_ensure_warmed_bounds_inline_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline warm-up must be capped even when connect_timeout_s is larger."""
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS, connect_timeout_s=10.0)
    started = asyncio.Event()

    async def slow_warmup(timeout_s: float = 10.0) -> bool:
        started.set()
        await asyncio.sleep(30.0)
        return True

    monkeypatch.setattr(engine, "warmup", slow_warmup)
    monkeypatch.setattr(
        "livekit.plugins.avaz.tts.DEFAULT_INLINE_WARMUP_TIMEOUT_S",
        0.05,
    )

    t0 = time.monotonic()
    await engine._ensure_warmed()
    elapsed = time.monotonic() - t0

    assert started.is_set()
    assert elapsed < 1.0
    assert engine._warmup_attempted is True
    assert engine._warmed is False


def test_ws_connect_kwargs_sets_ssl_for_wss() -> None:
    from livekit.plugins.avaz.tts import _ws_connect_kwargs

    kwargs = _ws_connect_kwargs("secret", ws_url="wss://dashboard.example/api/tts/stream-input")
    assert "ssl" in kwargs
    assert kwargs["additional_headers"]["X-API-Key"] == "secret"

    plain = _ws_connect_kwargs("secret", ws_url="ws://127.0.0.1:8080/tts/stream-input")
    assert "ssl" not in plain


@pytest.mark.asyncio
async def test_stream_connect_errors_raise_api_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.avaz import TTS

    engine = TTS(ws_url=_TEST_WS, turn_timeout_s=5.0)
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
