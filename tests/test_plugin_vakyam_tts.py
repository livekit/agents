"""Tests for Vakyam TTS plugin configuration and WebSocket protocol helpers."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock, patch

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, APITimeoutError

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
    old_pool = tts._pool_for(tts._opts)
    tts.update_options(language="en-IN", voice="Archana", sample_rate=8000, speed=0.9)
    assert tts._opts.language == "en-IN"
    assert tts.sample_rate == 8000
    assert tts._opts.speed == 0.9
    assert tts._session_config(tts._opts).sample_rate == 8000
    assert tts._pool_for(tts._opts) is not old_pool


def test_update_options_invalid_sample_rate_is_atomic() -> None:
    from livekit.plugins.vakyam import TTS

    tts = TTS(api_key="test-key")
    original_opts = tts._opts
    original_sample_rate = tts.sample_rate

    with pytest.raises(ValueError, match="sample_rate"):
        tts.update_options(sample_rate=22050)

    assert tts._opts == original_opts
    assert tts.sample_rate == original_sample_rate


def test_update_options_mixed_invalid_values_is_atomic() -> None:
    from livekit.plugins.vakyam import TTS

    tts = TTS(api_key="test-key")
    original_opts = tts._opts
    original_sample_rate = tts.sample_rate

    with pytest.raises(ValueError, match="voice"):
        tts.update_options(sample_rate=16000, voice="vc_")

    assert tts._opts == original_opts
    assert tts.sample_rate == original_sample_rate


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


def test_split_text_respects_provider_limit() -> None:
    from livekit.plugins.vakyam._utils import split_text

    chunks = split_text("word " * 1000, max_characters=100)
    assert " ".join(chunks) == ("word " * 1000).strip()
    assert all(0 < len(chunk) <= 100 for chunk in chunks)


@pytest.mark.asyncio
async def test_connection_pool_exclusively_checks_out_websockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.vakyam import TTS
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession

    class FakeWS:
        async def send(self, data: str) -> None:
            return None

        async def close(self) -> None:
            return None

    connected: list[AsyncStreamingTTSSession] = []

    async def fake_connect(self: AsyncStreamingTTSSession, *, timeout: float = 10.0) -> None:
        self._connection = FakeWS()
        connected.append(self)

    monkeypatch.setattr(AsyncStreamingTTSSession, "connect", fake_connect)
    synth = TTS(api_key="test-key")
    pool = synth._pool_for(synth._opts)

    first = await pool.get(timeout=1.0)
    second = await pool.get(timeout=1.0)
    assert first is not second
    assert len(connected) == 2

    pool.put(first)
    pool.put(second)
    reused = await pool.get(timeout=1.0)
    assert reused in {first, second}
    pool.put(reused)
    await synth.aclose()


@pytest.mark.asyncio
async def test_websocket_receive_timeout_resets_active_state() -> None:
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession, TTSSessionConfig

    class StalledWS:
        async def send(self, data: str) -> None:
            return None

        async def recv(self) -> bytes | str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    session = AsyncStreamingTTSSession(api_key="test-key", config=TTSSessionConfig())
    session._connection = StalledWS()

    with pytest.raises(APITimeoutError):
        await anext(session.synthesize_stream("Hello.", timeout=0.01))
    assert session.utterance_active is False


@pytest.mark.asyncio
async def test_websocket_connect_does_not_require_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``asyncio.timeout`` is unavailable on the Python 3.10 minimum version."""
    from websockets.asyncio import client

    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession, TTSSessionConfig

    class HandshakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self._responses = iter(
                [json.dumps({"type": "connected"}), json.dumps({"type": "configured"})]
            )

        async def send(self, data: str) -> None:
            self.sent.append(data)

        async def recv(self) -> str:
            return next(self._responses)

        async def close(self) -> None:
            return None

    ws = HandshakeWS()

    async def fake_connect(*args: object, **kwargs: object) -> HandshakeWS:
        assert kwargs["open_timeout"] == 0.1
        assert kwargs["close_timeout"] == 0.1
        return ws

    monkeypatch.setattr(client, "connect", fake_connect)
    monkeypatch.delattr(asyncio, "timeout", raising=False)

    session = AsyncStreamingTTSSession(api_key="test-key", config=TTSSessionConfig())
    await session.connect(timeout=0.1)

    assert session.connected
    assert json.loads(ws.sent[0])["type"] == "config"


@pytest.mark.asyncio
async def test_keepalive_failure_does_not_log_websocket_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.vakyam import TTS, tts as vakyam_tts

    class FailedSession:
        connected = True

        async def ping(self) -> None:
            raise RuntimeError("Bearer secret-key")

    class Pool:
        def __init__(self) -> None:
            self.removed: object | None = None

        def remove(self, session: object) -> None:
            self.removed = session

    debug = Mock()
    monkeypatch.setattr(vakyam_tts, "KEEPALIVE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(vakyam_tts.logger, "debug", debug)
    session = FailedSession()
    pool = Pool()

    await TTS(api_key="test-key")._keepalive_loop(session, pool)  # type: ignore[arg-type]

    assert pool.removed is session
    debug.assert_called_once_with(
        "Vakyam TTS keepalive failed (%s); evicting session", "RuntimeError"
    )


@pytest.mark.asyncio
async def test_stream_retry_uses_fresh_attempt_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.vakyam import TTS
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession

    class FakeWS:
        def __init__(self, incoming: list[bytes | str | Exception]) -> None:
            self._incoming = incoming

        async def send(self, data: str) -> None:
            return None

        async def recv(self) -> bytes | str:
            item = self._incoming.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        async def close(self) -> None:
            return None

    attempts = 0

    async def fake_connect(self: AsyncStreamingTTSSession, *, timeout: float = 10.0) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            self._connection = FakeWS([RuntimeError("stale socket")])
        else:
            self._connection = FakeWS(
                [
                    b"\0" * 2400,
                    json.dumps(
                        {
                            "type": "end_of_utterance",
                            "characters_used": 6,
                            "duration_seconds": 0.05,
                        }
                    ),
                ]
            )

    monkeypatch.setattr(AsyncStreamingTTSSession, "connect", fake_connect)
    synth = TTS(api_key="test-key")
    stream = synth.stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=0.1)
    )
    stream.push_text("Hello.")
    stream.end_input()

    frames = [event.frame async for event in stream]
    assert frames
    assert attempts == 2
    await synth.aclose()


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_partial_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.vakyam import TTS
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession

    class FailingWS:
        def __init__(self) -> None:
            self._incoming: list[bytes | Exception] = [
                b"\0" * 2400,
                RuntimeError("connection lost"),
            ]

        async def send(self, data: str) -> None:
            return None

        async def recv(self) -> bytes:
            item = self._incoming.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        async def close(self) -> None:
            return None

    attempts = 0

    async def fake_connect(self: AsyncStreamingTTSSession, *, timeout: float = 10.0) -> None:
        nonlocal attempts
        attempts += 1
        self._connection = FailingWS()

    monkeypatch.setattr(AsyncStreamingTTSSession, "connect", fake_connect)
    synth = TTS(api_key="test-key")
    stream = synth.stream(
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=0.1)
    )
    stream.push_text("Hello.")
    stream.end_input()

    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass
    assert attempts == 1
    await synth.aclose()


@pytest.mark.asyncio
async def test_barge_in_drains_and_reuses_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.vakyam import TTS
    from livekit.plugins.vakyam._websocket import AsyncStreamingTTSSession

    class ReusableWS:
        def __init__(self) -> None:
            self.utterance = 0
            self.audio_sent = False
            self.cancel_sent = asyncio.Event()

        async def send(self, data: str) -> None:
            msg_type = json.loads(data)["type"]
            if msg_type == "text":
                self.utterance += 1
                self.audio_sent = False
            elif msg_type == "cancel":
                self.cancel_sent.set()

        async def recv(self) -> bytes | str:
            if not self.audio_sent:
                self.audio_sent = True
                return b"\0" * 2400
            if self.utterance == 1:
                await self.cancel_sent.wait()
                return json.dumps(
                    {"type": "cancellation", "characters_used": 2, "duration_seconds": 0.05}
                )
            return json.dumps(
                {
                    "type": "end_of_utterance",
                    "characters_used": 6,
                    "duration_seconds": 0.05,
                }
            )

        async def close(self) -> None:
            return None

    connection = ReusableWS()
    attempts = 0

    async def fake_connect(self: AsyncStreamingTTSSession, *, timeout: float = 10.0) -> None:
        nonlocal attempts
        attempts += 1
        self._connection = connection

    monkeypatch.setattr(AsyncStreamingTTSSession, "connect", fake_connect)
    synth = TTS(api_key="test-key")

    interrupted = synth.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
    interrupted.push_text("First reply.")
    interrupted.end_input()
    await anext(interrupted)
    await asyncio.wait_for(interrupted.aclose(), timeout=0.5)

    completed = synth.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
    completed.push_text("Second reply.")
    completed.end_input()
    frames = [event.frame async for event in completed]

    assert frames
    assert attempts == 1
    assert connection.utterance == 2
    await synth.aclose()


@pytest.mark.asyncio
async def test_http_stream_accepts_octet_stream_and_uses_read_timeout() -> None:
    from livekit.plugins.vakyam import TTS

    class FakeContent:
        async def iter_chunks(self):
            yield b"\0" * 2400, True

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/octet-stream"}
        content = FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def text(self) -> str:
            return ""

    class FakeSession:
        def __init__(self) -> None:
            self.timeout = None

        def post(self, url, *, json, headers, timeout):
            self.timeout = timeout
            return FakeResponse()

    http_session = FakeSession()
    synth = TTS(api_key="test-key", http_session=http_session)  # type: ignore[arg-type]
    stream = synth.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0, timeout=0.25))

    frames = [event.frame async for event in stream]
    assert frames
    assert http_session.timeout.total is None
    assert http_session.timeout.sock_connect == 0.25
    assert http_session.timeout.sock_read == 0.25
    await synth.aclose()


@pytest.mark.asyncio
async def test_http_stream_rejects_non_audio_success_response() -> None:
    from livekit.plugins.vakyam import TTS

    class FakeContent:
        async def iter_chunks(self):
            if False:
                yield b"", False

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}
        content = FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def text(self) -> str:
            return '{"error":"upstream proxy failure"}'

    class FakeSession:
        def post(self, url, *, json, headers, timeout):
            return FakeResponse()

    synth = TTS(api_key="test-key", http_session=FakeSession())  # type: ignore[arg-type]
    stream = synth.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0, timeout=0.25))

    with pytest.raises(APIStatusError, match="non-audio") as exc_info:
        async for _ in stream:
            pass
    assert exc_info.value.status_code == 502
    await synth.aclose()


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

    with pytest.raises(APIStatusError, match="status 429") as exc_info:
        raise_http_error(
            429,
            json.dumps({"error": {"code": "rate_limit_exceeded", "message": "Too many requests."}}),
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert exc_info.value.body == {"status_code": 429, "error_code": "rate_limit_exceeded"}
    assert "Too many requests" not in str(exc_info.value)


def test_raise_http_error_does_not_retain_raw_response() -> None:
    from livekit.plugins.vakyam._utils import raise_http_error

    secret = "customer text and bearer secret"
    with pytest.raises(APIStatusError) as exc_info:
        raise_http_error(500, json.dumps({"error": {"message": secret}}))

    assert exc_info.value.body == {"status_code": 500}
    assert secret not in str(exc_info.value)


def test_raise_ws_error_does_not_retain_raw_event() -> None:
    from livekit.plugins.vakyam._utils import raise_ws_error

    secret = "customer text and bearer secret"
    with pytest.raises(APIStatusError) as exc_info:
        raise_ws_error({"type": "error", "error": {"code": "internal_error", "message": secret}})

    assert exc_info.value.body == {"type": "error", "code": "internal_error"}
    assert secret not in str(exc_info.value)


def test_websocket_auth_close_is_not_retryable() -> None:
    from livekit.plugins.vakyam._websocket import _websocket_connection_error

    class ReceivedClose:
        code = 4001
        reason = "invalid key"

    class Closed(Exception):
        rcvd = ReceivedClose()

    error = _websocket_connection_error(Closed(), message="connect failed")
    assert isinstance(error, APIStatusError)
    assert error.status_code == 401
    assert error.retryable is False
    assert error.body is None
    assert "invalid key" not in str(error)


def test_websocket_internal_close_is_retryable() -> None:
    from livekit.plugins.vakyam._websocket import _websocket_connection_error

    class ReceivedClose:
        code = 1011
        reason = "session lost"

    class Closed(Exception):
        rcvd = ReceivedClose()

    error = _websocket_connection_error(Closed(), message="synthesis failed")
    assert isinstance(error, APIStatusError)
    assert error.status_code == 1011
    assert error.retryable is True
    assert error.body is None
    assert "session lost" not in str(error)
