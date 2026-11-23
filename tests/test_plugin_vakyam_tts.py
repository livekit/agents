from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import Mock

import pytest
from websockets.asyncio import client

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, APITimeoutError
from livekit.plugins.vakyam import TTS, tts as vakyam_tts
from livekit.plugins.vakyam._utils import (
    http_stream_url,
    normalize_base_url,
    raise_http_error,
    raise_ws_error,
    speech_payload,
    split_text,
    validate_text,
    websocket_url,
)
from livekit.plugins.vakyam._websocket import (
    AsyncStreamingTTSSession,
    TTSSessionConfig,
    _websocket_connection_error,
)

pytestmark = pytest.mark.unit


class _ScriptedWS:
    """Returns scripted frames from recv, raising any that are exceptions."""

    def __init__(self, incoming: list[bytes | str | Exception]) -> None:
        self.sent: list[str] = []
        self._incoming = incoming

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> bytes | str:
        item = self._incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        return None


def _event(type_: str, characters: int = 6) -> str:
    return json.dumps({"type": type_, "characters_used": characters, "duration_seconds": 0.05})


def test_api_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAKYAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        TTS(api_key=None)

    monkeypatch.setenv("VAKYAM_API_KEY", "env-key")
    assert TTS()._opts.api_key == "env-key"


def test_defaults_and_custom_options() -> None:
    tts = TTS(api_key="test-key")
    assert (tts.model, tts.sample_rate, tts.capabilities.streaming) == ("raaga-v1", 24000, True)
    assert (tts._opts.voice, tts._opts.language, tts._opts.speed) == ("Archana", "ta-IN", 1.0)

    custom = TTS(
        api_key="test-key", voice="vc_01EXAMPLE", language="hi-IN", sample_rate=16000, speed=1.2
    )
    assert (custom._opts.voice, custom._opts.language, custom.sample_rate, custom._opts.speed) == (
        "vc_01EXAMPLE",
        "hi-IN",
        16000,
        1.2,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"language": "en-US"}, "language"),
        ({"sample_rate": 22050}, "sample_rate"),
        ({"speed": 3.0}, "speed"),
        ({"voice": "   "}, "voice"),
        ({"voice": "vc_"}, "vc_"),
    ],
)
def test_invalid_options_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TTS(api_key="test-key", **kwargs)


def test_update_options_applies_to_new_sessions() -> None:
    tts = TTS(api_key="test-key")
    old_pool = tts._pool_for(tts._opts)
    tts.update_options(language="en-IN", voice="Archana", sample_rate=8000, speed=0.9)

    assert (tts._opts.language, tts.sample_rate, tts._opts.speed) == ("en-IN", 8000, 0.9)
    assert tts._session_config(tts._opts).sample_rate == 8000
    assert tts._pool_for(tts._opts) is not old_pool


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"sample_rate": 22050}, "sample_rate"), ({"sample_rate": 16000, "voice": "vc_"}, "voice")],
)
def test_rejected_update_options_is_atomic(kwargs: dict[str, Any], match: str) -> None:
    tts = TTS(api_key="test-key")
    original_opts, original_rate = tts._opts, tts.sample_rate

    with pytest.raises(ValueError, match=match):
        tts.update_options(**kwargs)

    assert (tts._opts, tts.sample_rate) == (original_opts, original_rate)


def test_urls_require_https_except_localhost() -> None:
    assert websocket_url("https://api.vakyam.ai") == "wss://api.vakyam.ai/v1/tts/websocket"
    assert http_stream_url("https://api.vakyam.ai") == "https://api.vakyam.ai/v1/tts/stream"
    assert normalize_base_url("http://localhost:8080") == "http://localhost:8080"
    with pytest.raises(ValueError, match="HTTPS"):
        normalize_base_url("http://example.com")


def test_wire_payloads() -> None:
    config = TTSSessionConfig(
        model="raaga-v1", voice="Archana", language="ta-IN", sample_rate=24000, speed=1.0
    ).to_wire_message()
    assert (config["type"], config["model_id"], config["voice"], config["output_format"]) == (
        "config",
        "raaga-v1",
        "Archana",
        "pcm",
    )

    payload = speech_payload(
        text="வணக்கம்.",
        model="raaga-v1",
        voice="Archana",
        language="ta-IN",
        sample_rate=24000,
        speed=1.0,
    )
    assert (payload["model_id"], payload["output_format"], payload["text"]) == (
        "raaga-v1",
        "pcm",
        "வணக்கம்.",
    )


def test_text_limits() -> None:
    with pytest.raises(ValueError, match="3000"):
        validate_text("a" * 3001)

    chunks = split_text("word " * 1000, max_characters=100)
    assert " ".join(chunks) == ("word " * 1000).strip()
    assert all(0 < len(chunk) <= 100 for chunk in chunks)


async def test_connection_pool_exclusively_checks_out_websockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


async def test_websocket_receive_timeout_resets_active_state() -> None:

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


async def test_websocket_connect_does_not_require_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``asyncio.timeout`` is unavailable on the Python 3.10 minimum version."""

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


async def test_keepalive_failure_does_not_log_websocket_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

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


@pytest.mark.virtual_time
@pytest.mark.parametrize(
    ("scripts", "attempts"),
    [
        # a stale socket fails before any audio, so the retry starts from clean state
        ([[RuntimeError("stale socket")], [b"\0" * 2400, _event("end_of_utterance")]], 2),
        # audio already reached the caller, so a retry would repeat it
        ([[b"\0" * 2400, RuntimeError("connection lost")]] * 4, 1),
    ],
    ids=["fails-before-audio", "fails-after-audio"],
)
async def test_stream_retries_only_before_audio(
    monkeypatch: pytest.MonkeyPatch, scripts: list[list[Any]], attempts: int
) -> None:
    connects = 0

    async def fake_connect(self: AsyncStreamingTTSSession, *, timeout: float = 10.0) -> None:
        nonlocal connects
        self._connection = _ScriptedWS(list(scripts[connects]))
        connects += 1

    monkeypatch.setattr(AsyncStreamingTTSSession, "connect", fake_connect)
    synth = TTS(api_key="test-key")
    stream = synth.stream(
        conn_options=APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=0.1)
    )
    stream.push_text("Hello.")
    stream.end_input()

    if attempts == 1:
        with pytest.raises(APIConnectionError):
            async for _ in stream:
                pass
    else:
        assert [event.frame async for event in stream]
    assert connects == attempts
    await synth.aclose()


async def test_barge_in_drains_and_reuses_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
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


async def test_http_stream_accepts_octet_stream_and_uses_read_timeout() -> None:

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


async def test_http_stream_rejects_non_audio_success_response() -> None:

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


async def test_synthesize_stream_waits_for_end_of_utterance() -> None:
    session = AsyncStreamingTTSSession(api_key="test-key", config=TTSSessionConfig())
    ws = _ScriptedWS([b"pcm-1", b"pcm-2", _event("end_of_utterance", characters=4)])
    session._connection = ws

    assert [chunk async for chunk in session.synthesize_stream("Hello.")] == [b"pcm-1", b"pcm-2"]
    assert json.loads(ws.sent[0]) == {"type": "text", "text": "Hello."}
    assert session.last_result is not None
    assert (session.last_result.cancelled, session.last_result.characters_used) == (False, 4)
    assert session.utterance_active is False


async def test_synthesize_stream_cancel_drains_until_cancellation() -> None:
    session = AsyncStreamingTTSSession(api_key="test-key", config=TTSSessionConfig())
    ws = _ScriptedWS([b"pcm-1", _event("cancellation", characters=2)])
    session._connection = ws

    agen = session.synthesize_stream("Hello.")
    assert await agen.__anext__() == b"pcm-1"
    with pytest.raises(asyncio.CancelledError):
        await agen.athrow(asyncio.CancelledError)

    assert session.last_result is not None and session.last_result.cancelled is True
    assert "cancel" in [json.loads(item)["type"] for item in ws.sent]


def test_raise_http_error_parses_envelope() -> None:
    with pytest.raises(APIStatusError, match="status 429") as exc_info:
        raise_http_error(
            429,
            json.dumps({"error": {"code": "rate_limit_exceeded", "message": "Too many requests."}}),
        )
    assert (exc_info.value.status_code, exc_info.value.retryable) == (429, True)
    assert exc_info.value.body == {"status_code": 429, "error_code": "rate_limit_exceeded"}


_SECRET = "customer text and bearer secret"


@pytest.mark.parametrize(
    ("raise_error", "body"),
    [
        (
            lambda: raise_http_error(500, json.dumps({"error": {"message": _SECRET}})),
            {"status_code": 500},
        ),
        (
            lambda: raise_ws_error(
                {"type": "error", "error": {"code": "internal_error", "message": _SECRET}}
            ),
            {"type": "error", "code": "internal_error"},
        ),
    ],
    ids=["http", "ws"],
)
def test_provider_errors_drop_raw_text(raise_error: Any, body: dict[str, Any]) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        raise_error()

    assert exc_info.value.body == body
    assert _SECRET not in str(exc_info.value)


@pytest.mark.parametrize(
    ("code", "status_code", "retryable"), [(4001, 401, False), (1011, 1011, True)]
)
def test_websocket_close_maps_to_status_error(code: int, status_code: int, retryable: bool) -> None:
    class ReceivedClose:
        reason = "secret reason"

    ReceivedClose.code = code  # type: ignore[attr-defined]

    class Closed(Exception):
        rcvd = ReceivedClose()

    error = _websocket_connection_error(Closed(), message="failed")
    assert isinstance(error, APIStatusError)
    assert (error.status_code, error.retryable, error.body) == (status_code, retryable, None)
    assert "secret reason" not in str(error)
