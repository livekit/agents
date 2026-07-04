"""Tests for Tencent Cloud TTS plugin."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.parse
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import APIStatusError
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

pytestmark = pytest.mark.unit


class _FakeWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self.messages = messages
        self.closed = False

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data="")

    async def close(self) -> None:
        self.closed = True

    def exception(self):
        return None


class _FakeEmitter:
    def __init__(self) -> None:
        self.initialized: dict[str, object] | None = None
        self.audio: list[bytes] = []
        self.flushed = False
        self.started_segments: list[str] = []
        self.ended_segments = 0

    def initialize(self, **kwargs) -> None:
        self.initialized = kwargs

    def push(self, data: bytes) -> None:
        self.audio.append(data)

    def flush(self) -> None:
        self.flushed = True

    def start_segment(self, *, segment_id: str) -> None:
        self.started_segments.append(segment_id)

    def end_segment(self) -> None:
        self.ended_segments += 1


def _fake_create_task(coro, *args, **kwargs):
    coro.close()
    return MagicMock()


def _text_message(payload: dict[str, object]) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def _binary_message(payload: bytes) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=payload)


def test_tts_requires_tts_credentials(monkeypatch):
    from livekit.plugins.tencent import TTS

    monkeypatch.delenv("TENCENT_TTS_APP_ID", raising=False)
    monkeypatch.delenv("TENCENT_TTS_SECRET_ID", raising=False)
    monkeypatch.delenv("TENCENT_TTS_SECRET_KEY", raising=False)
    monkeypatch.setenv("TENCENT_ASR_APP_ID", "asr-app")
    monkeypatch.setenv("TENCENT_ASR_SECRET_ID", "asr-sid")
    monkeypatch.setenv("TENCENT_ASR_SECRET_KEY", "asr-key")

    with pytest.raises(ValueError, match="Tencent TTS credentials"):
        TTS()


def test_build_url_sorts_params_escapes_text_and_signs():
    from livekit.plugins.tencent import TTS

    tts = TTS(app_id="123", secret_id="sid", secret_key="skey")
    url = tts._build_url(
        tts._opts,
        text="hello world + 你好",
        session_id="session-1",
        now=1_700_000_000,
    )

    unsigned, signature = url.removeprefix("wss://").split("&Signature=", 1)
    path, query = unsigned.split("?", 1)
    keys = [item.split("=", 1)[0] for item in query.split("&")]

    assert path == "tts.cloud.tencent.com/stream_ws"
    assert keys == sorted(keys)
    assert "Text=hello+world+%2B+%E4%BD%A0%E5%A5%BD" in query
    assert "AppId=123" in query
    assert "VoiceType=601010" in query

    sign_query = query.replace("Text=hello+world+%2B+%E4%BD%A0%E5%A5%BD", "Text=hello world + 你好")
    sign_url = f"tts.cloud.tencent.com/stream_ws?{sign_query}"
    expected_signature = base64.b64encode(
        hmac.new(b"skey", f"GET{sign_url}".encode(), hashlib.sha1).digest()
    ).decode("utf-8")
    assert signature == urllib.parse.quote_plus(expected_signature)


async def test_chunked_synthesize_receives_pcm_and_final():
    from livekit.plugins.tencent import TTS
    from livekit.plugins.tencent.tts import ChunkedStream

    fake_ws = _FakeWebSocket(
        [
            _text_message({"code": 0, "request_id": "req-1"}),
            _binary_message(b"audio-1"),
            _binary_message(b"audio-2"),
            _text_message({"code": 0, "final": 1, "request_id": "req-1"}),
        ]
    )
    fake_session = MagicMock()
    fake_session.ws_connect = AsyncMock(return_value=fake_ws)

    tts = TTS(
        app_id="123",
        secret_id="sid",
        secret_key="skey",
        http_session=fake_session,
    )
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = ChunkedStream(
            tts=tts, input_text="hello", conn_options=DEFAULT_API_CONNECT_OPTIONS
        )
    emitter = _FakeEmitter()

    await stream._run(emitter)

    assert emitter.initialized["sample_rate"] == 24000
    assert emitter.initialized["mime_type"] == "audio/pcm"
    assert emitter.audio == [b"audio-1", b"audio-2"]
    assert emitter.flushed is True
    assert fake_ws.closed is True


async def test_synthesize_nonzero_code_raises_status_error():
    from livekit.plugins.tencent import TTS
    from livekit.plugins.tencent.tts import ChunkedStream

    fake_ws = _FakeWebSocket(
        [
            _text_message({"code": 0, "request_id": "req-1"}),
            _text_message({"code": 4001, "message": "auth failed", "request_id": "req-1"}),
        ]
    )
    fake_session = MagicMock()
    fake_session.ws_connect = AsyncMock(return_value=fake_ws)

    tts = TTS(
        app_id="123",
        secret_id="sid",
        secret_key="skey",
        http_session=fake_session,
    )
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = ChunkedStream(
            tts=tts, input_text="hello", conn_options=DEFAULT_API_CONNECT_OPTIONS
        )

    with pytest.raises(APIStatusError) as exc_info:
        await stream._run(_FakeEmitter())

    assert exc_info.value.status_code == 4001
    assert exc_info.value.request_id == "req-1"
    assert exc_info.value.body["message"] == "auth failed"


async def test_stream_flush_synthesizes_each_segment():
    from livekit.plugins.tencent import TTS
    from livekit.plugins.tencent.tts import SynthesizeStream

    fake_session = MagicMock()
    fake_session.ws_connect = AsyncMock(
        side_effect=[
            _FakeWebSocket(
                [
                    _text_message({"code": 0}),
                    _binary_message(b"audio-a"),
                    _text_message({"code": 0, "final": 1}),
                ]
            ),
            _FakeWebSocket(
                [
                    _text_message({"code": 0}),
                    _binary_message(b"audio-b"),
                    _text_message({"code": 0, "final": 1}),
                ]
            ),
        ]
    )

    tts = TTS(
        app_id="123",
        secret_id="sid",
        secret_key="skey",
        http_session=fake_session,
    )
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = SynthesizeStream(tts=tts, conn_options=DEFAULT_API_CONNECT_OPTIONS)
    emitter = _FakeEmitter()

    stream._input_ch.send_nowait("first")
    stream._input_ch.send_nowait(stream._FlushSentinel())
    stream._input_ch.send_nowait("second")
    stream._input_ch.send_nowait(stream._FlushSentinel())
    stream._input_ch.close()

    await stream._run(emitter)

    assert emitter.initialized["stream"] is True
    assert emitter.audio == [b"audio-a", b"audio-b"]
    assert len(emitter.started_segments) == 2
    assert emitter.ended_segments == 2
    assert fake_session.ws_connect.await_count == 2
