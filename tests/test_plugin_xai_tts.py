"""Unit tests for xAI TTS websocket behavior."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectOptions
from livekit.plugins.xai import tts as xai_tts

pytestmark = pytest.mark.plugin("xai")


class _FakeWebSocket:
    def __init__(self, audio: bytes = b"audio") -> None:
        self._audio = audio
        self._messages: asyncio.Queue[Any] = asyncio.Queue()
        self.close_code: int | None = None
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send_str(self, data: str) -> None:
        packet = json.loads(data)
        self.sent.append(packet)
        if packet["type"] == "text.done":
            await self._messages.put(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps(
                        {
                            "type": "audio.delta",
                            "delta": base64.b64encode(self._audio).decode("ascii"),
                        }
                    ),
                )
            )
            await self._messages.put(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps({"type": "audio.done"}),
                )
            )

    async def receive(self) -> Any:
        return await self._messages.get()

    async def close(self) -> None:
        self.closed = True


class _FakeEmitter:
    def __init__(self) -> None:
        self.started_segments: list[str] = []
        self.ended_segments = 0
        self.audio_chunks: list[bytes] = []

    def start_segment(self, *, segment_id: str) -> None:
        self.started_segments.append(segment_id)

    def push(self, audio: bytes) -> None:
        self.audio_chunks.append(audio)

    def end_segment(self) -> None:
        self.ended_segments += 1


def _new_word_stream(tts: xai_tts.TTS, text: str):
    word_stream = tts._opts.tokenizer.stream()  # pyright: ignore[reportPrivateUsage]
    word_stream.push_text(text)
    word_stream.end_input()
    return word_stream


def _new_synthesize_stream(tts: xai_tts.TTS) -> xai_tts.SynthesizeStream:
    stream = object.__new__(xai_tts.SynthesizeStream)
    stream._tts = tts  # pyright: ignore[reportPrivateUsage]
    stream._opts = replace(tts._opts)  # pyright: ignore[reportPrivateUsage]
    stream._conn_options = APIConnectOptions(max_retry=0, timeout=1.0)  # pyright: ignore[reportPrivateUsage]
    stream._started_time = 0.0  # pyright: ignore[reportPrivateUsage]
    stream._acquire_time = 0.0  # pyright: ignore[reportPrivateUsage]
    stream._connection_reused = False  # pyright: ignore[reportPrivateUsage]
    return stream


@pytest.mark.asyncio
async def test_streaming_segments_reuse_websocket_connection(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key")
    stream = _new_synthesize_stream(tts)
    emitter = _FakeEmitter()
    websockets: list[_FakeWebSocket] = []

    async def connect_ws(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        ws = _FakeWebSocket()
        websockets.append(ws)
        return ws

    async def close_ws(ws: _FakeWebSocket) -> None:
        await ws.close()

    monkeypatch.setattr(tts, "_connect_ws", connect_ws)
    monkeypatch.setattr(tts, "_close_ws", close_ws)

    await stream._run_ws(_new_word_stream(tts, "first"), emitter)  # pyright: ignore[reportPrivateUsage]
    await stream._run_ws(_new_word_stream(tts, "second"), emitter)  # pyright: ignore[reportPrivateUsage]

    assert len(websockets) == 1
    assert websockets[0].closed is False
    assert emitter.audio_chunks == [b"audio", b"audio"]
    assert emitter.ended_segments == 2

    await tts.aclose()

    assert websockets[0].closed is True


def test_sample_rate_defaults_to_24000() -> None:
    tts = xai_tts.TTS(api_key="test-key")
    assert tts.sample_rate == 24000
    assert tts._opts.sample_rate == 24000  # pyright: ignore[reportPrivateUsage]


def test_invalid_sample_rate_raises() -> None:
    with pytest.raises(ValueError):
        xai_tts.TTS(api_key="test-key", sample_rate=12345)


@pytest.mark.asyncio
async def test_sample_rate_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class _FakeSession:
        async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
            captured["url"] = url
            return _FakeWebSocket()

    tts = xai_tts.TTS(api_key="test-key", sample_rate=16000)
    assert tts.sample_rate == 16000
    assert tts._opts.sample_rate == 16000  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(tts, "_ensure_session", lambda: _FakeSession())
    await tts._connect_ws(1.0, tts._opts)  # pyright: ignore[reportPrivateUsage]
    assert "sample_rate=16000" in captured["url"]

    await tts.aclose()
