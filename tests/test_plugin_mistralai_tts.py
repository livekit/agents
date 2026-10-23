"""Tests for Mistral realtime TTS streaming.

Runs a real `SynthesizeStream` against a fake WebSocket that speaks the
`/v1/audio/speech/realtime` protocol, so the session setup, the text events and
the audio conversion are assertable without a network.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import aiohttp
import numpy as np
import pytest

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.plugins import mistralai

pytestmark = pytest.mark.plugin("mistralai")

_MODEL = "voxtral-mini-tts-realtime-3"
_CONN_OPTIONS = APIConnectOptions(max_retry=0, timeout=1.0)
_SAMPLES = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], dtype="<f4")


class _Message:
    def __init__(self, data: dict[str, Any]) -> None:
        self.type = aiohttp.WSMsgType.TEXT
        self.data = json.dumps(data)
        self.extra = None


class _FakeWebSocket:
    def __init__(self, url: str, headers: dict[str, str], error: str | None) -> None:
        self.url = url
        self.headers = headers
        self.sent: list[dict[str, Any]] = []
        self.close_code = None
        self._error = error
        self._events: asyncio.Queue[_Message] = asyncio.Queue()
        self._events.put_nowait(_Message({"type": "session.created", "session": {}}))

    async def send_str(self, message: str) -> None:
        event = json.loads(message)
        self.sent.append(event)
        if event["type"] == "session.update":
            self._events.put_nowait(_Message({"type": "session.updated", "session": {}}))
        elif event["type"] == "input_text.flush":
            if self._error:
                self._events.put_nowait(
                    _Message({"type": "error", "error": {"message": self._error}})
                )
                return
            # split mid-sample to exercise the float32 remainder
            audio = _SAMPLES.tobytes()
            for chunk in (audio[:6], audio[6:]):
                self._events.put_nowait(
                    _Message(
                        {
                            "type": "speech.audio.delta",
                            "audio_data": base64.b64encode(chunk).decode(),
                        }
                    )
                )
            self._events.put_nowait(
                _Message(
                    {
                        "type": "speech.audio.done",
                        "usage": {"prompt_tokens": 7, "completion_tokens": 11},
                    }
                )
            )

    async def receive(self) -> _Message:
        return await self._events.get()

    async def close(self) -> None:
        pass


class _FakeSession:
    def __init__(self, error: str | None = None) -> None:
        self.sockets: list[_FakeWebSocket] = []
        self._error = error

    async def ws_connect(self, url: str, *, headers: dict[str, str]) -> _FakeWebSocket:
        ws = _FakeWebSocket(url, headers, self._error)
        self.sockets.append(ws)
        return ws


async def _synthesize(tts: mistralai.TTS, *texts: str) -> bytes:
    stream = tts.stream(conn_options=_CONN_OPTIONS)
    for text in texts:
        stream.push_text(text)
    stream.end_input()
    return b"".join([bytes(ev.frame.data) async for ev in stream])


async def test_stream_opens_session_and_converts_audio() -> None:
    session = _FakeSession()
    tts = mistralai.TTS(
        model=_MODEL,
        api_key="test-key",
        voice="fr_marie_neutral",
        http_session=session,  # type: ignore[arg-type]
    )
    assert tts.capabilities.streaming

    audio = await _synthesize(tts, "Hello ", "world.")
    await tts.aclose()

    [ws] = session.sockets
    assert ws.url == f"wss://api.mistral.ai/v1/audio/speech/realtime?model={_MODEL}"
    assert ws.headers["Authorization"] == "Bearer test-key"
    assert ws.sent == [
        {
            "type": "session.update",
            "session": {
                "streaming_mode": "responsive",
                "response_format": "pcm",
                "voice": "fr_marie_neutral",
            },
        },
        {"type": "input_text.append", "text": "Hello "},
        {"type": "input_text.append", "text": "world."},
        {"type": "input_text.flush"},
    ]
    expected = (np.clip(_SAMPLES, -1.0, 1.0) * 32767).astype("<i2")
    assert np.frombuffer(audio, dtype="<i2")[: len(expected)].tolist() == expected.tolist()


async def test_connection_is_reused_until_the_voice_changes() -> None:
    session = _FakeSession()
    tts = mistralai.TTS(model=_MODEL, api_key="test-key", http_session=session)  # type: ignore[arg-type]

    await _synthesize(tts, "First.")
    await _synthesize(tts, "Second.")
    assert len(session.sockets) == 1

    tts.update_options(ref_audio="c2FtcGxl")
    await _synthesize(tts, "Third.")
    await tts.aclose()

    assert len(session.sockets) == 2
    assert session.sockets[1].sent[0]["session"]["ref_audio"] == "c2FtcGxl"
    assert "voice" not in session.sockets[1].sent[0]["session"]


async def test_whitespace_only_input_is_not_sent() -> None:
    session = _FakeSession()
    tts = mistralai.TTS(model=_MODEL, api_key="test-key", http_session=session)  # type: ignore[arg-type]

    assert await _synthesize(tts, " ", "\n") == b""
    await tts.aclose()

    assert session.sockets == []


async def test_error_event_raises_status_error() -> None:
    session = _FakeSession(error="voice not found")
    tts = mistralai.TTS(model=_MODEL, api_key="test-key", http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError, match="voice not found"):
        await _synthesize(tts, "Hello.")
    await tts.aclose()


def test_non_realtime_model_does_not_stream() -> None:
    tts = mistralai.TTS(model="voxtral-mini-tts-latest", api_key="test-key")

    assert not tts.capabilities.streaming
    with pytest.raises(NotImplementedError):
        tts.stream()
