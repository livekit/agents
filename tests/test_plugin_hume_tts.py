from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from livekit.agents import APIConnectOptions
from livekit.plugins.hume import tts as hume_tts


class _WSMessage:
    def __init__(self, msg_type: aiohttp.WSMsgType, data: str | bytes | None = None) -> None:
        self.type = msg_type
        self.data = data
        self.extra = None


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.close_called = False
        self.close_code: int | None = None
        self._messages: asyncio.Queue[_WSMessage] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        packet = json.loads(data)
        self.sent.append(packet)

        if packet.get("close"):
            audio = base64.b64encode(b"\0\0" * 4800).decode("utf-8")
            self._messages.put_nowait(
                _WSMessage(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps(
                        {
                            "type": "audio",
                            "audio": audio,
                            "audio_format": "pcm",
                            "chunk_index": 0,
                            "generation_id": "generation-id",
                            "is_last_chunk": True,
                            "request_id": "request-id",
                            "snippet_id": "snippet-id",
                            "text": "Hello",
                            "transcribed_text": None,
                            "utterance_index": None,
                        }
                    ),
                )
            )
            self._messages.put_nowait(_WSMessage(aiohttp.WSMsgType.CLOSED))

    async def receive(self) -> _WSMessage:
        return await self._messages.get()

    async def close(self) -> None:
        self.close_called = True

    def exception(self) -> Exception | None:
        return None


class _FakeSession:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def ws_connect(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> asyncio.Future[_FakeWebSocket]:
        self.url = url
        self.headers = headers

        fut: asyncio.Future[_FakeWebSocket] = asyncio.Future()
        fut.set_result(self.ws)
        return fut


def test_streaming_capability_defaults_to_true() -> None:
    tts = hume_tts.TTS(api_key="test-key")

    assert tts.capabilities.streaming is True


def test_streaming_capability_can_be_disabled() -> None:
    tts = hume_tts.TTS(api_key="test-key", streaming=False)

    assert tts.capabilities.streaming is False


def test_stream_rejects_utterance_list_context() -> None:
    tts = hume_tts.TTS(api_key="test-key", context=[{"text": "previous"}])

    with pytest.raises(ValueError, match="utterance-list context"):
        tts.stream()


async def test_stream_sends_text_chunks_directly_to_hume_input_stream() -> None:
    ws = _FakeWebSocket()
    session = _FakeSession(ws)
    tts = hume_tts.TTS(
        api_key="test-key",
        voice={"id": "voice-id"},
        description="warm and direct",
        speed=1.1,
        trailing_silence=0.2,
        context="context-generation-id",
        instant_mode=True,
        audio_format=hume_tts.AudioFormat.pcm,
        http_session=session,  # type: ignore[arg-type]
    )

    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0)) as stream:
        stream.push_text("Hel")
        stream.push_text("lo")
        stream.end_input()
        events = [event async for event in stream]

    assert events
    assert {event.frame.sample_rate for event in events} == {hume_tts.SUPPORTED_SAMPLE_RATE}
    assert [packet.get("text") for packet in ws.sent if "text" in packet] == ["Hel", "lo"]
    assert ws.sent[0] == {
        "text": "Hel",
        "voice": {"id": "voice-id"},
        "description": "warm and direct",
        "speed": 1.1,
        "trailing_silence": 0.2,
    }
    assert ws.sent[-1] == {"close": True}

    assert session.url is not None
    parsed = urlparse(session.url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.hume.ai"
    assert parsed.path == "/v0/tts/stream/input"
    assert query["api_key"] == ["test-key"]
    assert query["context_generation_id"] == ["context-generation-id"]
    assert query["format_type"] == ["pcm"]
    assert query["instant_mode"] == ["true"]
    assert query["no_binary"] == ["true"]
    assert query["strip_headers"] == ["true"]
    assert query["version"] == ["1"]
    assert session.headers is not None
    assert session.headers["X-Hume-Client-Name"] == "livekit"
    assert ws.close_called is True
