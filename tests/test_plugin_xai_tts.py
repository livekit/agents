"""Unit tests for xAI TTS websocket behavior."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectOptions
from livekit.plugins.xai import tts as xai_tts

pytestmark = pytest.mark.plugin("xai")


class _FakeWebSocket:
    def __init__(self, audio: bytes = b"\x00\x00" * 240) -> None:
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


@pytest.mark.asyncio
async def test_streaming_segments_reuse_websocket_connection(monkeypatch: pytest.MonkeyPatch):
    tts = xai_tts.TTS(api_key="test-key")
    websockets: list[_FakeWebSocket] = []

    async def connect_ws(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        ws = _FakeWebSocket()
        websockets.append(ws)
        return ws

    async def close_ws(ws: _FakeWebSocket) -> None:
        await ws.close()

    monkeypatch.setattr(tts, "_connect_ws", connect_ws)
    monkeypatch.setattr(tts, "_close_ws", close_ws)

    try:
        async with tts.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("first")
            stream.flush()
            stream.push_text("second")
            stream.end_input()
            heard = b"".join([event.frame.data.tobytes() async for event in stream])
        assert len(websockets) == 1
        assert not websockets[0].closed
        assert heard == b"\x00\x00" * 480
    finally:
        await tts.aclose()

    assert websockets[0].closed is True
