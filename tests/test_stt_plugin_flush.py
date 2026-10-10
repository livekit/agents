from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit import rtc

pytestmark = pytest.mark.unit


class _WebSocket:
    def __init__(self) -> None:
        self.sent: list[bytes | dict | str] = []
        self.closed = False
        self.close_code = 1000
        self._closed = asyncio.Event()
        self._response = SimpleNamespace(headers={}, status=101)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(bytes(data))

    async def send_str(self, data: str) -> None:
        try:
            self.sent.append(json.loads(data))
        except json.JSONDecodeError:
            self.sent.append(data)

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)

    async def close(self) -> None:
        self.closed = True
        self._closed.set()

    def audio_and_flushes(self, control: str) -> list[bytes | str]:
        result: list[bytes | str] = []
        for message in self.sent:
            if isinstance(message, bytes):
                result.append(message)
            elif isinstance(message, str):
                if message == control:
                    result.append(control)
            elif message.get("type", message.get("event")) == control:
                result.append(control)
            elif "audio" in message:
                audio = message["audio"]
                if isinstance(audio, dict) and audio.get("data"):
                    result.append(base64.b64decode(audio["data"]))
        return result


class _Session:
    closed = False

    def __init__(self) -> None:
        self.ws = _WebSocket()
        self.connections = 0

    async def ws_connect(self, *args: Any, **kwargs: Any) -> _WebSocket:
        self.connections += 1
        return self.ws

    async def close(self) -> None:
        self.closed = True


def _frame(ms: int, value: int) -> rtc.AudioFrame:
    samples = 16000 * ms // 1000
    return rtc.AudioFrame(bytes([value, 0]) * samples, 16000, 1, samples)


async def _wait_for_flushes(ws: _WebSocket, control: str, count: int) -> None:
    async def wait() -> None:
        while ws.audio_and_flushes(control).count(control) != count:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.parametrize("first_ms", [30, 50])
@pytest.mark.parametrize(
    ("provider", "class_name", "options", "control"),
    [
        ("deepgram", "STT", {}, "Finalize"),
        ("deepgram", "STTv2", {}, "ForceEndTurn"),
        ("assemblyai", "STT", {}, "ForceEndpoint"),
        ("cartesia", "STT", {"model": "ink-whisper"}, "finalize"),
        ("soniox", "STT", {}, "finalize"),
        ("sarvam", "STT", {"flush_signal": True}, "flush"),
        ("sarvam", "STTRealtime", {"endpointing": "manual"}, "speech_end"),
    ],
)
async def test_flush_keeps_connection_and_audio_order(
    provider: str, class_name: str, options: dict, control: str, first_ms: int, monkeypatch
) -> None:
    plugin = pytest.importorskip(f"livekit.plugins.{provider}")
    session = _Session()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: session)
    model = getattr(plugin, class_name)(api_key="test-key", http_session=session, **options)
    assert model.capabilities.manual_flush
    stream = model.stream()
    first, second = _frame(first_ms, 1), _frame(50, 2)
    try:
        stream.push_frame(first)
        stream.flush()
        await _wait_for_flushes(session.ws, control, 1)
        stream.push_frame(second)
        stream.flush()
        await _wait_for_flushes(session.ws, control, 2)

        messages = session.ws.audio_and_flushes(control)
        boundary = messages.index(control)
        assert b"".join(messages[:boundary]) == first.data.tobytes()
        assert b"".join(messages[boundary + 1 : -1]) == second.data.tobytes()
        assert messages[-1] == control
        assert session.connections == 1
        assert not session.ws.closed
        if provider == "sarvam" and class_name == "STT":
            assert "end_of_stream" not in session.ws.audio_and_flushes("end_of_stream")
            tail = _frame(10, 3)
            stream.push_frame(tail)
            stream.end_input()
            await _wait_for_flushes(session.ws, "end_of_stream", 1)
            messages = session.ws.audio_and_flushes(control)
            assert messages[-2:] == [tail.data.tobytes(), control]
    finally:
        await stream.aclose()
        await model.aclose()


@pytest.mark.parametrize(
    ("provider", "class_name", "options"),
    [
        ("cartesia", "STT", {"model": "ink-2"}),
        ("sarvam", "STT", {}),
        ("sarvam", "STTRealtime", {"endpointing": "vad"}),
        ("gladia", "STT", {}),
    ],
)
def test_modes_without_manual_flush_stay_disabled(provider, class_name, options) -> None:
    plugin = pytest.importorskip(f"livekit.plugins.{provider}")
    model = getattr(plugin, class_name)(api_key="test-key", **options)
    assert not model.capabilities.manual_flush
