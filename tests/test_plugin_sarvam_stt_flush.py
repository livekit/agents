from __future__ import annotations

import asyncio
import base64
import json

import aiohttp
import pytest

from livekit import rtc
from livekit.plugins import sarvam

pytestmark = pytest.mark.unit


class _WebSocket:
    close_code = 1000

    def __init__(self) -> None:
        self.closed = False
        self.sent: asyncio.Queue[dict] = asyncio.Queue()
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        message = json.loads(data)
        self.sent.put_nowait(message)
        if message.get("type") == "end_of_stream":
            await self.close()

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, 1000, "")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()

    async def next_message(self) -> dict:
        return await asyncio.wait_for(self.sent.get(), timeout=2)


class _Session:
    def __init__(self) -> None:
        self.closed = False
        self.ws = _WebSocket()
        self.connections = 0

    async def ws_connect(self, *args: object, **kwargs: object) -> _WebSocket:
        self.connections += 1
        return self.ws

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _Session:
    session = _Session()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: session)
    return session


def _frame(ms: int, value: int, sample_rate: int) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(bytes([value, 0]) * samples, sample_rate, 1, samples)


async def _expect_audio(ws: _WebSocket, frame: rtc.AudioFrame) -> None:
    received = b""
    while len(received) < len(frame.data.tobytes()):
        message = await ws.next_message()
        assert "type" not in message
        assert message["audio"]["sample_rate"] == frame.sample_rate
        assert message["audio"]["encoding"] == "pcm_s16le"
        chunk = base64.b64decode(message["audio"]["data"])
        assert chunk
        received += chunk
    assert received == frame.data.tobytes()


@pytest.mark.parametrize("flush_signal", [True, False, None])
@pytest.mark.parametrize("first_ms", [30, 75])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
async def test_flush_preserves_audio_and_accepts_next_segment(
    session: _Session, flush_signal: bool | None, first_ms: int, sample_rate: int
) -> None:
    async with sarvam.STT(
        api_key="test-key",
        flush_signal=flush_signal,
        sample_rate=sample_rate,
        input_audio_codec="pcm_s16le",
    ) as model:
        stream = model.stream()
        for value, ms in enumerate([first_ms, 50, 10], start=1):
            frame = _frame(ms, value, sample_rate)
            stream.push_frame(frame)
            stream.flush()
            await _expect_audio(session.ws, frame)
            if flush_signal:
                assert await session.ws.next_message() == {"type": "flush"}
            assert not session.ws.closed

        stream.end_input()
        if flush_signal:
            assert await session.ws.next_message() == {"type": "flush"}
        assert (await session.ws.next_message())["type"] == "end_of_stream"
        assert [event async for event in stream] == []
        assert session.ws.sent.empty()
        assert session.connections == 1


@pytest.mark.parametrize("tail_ms", [0, 10])
@pytest.mark.parametrize("flush_signal", [True, False, None])
async def test_end_input_drains_audio_before_terminating(
    session: _Session, tail_ms: int, flush_signal: bool | None
) -> None:
    async with sarvam.STT(
        api_key="test-key", flush_signal=flush_signal, input_audio_codec="pcm_s16le"
    ) as model:
        stream = model.stream()
        tail = _frame(tail_ms, 3, 16000)
        if tail_ms:
            stream.push_frame(tail)
        stream.end_input()
        if tail_ms:
            await _expect_audio(session.ws, tail)
        if flush_signal:
            assert await session.ws.next_message() == {"type": "flush"}
        assert (await session.ws.next_message())["type"] == "end_of_stream"
        assert [event async for event in stream] == []
        assert session.ws.sent.empty()


async def test_aclose_does_not_send_end_of_stream(session: _Session) -> None:
    async with sarvam.STT(api_key="test-key", input_audio_codec="pcm_s16le") as model:
        stream = model.stream()
        frame = _frame(50, 1, 16000)
        stream.push_frame(frame)
        await _expect_audio(session.ws, frame)
        await stream.aclose()
        assert session.ws.sent.empty()
        assert session.ws.closed


@pytest.mark.parametrize("flush_signal", [True, False, None])
@pytest.mark.parametrize("client_flushed", [True, False])
async def test_server_end_speech_does_not_flush_next_utterance(
    session: _Session, flush_signal: bool | None, client_flushed: bool
) -> None:
    async with sarvam.STT(
        api_key="test-key", flush_signal=flush_signal, input_audio_codec="pcm_s16le"
    ) as model:
        stream = model.stream()
        first = _frame(50, 1, 16000)
        stream.push_frame(first)
        await _expect_audio(session.ws, first)
        await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})
        if client_flushed:
            stream.flush()
            if flush_signal:
                assert await session.ws.next_message() == {"type": "flush"}
        await stream._handle_events({"data": {"signal_type": "END_SPEECH"}})

        for ms in [50, 10]:
            frame = _frame(ms, 2, 16000)
            stream.push_frame(frame)
            if ms == 10:
                stream.flush()
            await _expect_audio(session.ws, frame)
        if flush_signal:
            assert await session.ws.next_message() == {"type": "flush"}
        assert session.ws.sent.empty()
        assert not session.ws.closed
