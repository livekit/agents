from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.plugins.rtzr import stt as rtzr_stt
from livekit.plugins.rtzr.rtzrapi import RTZROpenAPIClient, RTZRStatusError

pytestmark = pytest.mark.unit


def _frame(value: int, *, samples: int = 160, sample_rate: int = 8000) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes([value, value]) * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _FakeWebSocket:
    def __init__(self, *, fail: str | None = None) -> None:
        self.sent: list[tuple[str, bytes | str]] = []
        self.closed = False
        self._messages: asyncio.Queue[Any] = asyncio.Queue()
        self._fail = fail
        self._byte_sends = 0

    async def send_bytes(self, data: bytes) -> None:
        self._byte_sends += 1
        if self._fail == "audio" and self._byte_sends == 2:
            raise aiohttp.ClientError("send failed")
        self.sent.append(("bytes", data))

    async def send_str(self, data: str) -> None:
        if self._fail == "finalize" and data == '{"type":"Finalize"}':
            raise aiohttp.ClientError("send failed")
        self.sent.append(("text", data))
        if data == "EOS":
            await self._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> Any:
        return await self._messages.get()

    async def close(self) -> None:
        self.closed = True


async def test_expiring_token_refresh_is_shared_by_concurrent_callers(monkeypatch) -> None:
    client = RTZROpenAPIClient(client_id="client-id", client_secret="client-secret")
    monkeypatch.setattr("livekit.plugins.rtzr.rtzrapi.time.time", lambda: 10000)
    client._token = {"access_token": "old", "expire_at": 11000}
    refresh_count = 0

    async def refresh_token() -> None:
        nonlocal refresh_count
        refresh_count += 1
        await asyncio.sleep(0)
        client._token = {"access_token": "token", "expire_at": 10**12}

    client._refresh_token = refresh_token  # type: ignore[method-assign]

    assert await asyncio.gather(client.get_token(), client.get_token()) == ["token", "token"]
    assert refresh_count == 1


@asynccontextmanager
async def _stream(monkeypatch, *, max_retry=0, language="ko", fail_first=None):
    monkeypatch.setenv("RTZR_CLIENT_ID", "client-id")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "client-secret")
    plugin = rtzr_stt.STT(model="whisper")
    stream = plugin.stream(
        language=language,
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0),
    )
    sockets = []

    async def connect():
        ws = _FakeWebSocket(fail=fail_first if not sockets else None)
        sockets.append(ws)
        return ws

    monkeypatch.setattr(stream, "_connect_ws", connect)
    try:
        yield plugin, stream, sockets
    finally:
        await stream.aclose()
        await plugin.aclose()


async def _wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


async def test_empty_and_repeated_flush(monkeypatch):
    async with _stream(monkeypatch) as (_, stream, sockets):
        stream.flush()
        await asyncio.sleep(0)
        assert not sockets
        stream.push_frame(_frame(1, samples=1))
        stream.flush()
        stream.flush()
        stream.end_input()
        events = [event async for event in stream]
        assert sockets[0].sent == [
            ("bytes", b"\x01\x01"),
            ("text", '{"type":"Finalize"}'),
            ("text", "EOS"),
        ]
        usage = [ev.recognition_usage.audio_duration for ev in events if ev.recognition_usage]
        assert usage == [1 / 8000]


async def test_finalize_preserves_audio_transcripts_and_usage(monkeypatch):
    async with _stream(monkeypatch, language="en") as (plugin, stream, sockets):
        frames = [_frame(1, samples=161), _frame(2, samples=1)]
        for index, frame in enumerate(frames):
            stream.push_frame(frame)
            stream.flush()
            await _wait_until(
                lambda index=index: (
                    sockets and sum(kind == "text" for kind, _ in sockets[0].sent) == index + 1
                )
            )
            for final in (False, True):
                await sockets[0]._messages.put(
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data=json.dumps({"alternatives": [{"text": str(index)}], "final": final}),
                    )
                )
        stream.end_input()
        events = [event async for event in stream]
        assert len(sockets) == 1 and sockets[0].closed
        assert b"".join(data for kind, data in sockets[0].sent if kind == "bytes") == (
            b"".join(frame.data.tobytes() for frame in frames)
        )
        assert [data for kind, data in sockets[0].sent if kind == "text"] == [
            '{"type":"Finalize"}',
            '{"type":"Finalize"}',
            "EOS",
        ]
        transcripts = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
        assert [ev.alternatives[0].text for ev in transcripts] == ["0", "1"]
        assert all(ev.alternatives[0].language == "en" for ev in transcripts)
        assert plugin._params.language == "ko"
        assert sum(ev.type == stt.SpeechEventType.END_OF_SPEECH for ev in events) == 2
        assert [ev.recognition_usage.audio_duration for ev in events if ev.recognition_usage] == (
            pytest.approx([161 / 8000, 1 / 8000])
        )


@pytest.mark.parametrize("payload", ["invalid json", "[]", '{"error":"private transcript"}'])
async def test_receive_failure_reaches_consumer(monkeypatch, payload):
    async with _stream(monkeypatch) as (_, stream, sockets):
        stream.push_frame(_frame(1, samples=400))
        await _wait_until(lambda: sockets)
        await sockets[0]._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=payload))
        with pytest.raises((APIConnectionError, APIStatusError)) as caught:
            await asyncio.wait_for(stream._task, 2)
        assert "private transcript" not in str(caught.value)
        assert sockets[0].closed


async def test_unexpected_disconnect_retries_and_receivers_are_closed(monkeypatch):
    async with _stream(monkeypatch, max_retry=1) as (_, stream, sockets):
        stream.push_frame(_frame(1, samples=400))
        await _wait_until(lambda: sockets and sockets[0].sent)
        await sockets[0]._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))
        await _wait_until(lambda: sockets[0].closed)
        stream.push_frame(_frame(2, samples=400))
        stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        assert len(sockets) == 2
        assert all(ws.closed for ws in sockets)
        assert b"".join(data for kind, data in sockets[1].sent if kind == "bytes") == (
            _frame(1, samples=400).data.tobytes() + _frame(2, samples=400).data.tobytes()
        )


async def test_replay_buffer_is_bounded_without_truncated_retry(monkeypatch):
    monkeypatch.setattr(rtzr_stt, "_MAX_REPLAY_DURATION_SECONDS", 0.02)
    async with _stream(monkeypatch, max_retry=1) as (_, stream, sockets):
        stream.push_frame(_frame(1, samples=481))
        await _wait_until(lambda: sockets and not stream._segment_audio)
        assert not stream._segment_replayable
        await sockets[0]._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._task, 2)
        assert len(sockets) == 1


async def test_idle_drain_serializes_new_audio(monkeypatch):
    monkeypatch.setattr(rtzr_stt, "_IDLE_CHECK_INTERVAL", 0.001)
    async with _stream(monkeypatch) as (_, stream, sockets):
        stream._idle_timeout = 0.001
        stream.push_frame(_frame(1, samples=400))
        await _wait_until(lambda: sockets)
        ws = sockets[0]
        drain = asyncio.Event()
        original_send = ws.send_str

        async def send(control):
            if control == "EOS":
                ws.sent.append(("text", control))
                await drain.wait()
                await ws._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))
            else:
                await original_send(control)

        ws.send_str = send
        await _wait_until(lambda: ("text", "EOS") in ws.sent)
        stream.push_frame(_frame(2, samples=401))
        stream.end_input()
        await asyncio.sleep(0)
        assert len(sockets) == 1
        drain.set()
        await asyncio.wait_for(stream._task, 2)
        assert len(sockets) == 2
        assert all(ws.closed for ws in sockets)
        assert (
            b"".join(data for kind, data in sockets[1].sent if kind == "bytes")
            == _frame(2, samples=401).data.tobytes()
        )


async def test_close_cancels_receiver_without_waiting_for_eos(monkeypatch):
    async with _stream(monkeypatch) as (_, stream, sockets):
        stream.push_frame(_frame(1, samples=400))
        await _wait_until(lambda: sockets)
        recv_task = stream._recv_task
        await asyncio.wait_for(stream.aclose(), 0.5)
        assert sockets[0].closed
        assert recv_task.done()
        assert not any(kind == "text" for kind, _ in sockets[0].sent)


async def test_drain_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(rtzr_stt, "_RECV_COMPLETION_TIMEOUT", 0.01)
    async with _stream(monkeypatch, max_retry=1) as (_, stream, sockets):
        stream.push_frame(_frame(1, samples=400))
        await _wait_until(lambda: sockets)
        sockets[0].send_str = AsyncMock()
        stream.end_input()
        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(stream._task, 1)
        assert sockets[0].closed
        assert stream._recv_task is None


@pytest.mark.parametrize("operation", ["audio", "finalize"])
async def test_send_failure_replays_complete_segment_without_duplicate_usage(
    monkeypatch, operation
):
    async with _stream(monkeypatch, max_retry=1, fail_first=operation) as (
        plugin,
        stream,
        sockets,
    ):
        metrics = []
        plugin.on("metrics_collected", metrics.append)
        frame = _frame(1, samples=4001)
        stream.push_frame(frame)
        stream.end_input()
        await asyncio.wait_for(stream._task, 2)
        if stream._metrics_task:
            await stream._metrics_task
        assert len(sockets) == 2 and all(ws.closed for ws in sockets)
        assert b"".join(data for kind, data in sockets[1].sent if kind == "bytes") == (
            frame.data.tobytes()
        )
        assert [data for kind, data in sockets[1].sent if kind == "text"] == [
            '{"type":"Finalize"}',
            "EOS",
        ]
        assert sum(metric.audio_duration for metric in metrics) == pytest.approx(frame.duration)


@pytest.mark.parametrize("statuses,expected_calls", [([401, 200], 2), ([401, 401], 2), ([403], 1)])
async def test_websocket_auth_retry(monkeypatch, statuses, expected_calls):
    session = SimpleNamespace(ws_connect=AsyncMock())
    client = RTZROpenAPIClient(client_id="id", client_secret="secret", http_session=session)
    refresh = AsyncMock()

    async def get_token():
        if client._token is None:
            await refresh()
            client._token = {"access_token": f"token-{refresh.await_count}", "expire_at": 10**12}
        return client._token["access_token"]

    monkeypatch.setattr(client, "get_token", get_token)
    ws = _FakeWebSocket()
    session.ws_connect.side_effect = [
        ws
        if status == 200
        else aiohttp.WSServerHandshakeError(None, (), status=status, message="secret")
        for status in statuses
    ]
    if statuses[-1] == 200:
        assert await client.connect_websocket({}) is ws
    else:
        with pytest.raises(RTZRStatusError) as caught:
            await client.connect_websocket({})
        assert "secret" not in str(caught.value)
    assert session.ws_connect.await_count == expected_calls
    assert refresh.await_count == expected_calls
    await client.close()
