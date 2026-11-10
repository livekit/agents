from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
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

    async def receive(self) -> Any:
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
        assert plugin.capabilities.manual_flush
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


@pytest.mark.parametrize("failure", ["receiver", "input_read"])
async def test_unexpected_disconnect_retries_and_receivers_are_closed(monkeypatch, failure):
    async with _stream(monkeypatch, max_retry=1) as (_, stream, sockets):
        recv = stream._input_ch.recv

        async def receive_and_disconnect():
            data = await recv()
            if (
                isinstance(data, rtc.AudioFrame)
                and data.data.tobytes() == _frame(2, samples=400).data.tobytes()
            ):
                stream._failure.set_exception(APIConnectionError("Disconnected during input read"))
            return data

        if failure == "input_read":
            monkeypatch.setattr(stream._input_ch, "recv", receive_and_disconnect)
        stream.push_frame(_frame(1, samples=400))
        await _wait_until(lambda: sockets and sockets[0].sent)
        if failure == "receiver":
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


@pytest.mark.no_concurrent
@pytest.mark.parametrize("finalize_first", [False, True])
async def test_replay_keeps_original_audio_timeline(monkeypatch, finalize_first):
    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    async with _stream(monkeypatch, max_retry=1) as (_, stream, sockets):
        stream.start_time_offset = 7.0
        clock.now = 1001.0
        stream.push_frame(_frame(1, samples=8000))
        await _wait_until(lambda: sockets and stream._connection_audio_duration > 0.8)
        if finalize_first:
            stream.flush()
            await _wait_until(lambda: ("text", '{"type":"Finalize"}') in sockets[0].sent)
        clock.now = 1011.0
        stream.push_frame(_frame(2, samples=8000))
        await _wait_until(lambda: stream._connection_audio_duration > 1.8)
        await sockets[0]._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))
        await _wait_until(lambda: len(sockets) == 2 and sockets[1].sent)
        clock.now = 1021.0
        stream.push_frame(_frame(3, samples=8000))
        await sockets[1]._messages.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "start_at": 0,
                        "duration": 500,
                        "final": True,
                        "alternatives": [
                            {
                                "text": "replayed",
                                "words": [{"text": "replayed", "start_at": 0, "duration": 400}],
                            }
                        ],
                    }
                ),
            )
        )
        stream.end_input()
        events = [event async for event in stream]
        transcript = next(ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
        assert transcript.alternatives[0].start_time == pytest.approx(
            8.0 if finalize_first else 7.0
        )
        endpoint = next(ev for ev in events if ev.type == stt.SpeechEventType.END_OF_SPEECH)
        assert endpoint.alternatives[0].end_time == pytest.approx(8.4 if finalize_first else 7.4)


async def test_replay_buffer_is_bounded_without_truncated_retry(monkeypatch):
    monkeypatch.setattr(rtzr_stt, "_MAX_REPLAY_DURATION_SECONDS", 0.02)
    async with _stream(monkeypatch, max_retry=1) as (plugin, stream, sockets):
        metrics = []
        plugin.on("metrics_collected", metrics.append)
        first = _frame(1, samples=160)
        stream.push_frame(first)
        await _wait_until(lambda: sockets and sockets[0].sent)
        assert stream._segment_replayable
        stream.push_frame(_frame(2, samples=321))
        await _wait_until(lambda: not stream._segment_replayable and not stream._segment_audio)
        stream.flush()
        await _wait_until(lambda: ("text", '{"type":"Finalize"}') in sockets[0].sent)
        assert stream._segment_replayable
        assert b"".join(data for kind, data in sockets[0].sent if kind == "bytes") == (
            first.data.tobytes() + _frame(2, samples=321).data.tobytes()
        )
        stream.push_frame(_frame(1, samples=481))
        await _wait_until(lambda: not stream._segment_replayable and not stream._segment_audio)
        await sockets[0]._messages.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._task, 2)
        if stream._metrics_task:
            await stream._metrics_task
        assert sum(metric.audio_duration for metric in metrics) == pytest.approx(961 / 8000)
        assert len(sockets) == 1


async def test_idle_drain_serializes_new_audio(monkeypatch):
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


@pytest.mark.parametrize("duration", [300, 0, None])
def test_endpoint_does_not_use_audio_duration_without_word_timing(duration):
    stream = object.__new__(rtzr_stt.SpeechStream)
    stream._connection_offset = 7.0
    stream._opts = SimpleNamespace(language=LanguageCode("ko"))
    message = {"start_at": 100, "final": True, "alternatives": [{"text": "안녕"}]}
    if duration is not None:
        message["duration"] = duration
    events, speaking = stream._process_transcript_event(message, False)
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    assert not speaking
    assert endpoint.alternatives == []


def test_endpoint_uses_nested_word_timing_relative_to_the_utterance():
    stream = object.__new__(rtzr_stt.SpeechStream)
    stream._connection_offset = 7.0
    stream._opts = SimpleNamespace(language=LanguageCode("ko"))
    message = {
        "start_at": 38_505,
        "duration": 6_745,
        "final": True,
        "alternatives": [
            {
                "text": "안녕",
                "words": [{"text": "안녕", "start_at": 5_680, "duration": 280}],
            }
        ],
    }
    events, _ = stream._process_transcript_event(message, False)
    transcript, endpoint = events[-2:]
    assert transcript.alternatives[0].words[0].start_time == pytest.approx(7.0 + 44.185)
    assert transcript.alternatives[0].words[0].end_time == pytest.approx(7.0 + 44.465)
    assert endpoint.alternatives[0].end_time == pytest.approx(7.0 + 44.465)


@pytest.mark.no_concurrent
async def test_idle_reconnect_reanchors_after_old_final(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    import aiohttp

    from livekit import rtc

    from .stt_reconnect import ReconnectClock, wait_until

    clock = ReconnectClock(monkeypatch)
    client = SimpleNamespace(
        build_config=lambda **kwargs: {},
        connect_websocket=lambda config, **kwargs: clock.ws_connect(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(rtzr_stt, "RTZROpenAPIClient", lambda **kwargs: client)
    provider = rtzr_stt.STT(sample_rate=16000)
    stream = provider.stream()
    stream.start_time_offset = 7.0
    stream._idle_timeout = 0.02
    try:
        await clock.push(stream, captured_at=1000.0)
        await wait_until(lambda: len(clock.sockets) == 1)
        first = clock.sockets[0]
        await wait_until(lambda: "EOS" in first.sent)
        clock.now = 1010.04
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 640))
        await asyncio.sleep(0)
        assert stream.start_time_offset == pytest.approx(7.0)
        assert stream._input_ch.qsize() == 1
        first.respond(
            {
                "start_at": 0,
                "duration": 900,
                "final": True,
                "alternatives": [
                    {"text": "one", "words": [{"text": "one", "start_at": 0, "duration": 900}]}
                ],
            }
        )
        first.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.END_OF_SPEECH:
                    assert event.alternatives[0].end_time == pytest.approx(7.9)
                    break
        await wait_until(stream._input_ch.empty)
        clock.now = 1010.1
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 960))
        stream.flush()
        await wait_until(lambda: len(clock.sockets) == 2)
        assert stream.start_time_offset == pytest.approx(17.0)
        second = clock.sockets[1]
        await wait_until(lambda: '{"type":"Finalize"}' in second.sent)
        assert sum(len(x) for x in second.sent if isinstance(x, bytes)) == 3200
        second.respond(
            {
                "start_at": 0,
                "duration": 100,
                "final": True,
                "alternatives": [
                    {"text": "two", "words": [{"text": "two", "start_at": 0, "duration": 100}]}
                ],
            }
        )
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.END_OF_SPEECH:
                    assert event.alternatives[0].end_time == pytest.approx(17.1)
                    break
    finally:
        for ws in clock.sockets:
            await ws.close()
        await stream.aclose()
        await provider.aclose()


@pytest.mark.no_concurrent
async def test_end_input_sends_buffered_tail(monkeypatch):
    from unittest.mock import AsyncMock

    from livekit import rtc

    from .stt_reconnect import ReconnectClock, wait_until

    clock = ReconnectClock(monkeypatch)
    client = SimpleNamespace(
        build_config=lambda **kwargs: {},
        connect_websocket=lambda config, **kwargs: clock.ws_connect(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(rtzr_stt, "RTZROpenAPIClient", lambda **kwargs: client)
    provider = rtzr_stt.STT(sample_rate=16000)
    stream = provider.stream()
    stream.start_time_offset = 7.0
    try:
        clock.now = 1000.04
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 640))
        stream.end_input()
        await wait_until(lambda: clock.sockets and "EOS" in clock.sockets[0].sent)
        assert stream.start_time_offset == pytest.approx(7.0)
        assert sum(len(x) for x in clock.sockets[0].sent if isinstance(x, bytes)) == 1280
        await clock.sockets[0].close()
        await stream._task
    finally:
        await stream.aclose()
        await provider.aclose()


@pytest.fixture
async def gateway(monkeypatch):
    pongs = asyncio.Queue()

    async def handler(request):
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        await ws.ping(b"probe")
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.PONG:
                pongs.put_nowait(msg.data)
                await ws.send_json({})
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/transcribe:streaming", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = RTZROpenAPIClient(client_id="test", client_secret="test", http_session=session)
        client._ws_base = str(server.make_url("")).rstrip("/").replace("http://", "ws://")
        monkeypatch.setattr(client, "get_token", AsyncMock(return_value="test-token"))
        yield client, pongs


async def test_standalone_client_answers_server_ping(gateway):
    client, pongs = gateway
    ws = await client.connect_websocket({})
    try:
        msg = await asyncio.wait_for(ws.receive(), 1)
        assert msg.type == aiohttp.WSMsgType.TEXT
        assert pongs.get_nowait() == b"probe"
    finally:
        await ws.close()


async def test_stt_monitor_answers_server_ping(gateway, monkeypatch):
    client, pongs = gateway
    connect = AsyncMock(wraps=client.connect_websocket)
    monkeypatch.setattr(client, "connect_websocket", connect)
    monkeypatch.setattr(rtzr_stt, "RTZROpenAPIClient", lambda **kwargs: client)
    provider = rtzr_stt.STT()
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    ws = None
    try:
        ws = await stream._connect_ws()
        with pytest.raises(APIConnectionError, match="closed before EOS"):
            await asyncio.wait_for(stream._recv_loop(ws), 1)
        assert pongs.get_nowait() == b"probe"
        assert connect.call_args.kwargs["autoping"] is False
    finally:
        if ws is not None:
            await ws.close()
        await stream.aclose()
        await provider.aclose()
