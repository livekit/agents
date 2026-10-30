from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, LanguageCode, stt
from livekit.plugins.rtzr import stt as rtzr_stt
from livekit.plugins.rtzr.rtzrapi import RTZROpenAPIClient

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("duration", [300, 0, None])
def test_endpoint_does_not_use_audio_duration_without_word_timing(duration):
    stream = object.__new__(rtzr_stt.SpeechStream)
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._rtzr_stt = SimpleNamespace(_params=SimpleNamespace(language=LanguageCode("ko")))
    message = {"start_at": 100, "final": True, "alternatives": [{"text": "안녕"}]}
    if duration is not None:
        message["duration"] = duration
    events, speaking, started_at = stream._process_transcript_event(message, False, None)
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    assert not speaking
    assert started_at is None
    assert endpoint.alternatives == []


def test_endpoint_uses_nested_word_timing_relative_to_the_utterance():
    stream = object.__new__(rtzr_stt.SpeechStream)
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._rtzr_stt = SimpleNamespace(_params=SimpleNamespace(language=LanguageCode("ko")))
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
    events, _, _ = stream._process_transcript_event(message, False, None)
    transcript, endpoint = events[-2:]
    assert transcript.alternatives[0].words[0].start_time == pytest.approx(7.0 + 44.185)
    assert transcript.alternatives[0].words[0].end_time == pytest.approx(7.0 + 44.465)
    assert endpoint.alternatives[0].end_time == pytest.approx(7.0 + 44.465)


@pytest.mark.no_concurrent
@pytest.mark.parametrize("reason", ["flush", "idle"])
async def test_internal_reconnect_reanchors_after_old_final(monkeypatch, reason):
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
    if reason == "idle":
        stream._idle_timeout = 0.02
    try:
        await clock.push(stream, captured_at=1000.0)
        await wait_until(lambda: len(clock.sockets) == 1)
        first = clock.sockets[0]
        if reason == "flush":
            stream.flush()
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
        await wait_until(lambda: len(clock.sockets) == 2)
        assert stream.start_time_offset == pytest.approx(17.0)
        second = clock.sockets[1]
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
        await asyncio.wait_for(stream._recv_loop(ws), 1)
        assert pongs.get_nowait() == b"probe"
        assert connect.call_args.kwargs["autoping"] is False
    finally:
        if ws is not None:
            await ws.close()
        await stream.aclose()
        await provider.aclose()
