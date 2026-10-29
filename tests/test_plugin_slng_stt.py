from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.slng import stt as slng_stt

pytestmark = pytest.mark.plugin("slng")


@pytest.mark.parametrize("end_time", [0.4, 0.0, None])
async def test_transcript_timing_includes_connection_offset(monkeypatch, end_time):
    messages = asyncio.Queue()
    message = {"type": "final_transcript", "transcript": "hello"}
    if end_time is not None:
        message["words"] = [{"start": 0.1, "end": end_time}]
    messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(message), ""))
    ws = SimpleNamespace(
        receive=messages.get,
        send_str=AsyncMock(),
        send_bytes=AsyncMock(),
        close=AsyncMock(),
        closed=False,
    )
    monkeypatch.setattr(slng_stt.SpeechStream, "_connect_ws", AsyncMock(return_value=ws))
    provider = slng_stt.STT(
        api_key="test-key", model="deepgram/nova:3", http_session=SimpleNamespace()
    )
    stream = provider.stream()
    stream.start_time_offset = 7.0
    try:
        stream.push_frame(
            rtc.AudioFrame(
                data=bytes(3200), sample_rate=16000, num_channels=1, samples_per_channel=1600
            )
        )
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if end_time:
                        assert stream.start_time_offset > 6.0
                        assert event.alternatives[0].end_time == pytest.approx(
                            stream.start_time_offset + 0.4
                        )
                    else:
                        assert event.alternatives[0].end_time == 0.0
                    break
            else:
                pytest.fail("missing final transcript")
    finally:
        await stream.aclose()
        await provider.aclose()


async def _next_transcript_end_time(stream):
    async with asyncio.timeout(2):
        async for event in stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                return event.alternatives[0].end_time
    pytest.fail("missing final transcript")


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from .stt_reconnect import ReconnectClock, wait_until

    clock = ReconnectClock(monkeypatch, ready={"type": "ready"})
    provider = slng_stt.STT(api_key="test-key", model="deepgram/nova:3", http_session=clock)
    stream = provider.stream()
    stream.start_time_offset = 7.0
    try:
        await clock.push(stream, captured_at=1000.0)
        await wait_until(lambda: len(clock.sockets) == 1)
        stream.update_options(language="fr")
        await wait_until(lambda: clock.sockets[0].closed)
        await clock.push(stream, captured_at=1010.0)
        await wait_until(lambda: len(clock.sockets) == 2)
        clock.sockets[1].respond(
            {
                "type": "final_transcript",
                "transcript": "hello",
                "words": [{"start": 0.1, "end": 0.4}],
            }
        )
        assert await _next_transcript_end_time(stream) == pytest.approx(17.4)
    finally:
        await stream.aclose()
        await provider.aclose()


@pytest.mark.no_concurrent
@pytest.mark.parametrize("replay_failure", [None, "connect", "partial"])
async def test_replayed_audio_keeps_its_original_timestamp(monkeypatch, replay_failure):
    from .stt_reconnect import ReconnectClock, wait_until

    clock = ReconnectClock(monkeypatch, ready={"type": "ready"})
    provider = slng_stt.STT(
        api_key="test-key", model="deepgram/nova:3", buffer_size_seconds=0.1, http_session=clock
    )
    stream = provider.stream()
    stream.start_time_offset = 7.0
    from livekit.agents import APIConnectionError

    connect = stream._connect_ws
    attempts = 0

    async def reconnect(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2 and replay_failure == "connect":
            raise APIConnectionError("replacement connection failed")
        ws = await connect(**kwargs)
        if attempts == 2 and replay_failure == "partial":
            send = ws.send_bytes
            writes = 0

            async def send_prefix(payload):
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise APIConnectionError("replay interrupted")
                await send(payload)

            ws.send_bytes = send_prefix
        return ws

    monkeypatch.setattr(stream, "_connect_ws", reconnect)
    try:
        await clock.push(stream, captured_at=1000.0)
        await wait_until(lambda: len(clock.sockets) == 1)
        first = clock.sockets[0]
        await wait_until(lambda: sum(len(x) for x in first.sent if isinstance(x, bytes)) == 32000)
        first.respond(
            {
                "type": "final_transcript",
                "transcript": "one",
                "words": [{"start": 0.0, "end": 1.0}],
            }
        )
        assert await _next_transcript_end_time(stream) == pytest.approx(8.0)

        clock.now = 1001.2
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 3200))
        await wait_until(lambda: sum(len(x) for x in first.sent if isinstance(x, bytes)) == 38400)
        first.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))
        await wait_until(lambda: len(clock.sockets) == (3 if replay_failure == "partial" else 2))
        second = clock.sockets[-1]
        await wait_until(lambda: sum(len(x) for x in second.sent if isinstance(x, bytes)) == 6400)
        # This provider starts at zero on the replayed suffix, after the already final first turn.
        second.respond(
            {
                "type": "final_transcript",
                "transcript": "two",
                "words": [{"start": 0.0, "end": 0.2}],
            }
        )
        assert await _next_transcript_end_time(stream) == pytest.approx(8.2)
    finally:
        await stream.aclose()
        await provider.aclose()
