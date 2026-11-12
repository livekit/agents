"""Streaming STT timestamps stay on the captured-audio timeline across a reconnect."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, http, stt
from livekit.plugins import deepgram


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/7796 reproduces: a reconnected socket's "
    "timestamps restart at the old offset (internal reconnect) or advance by wall time (retry)",
)
@pytest.mark.parametrize("reconnect", ["update_options", "connection_dropped"])
async def test_a_reconnected_stream_continues_the_audio_timeline(reconnect: str) -> None:
    """Speech after a reconnect is timed after the audio the first socket already heard.

    https://github.com/livekit/agents/pull/7796 https://github.com/livekit/agents/pull/7795
    """
    sockets: list[web.WebSocketResponse] = []
    second_socket = asyncio.Event()

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sockets.append(ws)
        first = len(sockets) == 1
        if not first:
            second_socket.set()
        received = 0
        async for message in ws:
            if message.type != web.WSMsgType.BINARY:
                continue
            received += len(message.data)
            if first and received == 32_000:  # 1.0 s of 16 kHz audio heard
                await ws.send_json(
                    {
                        "type": "Results",
                        "is_final": True,
                        "speech_final": True,
                        "metadata": {"request_id": "r1"},
                        "channel": {
                            "alternatives": [
                                {
                                    "transcript": "first",
                                    "confidence": 1,
                                    "words": [{"word": "first", "start": 0.75, "end": 1.0}],
                                }
                            ]
                        },
                    }
                )
                if reconnect == "connection_dropped":
                    await ws.close(code=1011)
            elif not first and received == 16_000:  # 0.5 s heard by the new socket
                await ws.send_json(
                    {
                        "type": "Results",
                        "is_final": True,
                        "speech_final": True,
                        "metadata": {"request_id": "r2"},
                        "channel": {
                            "alternatives": [
                                {
                                    "transcript": "again",
                                    "confidence": 1,
                                    "words": [{"word": "again", "start": 0.0, "end": 0.25}],
                                }
                            ]
                        },
                    }
                )
        return ws

    app = web.Application()
    app.router.add_get("/listen", peer)
    async with TestServer(app) as server, http.open() as session:
        model = deepgram.STT(
            api_key="test",
            sample_rate=16_000,
            http_session=session,
            base_url=str(server.make_url("/listen")),
        )
        options = APIConnectOptions(max_retry=1, retry_interval=0.3)
        finals: list[stt.SpeechData] = []
        try:
            async with model.stream(conn_options=options) as stream:
                async with asyncio.timeout(5):
                    for _ in range(100):
                        stream.push_frame(rtc.AudioFrame(b"\0\0" * 160, 16_000, 1, 160))
                    async for event in stream:
                        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                            finals.append(event.alternatives[0])
                            break
                    if reconnect == "update_options":
                        stream.update_options(language="fr")
                    await second_socket.wait()
                    for _ in range(50):
                        stream.push_frame(rtc.AudioFrame(b"\0\0" * 160, 16_000, 1, 160))
                    async for event in stream:
                        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                            finals.append(event.alternatives[0])
                            break
        finally:
            await model.aclose()

    before, after = finals
    assert (before.text, after.text) == ("first", "again")
    assert after.start_time == pytest.approx(before.end_time, abs=0.05)
    assert after.end_time == pytest.approx(before.end_time + 0.25, abs=0.05)
