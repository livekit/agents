"""Streaming STT over a websocket that goes half-open after the handshake."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, http, stt
from livekit.plugins import deepgram, elevenlabs, inworld, openai, telnyx
from livekit.plugins.openai import stt as openai_stt

HEARTBEAT = 0.05  # the heartbeat a plugin asks for, scaled down from its ~45 s


@pytest.mark.parametrize(
    "provider_name",
    [
        "openai",
        "elevenlabs",
        "inworld",
        "deepgram",
        "telnyx",
    ],
)
async def test_a_silent_socket_fails_the_stream_with_a_connection_error(
    provider_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A socket that stops answering after the handshake fails the stream within its heartbeat.

    https://github.com/livekit/agents/issues/7900 https://github.com/livekit/agents/issues/7358
    """
    accepted = asyncio.Event()
    hold = asyncio.Event()

    async def blackhole(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoping=False)  # never answers a ping
        await ws.prepare(request)
        accepted.set()
        await hold.wait()  # never reads, never sends
        return ws

    connect = http.HttpSession.ws_connect

    # a plugin that asks for no heartbeat still gets none
    # Any: ws_connect's keyword options, forwarded untouched
    async def scaled(self: http.HttpSession, url: str, **kwargs: Any) -> http.WebSocket:
        if kwargs.get("heartbeat") is not None:
            kwargs["heartbeat"] = HEARTBEAT
        return await connect(self, url, **kwargs)

    monkeypatch.setattr(http.HttpSession, "ws_connect", scaled)
    # a socket protocol's pings are the framework's
    for socket_stream in (deepgram.SpeechStream, elevenlabs.SpeechStream, inworld.SpeechStream):
        monkeypatch.setattr(socket_stream, "ping_interval", HEARTBEAT)
    monkeypatch.setattr(telnyx.stt.SpeechStream, "ping_interval", HEARTBEAT)
    monkeypatch.setattr(openai_stt.SpeechStream, "ping_interval", HEARTBEAT)

    app = web.Application()
    app.router.add_route("GET", "/{tail:.*}", blackhole)
    async with TestServer(app) as server, http.open() as session:
        url = str(server.make_url("")).rstrip("/")
        provider: stt.STT
        if provider_name == "openai":
            provider = openai.STT(api_key="key", base_url=url, use_realtime=True)
        elif provider_name == "elevenlabs":
            provider = elevenlabs.STT(
                api_key="key", model="scribe_v2_realtime", base_url=url, http_session=session
            )
        elif provider_name == "inworld":
            provider = inworld.STT(api_key="key", base_url=url, http_session=session)
        elif provider_name == "telnyx":
            provider = telnyx.STT(api_key="key", base_url=url)
        else:
            provider = deepgram.STT(api_key="key", base_url=url, http_session=session)
        frame = rtc.AudioFrame(
            data=bytes(320), sample_rate=16000, num_channels=1, samples_per_channel=160
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_frame(frame)
                await asyncio.wait_for(accepted.wait(), 2)
                with pytest.raises(APIConnectionError):
                    async with asyncio.timeout(10 * HEARTBEAT):
                        async for _ in stream:
                            pass
        finally:
            hold.set()
            await provider.aclose()
