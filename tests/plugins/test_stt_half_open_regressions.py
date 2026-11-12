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

_UNDETECTED = pytest.mark.xfail(
    strict=True,
    raises=TimeoutError,
    reason="https://github.com/livekit/agents/issues/7900 reproduces: no heartbeat, "
    "receive() waits forever on a silent socket",
)


@pytest.mark.parametrize(
    "provider_name",
    [
        pytest.param("openai", marks=_UNDETECTED),
        pytest.param("elevenlabs", marks=_UNDETECTED),
        pytest.param("inworld", marks=_UNDETECTED),
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

    # the heartbeat the plugin asks for, scaled down so the test waits ms rather than ~45 s; a
    # plugin that asks for none still gets none
    # Any: ws_connect's keyword options, forwarded untouched
    async def scaled(self: http.HttpSession, url: str, **kwargs: Any) -> http.WebSocket:
        if kwargs.get("heartbeat") is not None:
            kwargs["heartbeat"] = 0.2
        return await connect(self, url, **kwargs)

    monkeypatch.setattr(http.HttpSession, "ws_connect", scaled)

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
            provider = telnyx.STT(api_key="key", base_url=url, http_session=session)
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
                    async with asyncio.timeout(3):
                        async for _ in stream:
                            pass
        finally:
            hold.set()
            await provider.aclose()
