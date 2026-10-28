import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.rtzr import STT

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("payload", ["[]", '{"start_at": "invalid"}', "{"])
async def test_malformed_transcript_is_a_typed_connection_failure(
    monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    monkeypatch.setenv("RTZR_CLIENT_ID", "test")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "test")

    async def token(*, conn_options: APIConnectOptions) -> str:
        return "test"

    async def handle(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_str(payload)
        async for message in socket:
            if message.type == aiohttp.WSMsgType.TEXT and message.data == "EOS":
                break
        return socket

    app = web.Application()
    app.router.add_get("/v1/transcribe:streaming", handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        provider = STT(http_session=session)
        endpoint = str(server.make_url("/")).rstrip("/")
        monkeypatch.setattr(provider._client, "_ws_base", endpoint)
        monkeypatch.setattr(provider._client, "get_token", token)
        try:
            async with asyncio.timeout(3):
                with pytest.raises(APIConnectionError) as caught:
                    async with provider.stream(
                        conn_options=APIConnectOptions(max_retry=0)
                    ) as stream:
                        stream.push_frame(
                            rtc.AudioFrame(
                                data=b"\x00\x00" * 1600,
                                sample_rate=16000,
                                num_channels=1,
                                samples_per_channel=1600,
                            )
                        )
                        stream.end_input()
                        async for _ in stream:
                            pass
                assert caught.value.retryable is False
                assert isinstance(caught.value.__cause__, ValidationError)
        finally:
            await provider.aclose()
