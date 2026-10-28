import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.rtzr import STT

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("payload", ["[]", '{"start_at": "invalid"}', "{"])
async def test_malformed_transcript_is_a_typed_connection_failure(
    monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    monkeypatch.setenv("RTZR_CLIENT_ID", "test")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "test")

    async def handle(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_str(payload)
        await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/stream", handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        provider = STT(http_session=session)
        stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            async with session.ws_connect(server.make_url("/stream")) as socket:
                with pytest.raises(APIConnectionError) as caught:
                    await stream._recv_loop(socket)
                assert caught.value.retryable is False
                assert isinstance(caught.value.__cause__, ValidationError)
        finally:
            await stream.aclose()
            await provider.aclose()
