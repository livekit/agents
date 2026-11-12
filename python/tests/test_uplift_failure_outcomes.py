"""Provider errors must remain failures at the public synthesis boundary."""

import pytest

from livekit.agents import APIConnectOptions, APIError
from livekit.plugins.upliftai.tts import TTS, WEBSOCKET_NAMESPACE, WebSocketClient

pytestmark = pytest.mark.unit


class _RejectingSocket:
    def __init__(self, client):
        self.client = client

    async def emit(self, event, data, *, namespace):
        assert event == "synthesize"
        assert namespace == WEBSOCKET_NAMESPACE
        await self.client._on_message(
            {"type": "error", "requestId": data["requestId"], "message": "voice unavailable"}
        )

    async def disconnect(self):
        await self.client._on_disconnect()


async def test_provider_rejection_fails_synthesis():
    provider = TTS(api_key="key", output_format="PCM_22050_16")
    client = WebSocketClient(provider._opts)
    client.connected = True
    client.sio = _RejectingSocket(client)
    provider._client = client
    try:
        with pytest.raises(APIError) as error:
            async with provider.synthesize(
                "hello", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                async for _ in stream:
                    pytest.fail("rejected synthesis produced audio")
        assert error.value.retryable is False
        assert isinstance(error.value.body, dict)
        assert error.value.body["message"] == "voice unavailable"
    finally:
        await provider.aclose()


async def test_malformed_socket_message_fails_pending_synthesis_without_waiting_for_timeout():
    import asyncio

    import socketio
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from pydantic import ValidationError

    server = socketio.AsyncServer(async_mode="aiohttp", ping_interval=0.01)
    app = web.Application()
    server.attach(app)

    @server.on("connect", namespace=WEBSOCKET_NAMESPACE)
    async def connected(sid, environ, auth):
        await server.emit("message", {"type": "ready"}, to=sid, namespace=WEBSOCKET_NAMESPACE)

    @server.on("synthesize", namespace=WEBSOCKET_NAMESPACE)
    async def synthesize(sid, data):
        await server.emit(
            "message",
            {"type": "audio", "requestId": data["requestId"], "audio": []},
            to=sid,
            namespace=WEBSOCKET_NAMESPACE,
        )

    async with TestServer(app) as peer:
        provider = TTS(api_key="key", base_url=str(peer.make_url("")), output_format="PCM_22050_16")
        try:
            async with asyncio.timeout(1):
                with pytest.raises(APIError) as failure:
                    async with provider.synthesize(
                        "hello", conn_options=APIConnectOptions(max_retry=0)
                    ) as stream:
                        async for _ in stream:
                            pytest.fail("malformed provider message produced audio")
            assert failure.value.retryable is False
            assert isinstance(failure.value.__cause__, ValidationError)
        finally:
            await provider.aclose()
            await server.shutdown()
            # Engine.IO's scheduled ping finishes after disconnect, even after shutdown().
            await asyncio.sleep(0.02)
