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
