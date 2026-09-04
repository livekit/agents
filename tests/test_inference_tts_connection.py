from __future__ import annotations

import aiohttp
import pytest

from livekit.agents.inference import TTS

pytestmark = pytest.mark.unit


async def test_dial_includes_model() -> None:
    request_url = ""

    class FakeWebSocket:
        closed = False

        async def send_str(self, _data: str) -> None:
            pass

    class FakeSession:
        async def ws_connect(self, url: str, *, headers: dict[str, str]) -> FakeWebSocket:
            nonlocal request_url
            request_url = url
            return FakeWebSocket()

    tts = TTS(
        model="cartesia/sonic-3",
        voice="test-voice",
        api_key="test-key",
        api_secret="test-secret",
        base_url="http://127.0.0.1:1234",
        http_session=FakeSession(),  # type: ignore[arg-type]
    )

    connection = await tts._connect_ws(timeout=1.0)

    assert aiohttp.client_reqrep.URL(request_url).query.get("model") == "cartesia/sonic-3"
    assert connection.ws is not None
