"""Unit tests for the SLNG STT websocket connection."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


async def test_socket_is_opened_with_a_heartbeat() -> None:
    """aiohttp defaults `heartbeat` to None, so a socket that died without a
    FIN/RST leaves the receive side parked forever: nothing reports a failure, so
    the stream neither reconnects nor lets a FallbackAdapter move on.

    `SpeechStream.__init__` takes nine keyword-only arguments that `_connect_ws`
    never reads, so the stream is built bare and given only what the connect path
    touches.
    """
    from livekit.plugins.slng.stt import SpeechStream

    captured: dict[str, Any] = {}

    class _FakeWebSocket:
        async def send_str(self, message: str) -> None:
            pass

    async def _ws_connect(url: str, **kwargs: Any) -> _FakeWebSocket:
        captured.update(kwargs)
        return _FakeWebSocket()

    stream = object.__new__(SpeechStream)
    stream._api_token = "test-key"
    stream._extra_headers = {}
    # `init` short-circuits the payload builder, which needs the full opts object
    stream._connections = [SimpleNamespace(headers={}, init={"type": "start"})]
    stream._active_endpoint_index = 0
    stream._conn_options = SimpleNamespace(timeout=5.0)
    stream._session = SimpleNamespace(ws_connect=_ws_connect)

    # the init exchange after the handshake is not what is under test, and a fake
    # socket cannot answer it; the kwargs reaching ws_connect are already recorded
    with contextlib.suppress(Exception):
        await stream._connect_ws(model_endpoint="wss://example.test/v1", model="deepgram/nova:3")

    assert captured.get("heartbeat") == 30.0
