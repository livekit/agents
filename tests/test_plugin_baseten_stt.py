"""Unit tests for the Baseten STT websocket connection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


async def test_socket_is_opened_with_a_heartbeat() -> None:
    """Without a heartbeat aiohttp never notices a socket that died without a FIN/RST.

    The five-second `wait_for` in the receive loop is a polling timeout, not a
    liveness bound: it fires, hits `continue`, and the loop carries on, so nothing
    escalates the dead socket. Measured against a server that accepts the upgrade
    and then answers nothing, this plugin sat for 75s without noticing.
    """
    from livekit.plugins.baseten.stt import SpeechStream

    captured: dict[str, Any] = {}

    class _FakeWebSocket:
        async def send_str(self, message: str) -> None:
            pass

        async def send_bytes(self, data: bytes) -> None:
            pass

    async def _ws_connect(url: str, **kwargs: Any) -> _FakeWebSocket:
        captured.update(kwargs)
        return _FakeWebSocket()

    stream = object.__new__(SpeechStream)
    stream._model_endpoint = "wss://model.baseten.co/v1/websocket"
    stream._api_key = "test-key"
    stream._session = SimpleNamespace(ws_connect=_ws_connect)
    stream._opts = SimpleNamespace(language="en", sample_rate=16000, model="whisper")

    import contextlib

    # the metadata handshake after the upgrade is not what is under test, and a fake
    # socket cannot answer it; the kwargs reaching ws_connect are already recorded
    with contextlib.suppress(Exception):
        await stream._connect_ws()

    assert captured.get("heartbeat") == 30.0
