"""Unit tests for the Inworld STT websocket connection."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _make_stream():
    """Construct an Inworld `SpeechStream` without spawning its real `_main_task`."""
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.inworld import STT
    from livekit.plugins.inworld.stt import SpeechStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    stt_instance = STT(api_key="test-key")
    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        return SpeechStream(stt=stt_instance, conn_options=DEFAULT_API_CONNECT_OPTIONS)


async def test_socket_is_opened_with_a_heartbeat() -> None:
    """aiohttp defaults `heartbeat` to None, so a socket that dies without a
    FIN/RST leaves the receive side parked forever: the stream never raises, and
    neither the reconnect nor a FallbackAdapter switch is ever triggered."""
    stream = _make_stream()
    captured: dict[str, Any] = {}

    class _FakeWebSocket:
        async def send_str(self, message: str) -> None:
            pass

    class _FakeSession:
        async def ws_connect(self, url: str, **kwargs: Any) -> _FakeWebSocket:
            captured.update(kwargs)
            return _FakeWebSocket()

    stream._ensure_session = lambda: _FakeSession()  # type: ignore[method-assign]

    await stream._connect_ws()

    assert captured.get("heartbeat") == 30.0
