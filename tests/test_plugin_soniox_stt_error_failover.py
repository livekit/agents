"""Soniox STT must raise on heartbeat ERROR so FallbackAdapter can fail over (#8031)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import aiohttp
import pytest

from livekit.agents import APIConnectionError
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

pytestmark = pytest.mark.plugin("soniox")


class _ErrorSocket:
    """aiohttp delivers heartbeat timeouts as ERROR rather than a close frame."""

    def __init__(self) -> None:
        self.receives = 0

    def exception(self) -> BaseException:
        return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.receives += 1
        await asyncio.sleep(0)
        if self.receives == 1:
            return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), None)
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_heartbeat_error_raises_instead_of_silent_reconnect() -> None:
    from livekit.plugins.soniox import STT
    from livekit.plugins.soniox.stt import SpeechStream

    stt = STT(api_key="test-key")
    with patch("asyncio.create_task"):
        stream = SpeechStream(stt=stt, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    stream._ws = _ErrorSocket()  # type: ignore[assignment]

    with pytest.raises(APIConnectionError, match="soniox connection lost") as exc_info:
        await stream._recv_messages_task()

    assert exc_info.value.reset_retry_budget is False

    # Must not arm the plugin-local reconnect loop (#8031).
    assert not stream._reconnect_event.is_set()
