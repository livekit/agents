from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.plugin("elevenlabs")


class _ErrorSocket:
    """Sends one error frame, then stays open like a server that did not hang up."""

    def __init__(self, message_type: str) -> None:
        import aiohttp

        self.closed = False
        self._msg = aiohttp.WSMessage(
            aiohttp.WSMsgType.TEXT,
            json.dumps({"message_type": message_type, "message": "boom"}),
            None,
        )
        self._sent = False
        self._park = asyncio.Event()

    async def send_str(self, data: str) -> None:
        pass

    async def receive(self):
        if not self._sent:
            self._sent = True
            return self._msg
        await self._park.wait()

    async def close(self) -> None:
        self.closed = True
        self._park.set()


async def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("message_type", ["auth_error", "quota_exceeded", "transcriber_error"])
async def test_realtime_error_message_fails_the_connection(message_type):
    from livekit.plugins.elevenlabs import STT

    sockets: list[_ErrorSocket] = []
    errors: list[Any] = []

    class _Session:
        closed = False

        async def ws_connect(self, url, **kwargs):
            ws = _ErrorSocket(message_type)
            sockets.append(ws)
            return ws

    instance = STT(api_key="test-key", language_code="en", http_session=cast(Any, _Session()))
    instance.on("error", errors.append)
    stream = instance.stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01, timeout=1.0)
    )
    try:
        # _process_stream_event raises APIConnectionError for these frames; the
        # retry / "error" event only happens if recv_task lets that escape
        await _wait_until(lambda: len(sockets) > 1 or errors)
    finally:
        await stream.aclose()
