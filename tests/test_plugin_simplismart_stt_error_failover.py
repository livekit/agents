"""Simplismart STT must raise on heartbeat ERROR so FallbackAdapter can fail over (#8031)."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectionError
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.simplismart import stt as simplismart_stt

pytestmark = pytest.mark.unit


class _ErrorSocket:
    def __init__(self) -> None:
        self.receives = 0
        self.closed = False

    def exception(self) -> BaseException:
        return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

    async def receive(self) -> aiohttp.WSMessage:
        self.receives += 1
        await asyncio.sleep(0)
        return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), None)

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_heartbeat_error_raises_instead_of_silent_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stt = simplismart_stt.STT(api_key="sk_test", streaming=True)
    stream = stt.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)

    async def fake_connect() -> Any:
        return _ErrorSocket()

    async def fake_send_config(ws: Any) -> None:
        return None

    monkeypatch.setattr(stream, "_connect_ws", fake_connect)
    monkeypatch.setattr(stream, "_send_initial_config", fake_send_config)

    # Keep send_task from parking on audio input forever: close input promptly.
    async def run_and_close() -> None:
        task = asyncio.create_task(stream._run())
        await asyncio.sleep(0.05)
        # If it reconnects silently, _run keeps going; we want the raise.
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except APIConnectionError:
            raise
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    import contextlib

    with pytest.raises(APIConnectionError, match="simplismart connection lost") as exc_info:
        await run_and_close()

    assert exc_info.value.reset_retry_budget is False
    assert not stream._reconnect_event.is_set()
    await stt.aclose()
