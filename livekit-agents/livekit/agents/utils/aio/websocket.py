from __future__ import annotations

import asyncio
import contextlib

import aiohttp

from ..._exceptions import APIConnectionError
from ...types import DEFAULT_API_CONNECT_OPTIONS

_CLOSE_TIMEOUT = 0.1


class WebSocketMonitor:
    """Receive WebSocket messages with an idle probe and a bounded silence timeout.

    The socket must be connected with ``autoping=False`` and without aiohttp's
    heartbeat. Incoming control frames are handled here so they count as activity.
    Use one monitor per active connection, including across cancelled receive calls.
    """

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        *,
        liveness_timeout: float = DEFAULT_API_CONNECT_OPTIONS.liveness_timeout,
    ) -> None:
        self._ws = ws
        self._liveness_timeout = liveness_timeout
        self._ping_interval = liveness_timeout / 3
        self._loop = asyncio.get_running_loop()
        self._last_received = self._loop.time()
        self._ping_sent = False

    async def receive(self) -> aiohttp.WSMessage:
        receiver = asyncio.create_task(self._ws.receive())
        ping: asyncio.Task[None] | None = None
        try:
            while True:
                deadline = self._last_received + (
                    self._liveness_timeout if self._ping_sent else self._ping_interval
                )
                tasks: list[asyncio.Task[aiohttp.WSMessage] | asyncio.Task[None]] = [receiver]
                if ping is not None:
                    tasks.append(ping)
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=max(0.0, deadline - self._loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receiver in done:
                    msg = receiver.result()
                    if msg.type in (
                        aiohttp.WSMsgType.TEXT,
                        aiohttp.WSMsgType.BINARY,
                        aiohttp.WSMsgType.PING,
                        aiohttp.WSMsgType.PONG,
                    ):
                        self._last_received = self._loop.time()
                        self._ping_sent = False
                        if ping is not None:
                            ping.cancel()
                            await asyncio.gather(ping, return_exceptions=True)
                            ping = None

                    if msg.type == aiohttp.WSMsgType.PING:
                        await asyncio.wait_for(self._ws.pong(msg.data), self._liveness_timeout)
                    elif msg.type != aiohttp.WSMsgType.PONG:
                        return msg
                    receiver = asyncio.create_task(self._ws.receive())
                    continue

                if ping is not None and ping in done:
                    ping.result()
                    ping = None
                    continue

                if self._loop.time() - self._last_received >= self._liveness_timeout:
                    raise aiohttp.ServerTimeoutError(
                        f"No WebSocket messages received for {self._liveness_timeout} seconds"
                    )
                self._ping_sent = True
                ping = asyncio.create_task(self._ws.ping())
        except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError) as exc:
            # A dead peer cannot complete a close handshake. Bound cleanup so the
            # original failure reaches retry/fallback without another close timeout.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._ws.close(), _CLOSE_TIMEOUT)
            raise APIConnectionError("WebSocket connection lost") from exc
        finally:
            receiver.cancel()
            if ping is not None:
                ping.cancel()
            await asyncio.gather(
                receiver, *([ping] if ping is not None else []), return_exceptions=True
            )

    def __aiter__(self) -> WebSocketMonitor:
        return self

    async def __anext__(self) -> aiohttp.WSMessage:
        msg = await self.receive()
        if msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise StopAsyncIteration
        return msg
