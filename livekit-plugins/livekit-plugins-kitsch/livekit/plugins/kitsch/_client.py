from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import APIConnectOptions, APIError, APIStatusError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

DEFAULT_OPTIONS = APIConnectOptions(max_retry=0, timeout=120)
SAMPLE_RATE = 24_000


def error(message: str) -> APIError:
    return APIError(message, retryable=False)


async def _reject_redirect(
    request: aiohttp.ClientRequest,
    handler: Callable[[aiohttp.ClientRequest], Awaitable[aiohttp.ClientResponse]],
) -> aiohttp.ClientResponse:
    response = await handler(request)
    if 300 <= response.status < 400:
        response.release()
        raise error("Kitsch API redirects are not supported.")
    return response


class Client:
    def __init__(self, api_key: str | None, base_url: str) -> None:
        key = api_key if api_key is not None else os.environ.get("KITSCH_API_KEY")
        if not key or not key.strip() or "\n" in key or "\r" in key:
            raise ValueError("Set KITSCH_API_KEY or provide a valid api_key.")
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
            or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ValueError("base_url must be an HTTPS origin (HTTP for loopback only).")
        self.origin = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}"}
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(middlewares=(_reject_redirect,))
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    @asynccontextmanager
    async def post(
        self, path: str, options: APIConnectOptions, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        try:
            async with self.session.post(
                self.origin + path,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=options.timeout),
                allow_redirects=False,
                **kwargs,
            ) as response:
                if response.status != 200:
                    raise APIStatusError(
                        "Kitsch API request failed.",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        retryable=False,
                    )
                yield response
        except (asyncio.TimeoutError, aiohttp.ClientError):
            raise error("Kitsch API connection failed or timed out.") from None

    @asynccontextmanager
    async def websocket(
        self, path: str, options: APIConnectOptions, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientWebSocketResponse, None]:
        url = self.origin.replace("http", "ws", 1) + path
        try:
            ws = await asyncio.wait_for(
                self.session.ws_connect(
                    url,
                    headers=self.headers,
                    max_msg_size=2 * 1024 * 1024,
                    **kwargs,
                ),
                timeout=options.timeout,
            )
            async with ws:
                yield ws
        except aiohttp.WSServerHandshakeError as exc:
            raise APIStatusError(
                "Kitsch WebSocket handshake failed.",
                status_code=exc.status,
                retryable=False,
            ) from None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            raise error("Kitsch WebSocket connection failed or timed out.") from None


async def receive(ws: aiohttp.ClientWebSocketResponse, timeout: float) -> dict[str, Any]:
    message = await ws.receive(timeout=timeout)
    if message.type != aiohttp.WSMsgType.TEXT:
        raise error("Kitsch stream closed before completion.")
    try:
        data = json.loads(message.data)
    except (ValueError, TypeError):
        raise error("Invalid Kitsch stream response.") from None
    if not isinstance(data, dict):
        raise error("Invalid Kitsch stream response.")
    return data


async def duplex(sender: Awaitable[None], receiver: Awaitable[None]) -> None:
    tasks = [asyncio.ensure_future(sender), asyncio.ensure_future(receiver)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
