from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    utils,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable

DEFAULT_OPTIONS = APIConnectOptions(timeout=120)
SAMPLE_RATE = 24_000


def error(message: str) -> APIError:
    return APIError(message, retryable=False)


class Client:
    def __init__(
        self, api_key: str | None, base_url: str, http_session: aiohttp.ClientSession | None = None
    ) -> None:
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
        self._session = http_session

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

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
                if 300 <= response.status < 400:
                    raise error("Kitsch Labs API redirects are not supported.")
                if response.status != 200:
                    raise APIStatusError(
                        "Kitsch Labs API request failed.",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        retryable=response.status in {408, 429} or response.status >= 500,
                    )
                yield response
        except asyncio.TimeoutError:
            raise APITimeoutError("Kitsch Labs API request timed out.") from None
        except aiohttp.ClientError:
            raise APIConnectionError("Kitsch Labs API connection failed.") from None

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
                # aiohttp follows WS redirects but strips cross-origin Authorization.
                # Reject the handshake before sending any customer content.
                if ws._response.history:
                    raise error("Kitsch Labs API redirects are not supported.")
                # aiohttp omits its decoded-text overload on Python 3.10.
                yield cast("aiohttp.ClientWebSocketResponse", ws)
        except aiohttp.WSServerHandshakeError as exc:
            raise APIStatusError(
                "Kitsch Labs WebSocket handshake failed.",
                status_code=exc.status,
                retryable=exc.status in {408, 429} or exc.status >= 500,
            ) from None
        except asyncio.TimeoutError:
            raise APITimeoutError("Kitsch Labs WebSocket request timed out.") from None
        except aiohttp.ClientError:
            raise APIConnectionError("Kitsch Labs WebSocket connection failed.") from None


async def receive(ws: aiohttp.ClientWebSocketResponse, timeout: float) -> dict[str, Any]:
    message = await ws.receive(timeout=timeout)
    if message.type != aiohttp.WSMsgType.TEXT:
        raise APIConnectionError("Kitsch Labs stream closed before completion.")
    try:
        data = json.loads(message.data)
    except (ValueError, TypeError):
        raise error("Invalid Kitsch Labs stream response.") from None
    if not isinstance(data, dict):
        raise error("Invalid Kitsch Labs stream response.")
    return data


async def duplex(sender: Awaitable[None], receiver: Awaitable[None]) -> None:
    tasks = [asyncio.ensure_future(sender), asyncio.ensure_future(receiver)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
