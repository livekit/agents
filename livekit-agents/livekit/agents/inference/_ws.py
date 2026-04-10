from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import TracebackType
from typing import overload

import aiohttp

from .._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
)
from ..log import logger
from ._utils import create_access_token, get_inference_headers

_WS_CLOSE_TYPES = frozenset(
    {
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSING,
    }
)

_PAYLOAD_TO_WS_TYPE: dict[type[str] | type[bytes], aiohttp.WSMsgType] = {
    str: aiohttp.WSMsgType.TEXT,
    bytes: aiohttp.WSMsgType.BINARY,
}


class InferenceWebSocket:
    """Context manager that connects to a LiveKit inference WebSocket endpoint.

    Handles URL scheme conversion (http->ws), authentication, connection timeout,
    error wrapping, and provides recv iterators with automatic close detection.

    Usage::

        async with InferenceWebSocket(
            session=http_session,
            base_url="https://agent-gateway.livekit.cloud/v1",
            path="/stt?model=deepgram/nova-3",
            api_key=api_key,
            api_secret=api_secret,
            timeout=10.0,
        ) as iws:
            await iws.send(session_create_json)
            ...
    """

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        base_url: str,
        path: str,
        api_key: str,
        api_secret: str,
        timeout: float,
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._path = path
        self._api_key = api_key
        self._api_secret = api_secret
        self._timeout = timeout
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._closing = False

    async def __aenter__(self) -> InferenceWebSocket:
        base_url = self._base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)

        headers = {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._api_key, self._api_secret)}",
        }

        try:
            self._ws = await asyncio.wait_for(
                self._session.ws_connect(f"{base_url}{self._path}", headers=headers),
                self._timeout,
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                raise APIStatusError(
                    f"inference quota exceeded: {e.message}",
                    status_code=e.status,
                    retryable=False,
                ) from e
            raise create_api_error_from_http(e.message, status=e.status) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("inference websocket connection timed out") from e
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError(
                f"failed to connect to inference websocket at {self._path}"
            ) from e

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None

    @property
    def ws(self) -> aiohttp.ClientWebSocketResponse:
        assert self._ws is not None, "InferenceWebSocket not connected"
        return self._ws

    @property
    def closed(self) -> bool:
        return self._ws is None or self._ws.closed

    def mark_closing(self) -> None:
        """Signal that a graceful close has been initiated by the caller.

        After calling this, the recv iterators will return cleanly when the
        server closes the WebSocket instead of raising ``APIStatusError``.
        """
        self._closing = True

    @overload
    async def send(self, data: str) -> None: ...

    @overload
    async def send(self, data: bytes) -> None: ...

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, str):
            await self.ws.send_str(data)
        else:
            await self.ws.send_bytes(data)

    @overload
    def recv(self, payload_type: type[str]) -> AsyncIterator[str]: ...

    @overload
    def recv(self, payload_type: type[bytes]) -> AsyncIterator[bytes]: ...

    async def recv(self, payload_type: type[str] | type[bytes] = str) -> AsyncIterator[str | bytes]:
        """Yield payloads from the WebSocket.

        Args:
            payload_type: ``str`` for text frames, ``bytes`` for binary frames.

        Handles CLOSED/CLOSE/CLOSING detection: returns cleanly if
        ``mark_closing()`` was called or the session is closed,
        otherwise raises ``APIStatusError``.
        """
        expected_ws_type = _PAYLOAD_TO_WS_TYPE[payload_type]
        ws = self.ws
        while True:
            msg = await ws.receive()
            if msg.type in _WS_CLOSE_TYPES:
                if self._closing or self._session.closed:
                    return
                raise APIStatusError(
                    message="inference websocket connection closed unexpectedly",
                    status_code=ws.close_code or -1,
                    body=f"{msg.data=} {msg.extra=}",
                )

            if msg.type != expected_ws_type:
                logger.warning("unexpected inference websocket message type %s", msg.type)
                continue

            yield msg.data
