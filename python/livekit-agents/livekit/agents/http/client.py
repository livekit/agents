from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import enum
import functools
import json as jsonlib
import os
import ssl
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import IO, Any

import aiohttp
import aiohttp.payload
import certifi
from aiohttp.abc import AbstractStreamWriter
from pydantic import JsonValue

from .._bindings import (
    APIConnectionError,
    APITimeoutError,
    create_api_error_from_http,
    redacted_cause,
)
from .._generated.types import Timeout
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

# Same contract as the Rust transport: each APIConnectOptions limit applies to the wait it names,
# failures are framework errors (never aiohttp's), and the total limit and retries belong to the
# caller's retry owner.

__all__ = [
    "FormData",
    "HttpResponse",
    "HttpSession",
    "WebSocket",
    "WebSocketMessage",
    "WebSocketMessageType",
    "http_session",
    "open",
]


class HttpSession:
    """A connection pool for one provider or job. ``http_session()`` is the job's shared one.

    Redirects are never followed: a 3xx is returned, so credential headers stay with the host
    they were meant for.
    """

    def __init__(self, *, proxy: str | None = None) -> None:
        self._pool = _Pool(proxy)
        self._request_id_header: str | None = None

    def with_request_id(self, header: str) -> HttpSession:
        """This session, sharing its connections, whose status errors (a rejected handshake,
        ``raise_for_status()``) carry the response's ``header`` as ``APIStatusError.request_id``.
        """
        view = copy.copy(self)
        view._request_id_header = header
        return view

    @property
    def closed(self) -> bool:
        """Whether ``aclose()`` ran; the job's session closes when the job ends."""
        return self._pool.closed

    async def aclose(self) -> None:
        await self._pool.aclose()

    @contextlib.asynccontextmanager
    async def request(
        self,
        method: str,
        url: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        json: object = None,
        data: bytes | str | FormData | AsyncIterable[bytes] | None = None,
    ) -> AsyncIterator[HttpResponse]:
        """``async with session.request(...) as response:``, under ``conn_options``' limits."""
        async with _under_limits(conn_options, None, self._request_id_header) as deadline:
            body: aiohttp.payload.Payload | None = None
            if json is not None:
                body = aiohttp.payload.JsonPayload(json)
            elif data is not None:
                body = _payload(data._form() if isinstance(data, FormData) else data)
            # aiohttp treats 0 as no limit; the smallest positive float expires at the first wait
            read = conn_options.read_timeout
            timeout = aiohttp.ClientTimeout(
                sock_read=5e-324 if read == 0 else read, ceil_threshold=float("inf")
            )
            async with self._pool.client().request(
                method,
                url,
                headers=headers,
                params=params,
                data=None if body is None else _TimedBody(body, deadline),
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                yield HttpResponse(response, self._request_id_header)

    def get(self, url: str, **kwargs: Any) -> contextlib.AbstractAsyncContextManager[HttpResponse]:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> contextlib.AbstractAsyncContextManager[HttpResponse]:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> contextlib.AbstractAsyncContextManager[HttpResponse]:
        return self.request("PUT", url, **kwargs)

    def patch(
        self, url: str, **kwargs: Any
    ) -> contextlib.AbstractAsyncContextManager[HttpResponse]:
        return self.request("PATCH", url, **kwargs)

    def delete(
        self, url: str, **kwargs: Any
    ) -> contextlib.AbstractAsyncContextManager[HttpResponse]:
        return self.request("DELETE", url, **kwargs)

    async def ws_connect(
        self,
        url: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        heartbeat: float | None = None,
        protocols: tuple[str, ...] = (),
        max_msg_size: int = 4 * 1024 * 1024,
    ) -> WebSocket:
        """Open a websocket under the pool and connect limits, then the read limit for the
        handshake response. Frames sent on it run under the write limit; a received message
        larger than ``max_msg_size`` bytes fails the socket."""
        async with _under_limits(conn_options, "read", self._request_id_header):
            ws = await self._pool.client().ws_connect(
                url,
                headers=headers,
                params=params,
                heartbeat=heartbeat,
                protocols=protocols,
                max_msg_size=max_msg_size,
            )
        return WebSocket(ws, conn_options.write_timeout)


class _Pool:
    def __init__(self, proxy: str | None) -> None:
        self._proxy = proxy
        self._client: aiohttp.ClientSession | None = None
        self.closed = False

    def client(self) -> aiohttp.ClientSession:
        if self.closed:
            raise RuntimeError("the http session is closed")
        if self._client is None:
            self._client = aiohttp.ClientSession(
                proxy=self._proxy,
                connector=aiohttp.TCPConnector(
                    limit_per_host=50, keepalive_timeout=120, ssl=_ssl_context()
                ),
                trace_configs=[_CONNECTION_TRACE],
            )
        return self._client

    async def aclose(self) -> None:
        self.closed = True
        if self._client is not None:
            await self._client.close()
            self._client = None


class HttpResponse:
    """A response whose body is read inside the ``async with`` block that returned it."""

    def __init__(self, response: aiohttp.ClientResponse, request_id_header: str | None) -> None:
        self._response = response
        self._request_id_header = request_id_header

    @property
    def status(self) -> int:
        return self._response.status

    @property
    def ok(self) -> bool:
        """Whether the status is 2xx; a redirect is not, since it is never followed."""
        return 200 <= self._response.status < 300

    @property
    def reason(self) -> str | None:
        return self._response.reason

    @property
    def headers(self) -> Mapping[str, str]:
        return self._response.headers

    @property
    def content_type(self) -> str:
        return self._response.content_type

    async def read(self) -> bytes:
        return await self._response.read()

    async def text(self) -> str:
        return await self._response.text()

    async def json(self) -> JsonValue:
        """The body as JSON, whatever its content type."""
        value: JsonValue = jsonlib.loads(await self._response.read())
        return value

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        """The body as it arrives."""
        async for chunk in self._response.content.iter_any():
            yield chunk

    async def iter_lines(self) -> AsyncIterator[bytes]:
        """The body line by line as it arrives, without line endings, however long a line is."""
        pending = b""
        async for chunk in self.iter_bytes():
            *lines, pending = (pending + chunk).split(b"\n")
            for line in lines:
                yield line.removesuffix(b"\r")
        if pending:
            yield pending.removesuffix(b"\r")

    async def raise_for_status(self) -> None:
        """Raise ``APIStatusError`` with the status, body and request id unless the status is
        2xx."""
        if self.ok:
            return
        body = (await self.read()).decode("utf-8", errors="replace")
        raise create_api_error_from_http(
            self.reason or "",
            status=self.status,
            request_id=_request_id(self.headers, self._request_id_header),
            body=body,
        )


class FormData:
    """A multipart form body."""

    def __init__(self) -> None:
        self._fields: list[tuple[str, str | bytes | IO[bytes], str | None, str | None]] = []

    def add_field(
        self,
        name: str,
        value: str | bytes | IO[bytes],
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        self._fields.append((name, value, filename, content_type))

    def _form(self) -> aiohttp.FormData:
        form = aiohttp.FormData()
        for name, value, filename, content_type in self._fields:
            form.add_field(name, value, filename=filename, content_type=content_type)
        return form


class WebSocketMessageType(enum.Enum):
    TEXT = "text"
    BINARY = "binary"
    CLOSE = "close"


@dataclass(frozen=True)
class WebSocketMessage:
    type: WebSocketMessageType
    data: str | bytes
    """The text, the binary payload, or the close reason."""

    @property
    def text(self) -> str:
        """A text message's text, or a close message's reason."""
        if not isinstance(self.data, str):
            raise TypeError("a binary message has no text")
        return self.data

    @property
    def binary(self) -> bytes:
        """A binary message's payload."""
        if not isinstance(self.data, bytes):
            raise TypeError(f"a {self.type.value} message has no binary payload")
        return self.data

    def json(self) -> JsonValue:
        value: JsonValue = jsonlib.loads(self.data)
        return value


class WebSocket:
    """An open websocket. Sends run under the write limit; reads belong to the protocol."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, write_timeout: float | None) -> None:
        self._ws = ws
        self._write_timeout = write_timeout

    @property
    def headers(self) -> Mapping[str, str]:
        """The upgrade response's headers, such as the provider's request id."""
        return self._ws._response.headers

    @property
    def closed(self) -> bool:
        return self._ws.closed

    @property
    def close_code(self) -> int | None:
        return self._ws.close_code

    async def send_str(self, data: str) -> None:
        await self._send(self._ws.send_str(data))

    async def send_bytes(self, data: bytes) -> None:
        await self._send(self._ws.send_bytes(data))

    async def send_json(self, data: object) -> None:
        await self._send(self._ws.send_str(jsonlib.dumps(data)))

    async def receive(self, timeout: float | None = None) -> WebSocketMessage:
        """The next message; ``CLOSE`` once the socket closes. A transport error raises
        ``APIConnectionError``; ``timeout`` expiring raises ``APITimeoutError`` (0 expires at the
        first wait)."""
        try:
            # aiohttp treats 0 as no limit; the smallest positive float expires at the first wait
            msg = await self._ws.receive(timeout=5e-324 if timeout == 0 else timeout)
        except TimeoutError as e:
            raise APITimeoutError("read timed out", timeout="read") from e
        except (aiohttp.ClientError, OSError) as e:
            raise APIConnectionError("websocket receive failed") from redacted_cause(e)
        match msg.type:
            case aiohttp.WSMsgType.TEXT:
                return WebSocketMessage(WebSocketMessageType.TEXT, msg.data)
            case aiohttp.WSMsgType.BINARY:
                return WebSocketMessage(WebSocketMessageType.BINARY, msg.data)
            case aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("websocket failed") from redacted_cause(msg.data)
            case _:  # CLOSE, CLOSING, CLOSED
                return WebSocketMessage(WebSocketMessageType.CLOSE, msg.extra or "")

    async def __aiter__(self) -> AsyncIterator[WebSocketMessage]:
        while (msg := await self.receive()).type is not WebSocketMessageType.CLOSE:
            yield msg

    async def close(self, code: int = 1000, message: bytes = b"") -> None:
        await self._ws.close(code=code, message=message)

    async def _send(self, send: Awaitable[None]) -> None:
        try:
            async with asyncio.timeout(self._write_timeout):
                await send
        except TimeoutError as e:
            raise APITimeoutError("write timed out", timeout="write") from e
        except (aiohttp.ClientError, OSError) as e:
            raise APIConnectionError("websocket send failed") from redacted_cause(e)


# aiohttp only has a read limit (sock_read): pool and connect limits come from its trace events,
# the write limit from wrapping the body writer.
class _Deadline:
    # aiohttp writes the body in its own task, so a stalled write expires this shared timeout
    # and the request task fails, naming the limit that ran out.

    def __init__(
        self,
        options: APIConnectOptions,
        timeout: asyncio.Timeout,
        after_connect: Timeout | None,
    ) -> None:
        self._options = options
        self._timeout = timeout
        self.after_connect = after_connect
        self.limit: Timeout | None = None

    def set_limit(self, limit: Timeout | None) -> None:
        if self._timeout.expired():
            return  # keep the limit that fired; an expired asyncio.Timeout cannot be rescheduled
        self.limit = limit
        match limit:
            case "pool":
                seconds = self._options.pool_timeout
            case "connect":
                seconds = self._options.connect_timeout
            case "write":
                seconds = self._options.write_timeout
            case "read":
                seconds = self._options.read_timeout
            case _:
                seconds = None
        self._timeout.reschedule(
            None if seconds is None else asyncio.get_running_loop().time() + seconds
        )


_current = contextvars.ContextVar[_Deadline | None]("http_deadline", default=None)


@contextlib.asynccontextmanager
async def _under_limits(
    options: APIConnectOptions, after_connect: Timeout | None, request_id_header: str | None
) -> AsyncIterator[_Deadline]:
    try:
        async with asyncio.timeout(None) as timeout:
            deadline = _Deadline(options, timeout, after_connect)
            token = _current.set(deadline)
            try:
                yield deadline
            finally:
                _current.reset(token)
    except aiohttp.SocketTimeoutError as e:  # aiohttp's own read limit
        raise APITimeoutError("read timed out", timeout="read") from e
    except aiohttp.ClientResponseError as e:  # a rejected websocket handshake
        raise create_api_error_from_http(
            e.message, status=e.status, request_id=_request_id(e.headers, request_id_header)
        ) from redacted_cause(e)
    except TimeoutError as e:
        if not timeout.expired() or deadline.limit is None:
            raise
        raise APITimeoutError(f"{deadline.limit} timed out", timeout=deadline.limit) from e
    except (aiohttp.ClientError, OSError) as e:
        raise APIConnectionError("request failed") from redacted_cause(e)


def _request_id(headers: Mapping[str, str] | None, header: str | None) -> str | None:
    return headers.get(header) if headers is not None and header is not None else None


def _trace_hook(limit: Timeout | None, connected: bool = False) -> Callable[..., Awaitable[None]]:
    async def hook(*_: object) -> None:
        if (deadline := _current.get()) is not None:
            deadline.set_limit(deadline.after_connect if connected else limit)

    return hook


_CONNECTION_TRACE = aiohttp.TraceConfig()
_CONNECTION_TRACE.on_connection_queued_start.append(_trace_hook("pool"))
_CONNECTION_TRACE.on_connection_create_start.append(_trace_hook("connect"))  # DNS, TCP, TLS
for _connected in (
    _CONNECTION_TRACE.on_connection_queued_end,
    _CONNECTION_TRACE.on_connection_create_end,
    _CONNECTION_TRACE.on_connection_reuseconn,
):
    _connected.append(_trace_hook(None, connected=True))
_CONNECTION_TRACE.freeze()


def _payload(data: object) -> aiohttp.payload.Payload:
    """aiohttp's own body resolution, done here so the payload can be wrapped."""
    if isinstance(data, aiohttp.FormData):
        data = data()
    try:
        return aiohttp.payload.PAYLOAD_REGISTRY.get(data, disposition=None)
    except aiohttp.payload.LookupError:
        return aiohttp.FormData(data)()  # type: ignore[arg-type]


class _TimedBody(aiohttp.payload.Payload):
    def __init__(self, body: aiohttp.payload.Payload, deadline: _Deadline) -> None:
        super().__init__(body, headers=body.headers)
        self._size = body.size
        self._deadline = deadline

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        return self._value.decode(encoding, errors)  # type: ignore[no-any-return]

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value.write(_TimedBodyWriter(writer, self._deadline))

    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        await self._value.write_with_length(
            _TimedBodyWriter(writer, self._deadline), content_length
        )


class _TimedBodyWriter:
    def __init__(self, writer: AbstractStreamWriter, deadline: _Deadline) -> None:
        self._writer = writer
        self._deadline = deadline

    async def write(self, chunk: bytes | bytearray | memoryview) -> None:
        self._deadline.set_limit("write")
        try:
            await self._writer.write(chunk)
        finally:
            self._deadline.set_limit(None)

    def __getattr__(self, name: str) -> object:
        return getattr(self._writer, name)


def _export_ca_paths() -> None:
    # Called once at startup; job processes inherit SSL_CERT_FILE/SSL_CERT_DIR, which OpenSSL and
    # rustls-native-certs both read, so aiohttp, livekit-rtc and the native clients trust the
    # same roots (certifi when the container has no trust store).
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    paths = ssl.get_default_verify_paths()
    if paths.cafile and os.path.isfile(paths.cafile):
        os.environ["SSL_CERT_FILE"] = paths.cafile
    if paths.capath and os.path.isdir(paths.capath):
        os.environ["SSL_CERT_DIR"] = paths.capath
    if not (os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR")):
        os.environ["SSL_CERT_FILE"] = certifi.where()
        logger.debug("no system trust store found, using the certifi bundle")


# cached: parsing the CA bundle is slow
@functools.cache
def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


_ContextVar = contextvars.ContextVar[HttpSession | None]("agent_http_session", default=None)


def http_session() -> HttpSession:
    """The job's shared ``HttpSession``, bound to the job's event loop."""
    if (session := _ContextVar.get()) is None:
        raise RuntimeError(
            "no http session outside a job: wrap the code in "
            "`async with livekit.agents.http.open():`, "
            "or pass an HttpSession to the plugin"
        )
    return session


@contextlib.asynccontextmanager
async def open() -> AsyncIterator[HttpSession]:  # noqa: A001
    """Bind a session for ``http_session()`` outside a job, closed on exit.

    Inside a job or another ``open()``, yields the existing session and leaves it open.

    Example::

        async with http.open():
            async with VoiceSession() as session:
                await session.start(MyAgent())
    """
    if (existing := _ContextVar.get()) is not None:
        yield existing
        return
    _new_session_ctx()
    try:
        yield http_session()
    finally:
        await _close_http_ctx()


def _new_session_ctx() -> None:
    from ..job import get_job_context

    try:
        proxy = get_job_context().proc.http_proxy
    except RuntimeError:
        proxy = None
    _ContextVar.set(HttpSession(proxy=proxy))


async def _close_http_ctx() -> None:
    if (session := _ContextVar.get()) is not None:
        await session.aclose()
        _ContextVar.set(None)
