"""HTTP and WebSocket for Python plugins, held to the same contract as the Rust transport.

Native providers use ``lk-agents-net``. Python plugins use aiohttp, and this module makes aiohttp
behave the same way:

- ``request()`` and ``ws_connect()`` enforce each ``APIConnectOptions`` limit (pool, connect,
  write, read) on the wait it names, and raise ``APITimeoutError`` naming the one that expired.
  The total limit and retries belong to the caller's retry owner, not to this module.
- ``http_session()`` is the job's shared session; ``open()`` provides one outside a job.
- Every client in the process trusts the same CA certificates (see ``_export_ca_paths``).

Limits and defaults are declared once in ABI; nothing here chooses a policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import os
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import aiohttp
import aiohttp.payload
import certifi
from aiohttp._websocket.writer import WebSocketWriter
from aiohttp.abc import AbstractStreamWriter
from yarl import URL

from .._bindings import APITimeoutError
from .._generated.types import TimeoutPhase
from ..log import logger
from ..types import APIConnectOptions

# --- public API -------------------------------------------------------------------------------


def http_session() -> aiohttp.ClientSession:
    """The job's shared ``aiohttp.ClientSession``, bound to the job's event loop."""
    if (lazy := _ContextVar.get()) is None:
        raise RuntimeError(
            "no http session outside a job: wrap the code in "
            "`async with livekit.agents.utils.http_context.open():`, "
            "or pass your own aiohttp.ClientSession to the plugin"
        )
    return lazy()


@contextlib.asynccontextmanager
async def open() -> AsyncIterator[aiohttp.ClientSession]:  # noqa: A001
    """Bind a session for ``http_session()`` outside a job, closed on exit.

    Inside a job or another ``open()``, yields the existing session and leaves it open.

    Example::

        async with utils.http_context.open():
            async with VoiceSession() as session:
                await session.start(MyAgent())
    """
    if _ContextVar.get() is not None:
        yield http_session()
        return
    _new_session_ctx()
    try:
        yield http_session()
    finally:
        await _close_http_ctx()


@contextlib.asynccontextmanager
async def request(
    session: aiohttp.ClientSession,
    method: str,
    url: str | URL,
    *,
    conn_options: APIConnectOptions,
    data: object = None,
    json: object = None,
    **kwargs: Any,
) -> AsyncIterator[aiohttp.ClientResponse]:
    """``session.request`` under the pool, connect, write and read limits of ``conn_options``."""
    async with _under_limits(session, conn_options, after_connect=None) as deadline:
        if json is not None:
            data = aiohttp.payload.JsonPayload(json, dumps=session.json_serialize)
        if data is not None:
            data = _TimedBody(_payload(data), deadline)
        # aiohttp treats 0 as no limit; the smallest positive float expires at the first wait
        read = conn_options.read_timeout
        timeout = aiohttp.ClientTimeout(
            sock_read=5e-324 if read == 0 else read, ceil_threshold=float("inf")
        )
        async with session.request(method, url, data=data, timeout=timeout, **kwargs) as response:
            yield response


async def ws_connect(
    session: aiohttp.ClientSession,
    url: str | URL,
    *,
    conn_options: APIConnectOptions,
    **kwargs: Any,
) -> aiohttp.ClientWebSocketResponse:
    """``session.ws_connect`` under the pool and connect limits, then the read limit for the
    handshake response. Frames sent on the open socket run under the write limit; reads belong
    to the caller's protocol."""
    async with _under_limits(session, conn_options, after_connect="read"):
        ws: aiohttp.ClientWebSocketResponse = await session.ws_connect(url, **kwargs)
    if isinstance(ws, aiohttp.ClientWebSocketResponse):  # not a test fake
        ws._writer = _TimedFrameWriter(ws._writer, conn_options.write_timeout)  # type: ignore[assignment]
    return ws


def websocket_headers(ws: aiohttp.ClientWebSocketResponse) -> Mapping[str, str]:
    """The HTTP upgrade response headers, including provider request identifiers."""
    return ws._response.headers


# --- limits -----------------------------------------------------------------------------------
#
# aiohttp only has a read limit (``sock_read``), so the others are added here:
#
# - a request runs under one asyncio timeout (``_Deadline``), moved to the limit of whatever
#   it is waiting on;
# - aiohttp's trace events say when it waits for the pool or opens a connection;
# - the request body's writer is wrapped, so each chunk waits under the write limit;
# - an open websocket's frame writer is wrapped the same way.


class _Deadline:
    """One request's timeout, set to the limit of the wait in progress.

    aiohttp writes the body in its own task, so a stalled body write expires this shared
    timeout: the request task is the one that fails, naming the limit that ran out.
    """

    def __init__(
        self,
        options: APIConnectOptions,
        timeout: asyncio.Timeout,
        after_connect: TimeoutPhase | None,
    ) -> None:
        self._options = options
        self._timeout = timeout
        self.after_connect = after_connect
        self.limit: TimeoutPhase | None = None

    def set_limit(self, limit: TimeoutPhase | None) -> None:
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
    session: aiohttp.ClientSession, options: APIConnectOptions, after_connect: TimeoutPhase | None
) -> AsyncIterator[_Deadline]:
    if _CONNECTION_TRACE not in session.trace_configs:
        session.trace_configs.append(_CONNECTION_TRACE)
    try:
        async with asyncio.timeout(None) as timeout:
            deadline = _Deadline(options, timeout, after_connect)
            token = _current.set(deadline)
            try:
                yield deadline
            finally:
                _current.reset(token)
    except aiohttp.SocketTimeoutError as e:  # aiohttp's own read limit
        raise APITimeoutError("read timed out", phase="read") from e
    except TimeoutError as e:
        if not timeout.expired() or deadline.limit is None:
            raise
        raise APITimeoutError(f"{deadline.limit} timed out", phase=deadline.limit) from e


def _trace_hook(
    limit: TimeoutPhase | None, connected: bool = False
) -> Callable[..., Awaitable[None]]:
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
    """A request body written through ``_TimedBodyWriter``."""

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
    """Each body chunk waits for the transport under the write limit."""

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


class _TimedFrameWriter:
    """Each websocket frame waits for the transport under the write limit."""

    def __init__(self, writer: WebSocketWriter, limit: float | None) -> None:
        self._writer = writer
        self._limit = limit

    async def send_frame(self, message: bytes, opcode: int, compress: int | None = None) -> None:
        try:
            async with asyncio.timeout(self._limit):
                await self._writer.send_frame(message, opcode, compress)
        except TimeoutError as e:
            raise APITimeoutError("write timed out", phase="write") from e

    def __getattr__(self, name: str) -> object:
        return getattr(self._writer, name)


# --- trusted CA certificates ------------------------------------------------------------------


def _export_ca_paths() -> None:
    """Export the CA paths Python resolved, or certifi when there are none, to
    ``SSL_CERT_FILE``/``SSL_CERT_DIR``. Explicit values are kept.

    ``cli.run_app`` calls it once at startup. Job processes inherit the variables, and OpenSSL
    and rustls-native-certs both read them, so aiohttp, livekit-rtc and the native clients trust
    the same roots, including in minimal containers without ca-certificates.
    """
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


@functools.cache
def _ssl_context() -> ssl.SSLContext:
    """Built once (parsing the bundle is slow) and shared by every job's session. Like the
    native roots, it is not reloaded."""
    return ssl.create_default_context()


# --- the job's session ------------------------------------------------------------------------


class _LazySession:
    """The job's session, created on first use (and again if closed)."""

    def __init__(self) -> None:
        self.session: aiohttp.ClientSession | None = None

    def __call__(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            from ..job import get_job_context

            try:
                proxy = get_job_context().proc.http_proxy
            except RuntimeError:
                proxy = None
            connector = aiohttp.TCPConnector(
                limit_per_host=50, keepalive_timeout=120, ssl=_ssl_context()
            )
            self.session = aiohttp.ClientSession(
                proxy=proxy, connector=connector, trace_configs=[_CONNECTION_TRACE]
            )
        return self.session


_ContextVar = contextvars.ContextVar[_LazySession | None]("agent_http_session", default=None)


def _new_session_ctx() -> None:
    _ContextVar.set(_LazySession())


async def _close_http_ctx() -> None:
    if (lazy := _ContextVar.get()) is not None:
        if lazy.session is not None:
            await lazy.session.close()
        _ContextVar.set(None)
