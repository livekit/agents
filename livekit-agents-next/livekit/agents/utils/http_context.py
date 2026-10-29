from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import os
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from typing import Any, Literal

import aiohttp
import aiohttp.payload
import certifi
from aiohttp._websocket.writer import WebSocketWriter
from aiohttp.abc import AbstractStreamWriter
from yarl import URL

from .._exceptions import APITimeoutError
from ..log import logger
from ..types import APIConnectOptions

_ClientFactory = Callable[[], aiohttp.ClientSession]
_ContextVar = contextvars.ContextVar[_ClientFactory | None]("agent_http_session")


_Phase = Literal["pool", "connect", "write", "read"]


class _Phases:
    """One request's deadline, armed with the limit of the phase it is waiting in."""

    def __init__(
        self, options: APIConnectOptions, deadline: asyncio.Timeout, after_connect: _Phase | None
    ) -> None:
        self._options = options
        self._deadline = deadline
        self.after_connect = after_connect
        self.armed: _Phase | None = None

    def arm(self, phase: _Phase | None) -> None:
        self.armed = phase
        match phase:
            case "pool":
                limit = self._options.pool_timeout
            case "connect":
                limit = self._options.connect_timeout
            case "write":
                limit = self._options.write_timeout
            case "read":
                limit = self._options.read_timeout
            case None:
                limit = None
        self._deadline.reschedule(
            None if limit is None else asyncio.get_running_loop().time() + limit
        )


_current_phases = contextvars.ContextVar[_Phases | None]("http_request_phases", default=None)


def _on(phase: _Phase | None | Literal["connected"]) -> Callable[..., Awaitable[None]]:
    async def arm(*_: object) -> None:
        if (phases := _current_phases.get()) is not None:
            phases.arm(phases.after_connect if phase == "connected" else phase)

    return arm


# aiohttp reports the pool wait and the connection setup (DNS, TCP, TLS) as separate events.
_PHASE_TRACE = aiohttp.TraceConfig()
_PHASE_TRACE.on_connection_queued_start.append(_on("pool"))
_PHASE_TRACE.on_connection_queued_end.append(_on("connected"))
_PHASE_TRACE.on_connection_create_start.append(_on("connect"))
_PHASE_TRACE.on_connection_create_end.append(_on("connected"))
_PHASE_TRACE.on_connection_reuseconn.append(_on("connected"))
_PHASE_TRACE.freeze()


class _WriteProgress:
    """A body writer whose every chunk waiting for the transport runs under the write limit."""

    def __init__(self, writer: AbstractStreamWriter, phases: _Phases) -> None:
        self._writer = writer
        self._phases = phases

    async def write(self, chunk: bytes | bytearray | memoryview) -> None:
        self._phases.arm("write")
        try:
            await self._writer.write(chunk)
        finally:
            self._phases.arm(None)

    def __getattr__(self, name: str) -> object:
        return getattr(self._writer, name)


class _TimedBody(aiohttp.payload.Payload):
    def __init__(self, body: aiohttp.payload.Payload, phases: _Phases) -> None:
        super().__init__(body, headers=body.headers)
        self._size = body.size
        self._phases = phases

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        return self._value.decode(encoding, errors)  # type: ignore[no-any-return]

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value.write(_WriteProgress(writer, self._phases))

    async def write_with_length(
        self, writer: AbstractStreamWriter, content_length: int | None
    ) -> None:
        await self._value.write_with_length(_WriteProgress(writer, self._phases), content_length)


@contextlib.asynccontextmanager
async def _phase_limits(
    session: aiohttp.ClientSession, options: APIConnectOptions, after_connect: _Phase | None
) -> AsyncIterator[_Phases]:
    if _PHASE_TRACE not in session.trace_configs:
        session.trace_configs.append(_PHASE_TRACE)
    try:
        async with asyncio.timeout(None) as deadline:
            phases = _Phases(options, deadline, after_connect)
            token = _current_phases.set(phases)
            try:
                yield phases
            finally:
                _current_phases.reset(token)
    except TimeoutError as e:
        if not deadline.expired() or phases.armed is None:
            raise
        raise APITimeoutError(f"{phases.armed} timed out", phase=phases.armed) from e


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
    """``session.request`` under the pool, connect, write and read limits of ``conn_options``.

    The total limit belongs to the operation's retry owner. A limit that expires raises
    ``APITimeoutError`` naming its phase; aiohttp's read limit raises its ``TimeoutError``.
    """
    async with _phase_limits(session, conn_options, after_connect=None) as phases:
        if json is not None:
            data = aiohttp.payload.JsonPayload(json, dumps=session.json_serialize)
        elif isinstance(data, aiohttp.FormData):
            data = data()
        if data is not None:
            try:
                body = aiohttp.payload.get_payload(data)
            except aiohttp.payload.LookupError:
                if not isinstance(data, Iterable):
                    raise TypeError(f"unsupported request body: {type(data).__name__}") from None
                body = aiohttp.FormData(data)()
            data = _TimedBody(body, phases)
        # aiohttp disables a zero limit; the smallest positive one expires at the first wait.
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
    """``session.ws_connect`` under the pool and connect limits, then the read limit while the
    handshake response is expected. Every frame sent on the open socket waits for the transport
    under the write limit; what is received is the caller's protocol."""
    async with _phase_limits(session, conn_options, after_connect="read"):
        ws: aiohttp.ClientWebSocketResponse = await session.ws_connect(url, **kwargs)
    if isinstance(ws, aiohttp.ClientWebSocketResponse):
        ws._writer = _TimedFrameWriter(ws._writer, conn_options.write_timeout)  # type: ignore[assignment]
    return ws


def websocket_headers(ws: aiohttp.ClientWebSocketResponse) -> Mapping[str, str]:
    """The HTTP upgrade response headers, including provider request identifiers."""
    return ws._response.headers


class _TimedFrameWriter:
    """A websocket's frame writer whose every frame waiting for the transport (a peer that stopped
    reading) runs under the write limit."""

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


def _has_system_trust_store() -> bool:
    """Whether OpenSSL's default verify paths resolve to something on disk.

    cert_store_stats() can't tell us this: a hashed capath dir loads lazily and
    reports 0 even when present.
    """
    paths = ssl.get_default_verify_paths()
    return bool(
        (paths.cafile and os.path.exists(paths.cafile))
        or (paths.capath and os.path.isdir(paths.capath))
    )


def _set_default_cert_env() -> None:
    """Point ``SSL_CERT_FILE`` at certifi when there's no system trust store.

    Unlike building an ``SSLContext``, the env var also reaches job subprocesses
    (which inherit it) and the Rust livekit-rtc SDK, since both OpenSSL and
    rustls-native-certs honor it. Otherwise room connections in minimal
    containers fail with "no native root CA certificates found".
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    if _has_system_trust_store():
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()
    logger.debug("no system trust store found, setting SSL_CERT_FILE to the certifi bundle")


def _create_ssl_context() -> ssl.SSLContext:
    """TLS context for the shared http session.

    Falls back to certifi when no system trust store is resolvable (e.g. minimal
    containers without ca-certificates); ``SSL_CERT_FILE`` / ``SSL_CERT_DIR``
    take precedence.

    Building a context parses the whole CA bundle, so the process builds one per
    trust configuration and every job's session shares it. An ``SSLContext`` is
    safe to share across threads and event loops. Like aiohttp's own default
    context, it is not reloaded: a CA bundle replaced at the same path takes
    effect when the process restarts.
    """
    return _ssl_context_for(os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR"))


@functools.cache
def _ssl_context_for(cafile: str | None, capath: str | None) -> ssl.SSLContext:
    if cafile or capath:
        return ssl.create_default_context(cafile=cafile, capath=capath)

    ctx = ssl.create_default_context()
    if not _has_system_trust_store():
        ctx.load_verify_locations(cafile=certifi.where())
    return ctx


def _new_session_ctx() -> _ClientFactory:
    g_session: aiohttp.ClientSession | None = None

    def _new_session() -> aiohttp.ClientSession:
        nonlocal g_session
        if g_session is None or g_session.closed:
            logger.debug("http_session(): creating a new httpclient ctx")

            from ..job import get_job_context

            try:
                http_proxy = get_job_context().proc.http_proxy
            except RuntimeError:
                http_proxy = None

            connector = aiohttp.TCPConnector(
                limit_per_host=50,
                keepalive_timeout=120,  # the default is only 15s
                ssl=_create_ssl_context(),
            )
            g_session = aiohttp.ClientSession(
                proxy=http_proxy, connector=connector, trace_configs=[_PHASE_TRACE]
            )
        return g_session

    _ContextVar.set(_new_session)
    return _new_session


def http_session() -> aiohttp.ClientSession:
    """Optional utility function to avoid having to manually manage an aiohttp.ClientSession lifetime.
    On job processes, this http session will be bound to the main event loop.
    """  # noqa: E501

    val = _ContextVar.get(None)
    if val is None:
        raise RuntimeError(
            "Attempted to use an http session outside of a job context. This is probably because you are trying to use a plugin without using the agent worker api. "  # noqa: E501
            "If you're running plugins outside the agent worker (e.g. tests or scripts), wrap your code with `async with livekit.agents.utils.http_context.open(): ...`. "  # noqa: E501
            "Alternatively, create your own aiohttp.ClientSession, pass it into the plugin constructor as a kwarg, and manage its lifecycle."  # noqa: E501
        )

    return val()


async def _close_http_ctx() -> None:
    val = _ContextVar.get(None)
    if val is not None:
        logger.debug("http_session(): closing the httpclient ctx")
        await val().close()
        _ContextVar.set(None)


@contextlib.asynccontextmanager
async def open() -> AsyncIterator[aiohttp.ClientSession]:  # noqa: A001
    """Bind a process-local aiohttp.ClientSession to the current asyncio context.

    Use this when running plugins outside a job worker (e.g. tests, scripts,
    notebooks) so that ``http_session()`` returns a usable session inside the
    ``async with`` block. The session is closed and the context is reset on exit.

    If an http session context is already bound (nested call, or already set up
    by the worker), this is a no-op pass-through — the existing session is
    yielded and left untouched on exit.

    Example::

        async with utils.http_context.open():
            async with VoiceSession() as session:
                await session.start(MyAgent())
    """
    if _ContextVar.get(None) is not None:
        yield _ContextVar.get()()  # type: ignore[misc]
        return

    factory = _new_session_ctx()
    try:
        yield factory()
    finally:
        await _close_http_ctx()
