"""The Python HTTP client: its limits, errors and websocket messages, the job session, and the
connection pool plugins keep their provider sockets in, against loopback peers."""

from __future__ import annotations

import asyncio
import contextlib
import pickle
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import aiohttp
import pytest
from aiohttp import RequestInfo, WSServerHandshakeError, web
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.http import HttpSession, WebSocketMessageType, client as http
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils import ConnectionPool
from tests.fakes import until

FAST = 0.2
SLOW = 5.0
SECRET_API_KEY = "secret-api-key-do-not-log"
NO_POOL_WAIT = APIConnectOptions(pool_timeout=0)


def _options(
    *,
    read_timeout: float = SLOW,
    write_timeout: float = SLOW,
    pool_timeout: float = SLOW,
    connect_timeout: float = SLOW,
) -> APIConnectOptions:
    return APIConnectOptions(
        max_retry=0,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        pool_timeout=pool_timeout,
        connect_timeout=connect_timeout,
    )


@pytest.fixture
async def server() -> AsyncIterator[str]:
    """``/slow?delay=`` answers late, ``/hang`` never answers, ``/endless`` never finishes its
    body, ``/sink`` never reads the request body, ``/fail`` answers 500, ``/reject`` 401, ``/ws`` never reads
    frames and ``/echo`` echoes until told to close."""
    release = asyncio.Event()

    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(float(request.query["delay"]))
        return web.Response(text="ok")

    async def hang(request: web.Request) -> web.Response:
        await release.wait()
        return web.Response()

    async def endless(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await release.wait()
        return response

    async def sink(request: web.Request) -> web.Response:
        while request.transport is not None and not request.transport.is_closing():
            await asyncio.sleep(0.01)  # until the client gives up
        return web.Response()

    async def fail(request: web.Request) -> web.Response:
        return web.Response(status=500, text="quota exceeded", headers={"x-trace-id": "tr-1"})

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401, text="Unauthorized")

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/slow?delay=0")

    async def lines(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        for chunk in (b"data: a\r\nda", b"ta: " + b"b" * 100_000 + b"\n", b"tail"):
            await response.write(chunk)
        return response

    async def ws_idle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.transport.pause_reading()  # type: ignore[union-attr]
        await release.wait()
        return ws

    async def echo(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        ws.headers["x-request-id"] = "req-1"
        await ws.prepare(request)
        async for msg in ws:
            if isinstance(msg.data, str) and msg.data.startswith("close"):
                code = int(msg.data.partition(":")[2] or 4000)
                await ws.close(code=code, message=b"bye")
            elif msg.type is aiohttp.WSMsgType.TEXT:
                await ws.send_str(msg.data)
            else:
                await ws.send_bytes(msg.data)
        return ws

    app = web.Application(client_max_size=1 << 30)
    app.router.add_route("*", "/slow", slow)
    app.router.add_get("/hang", hang)
    app.router.add_get("/endless", endless)
    app.router.add_post("/sink", sink)
    app.router.add_get("/fail", fail)
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/lines", lines)
    app.router.add_get("/reject", reject)
    app.router.add_get("/ws", ws_idle)
    app.router.add_get("/echo", echo)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        release.set()
        await runner.cleanup()


@pytest.fixture
async def silent_peer() -> AsyncIterator[str]:
    """Accepts connections and never answers."""
    held: list[asyncio.StreamWriter] = []

    async def accept(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    srv = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        for w in held:
            w.close()
        srv.close()


@pytest.fixture
async def session() -> AsyncIterator[HttpSession]:
    session = HttpSession()
    yield session
    await session.aclose()


async def _timeout_of(call: Callable[[], Awaitable[object]]) -> str | None:
    """The limit ``call`` fails with; a limit that never fires fails the test after 3s."""
    with pytest.raises(APITimeoutError) as raised:
        async with asyncio.timeout(3):
            await call()
    return raised.value.timeout


async def _get(session: HttpSession, url: str, options: APIConnectOptions) -> None:
    async with session.get(url, conn_options=options) as response:
        await response.read()


# --- limits -----------------------------------------------------------------------------------


async def test_a_slow_response_fires_the_read_limit(server: str, session: HttpSession) -> None:
    """A response that never comes fails with the read limit."""
    phase = await _timeout_of(lambda: _get(session, f"{server}/hang", _options(read_timeout=FAST)))
    assert phase == "read"


async def test_a_response_within_the_read_limit_ignores_the_other_limits(
    server: str, session: HttpSession
) -> None:
    """Waiting for the response is not a write, pool or connect wait."""
    options = _options(read_timeout=SLOW, write_timeout=0.05, pool_timeout=0.05, connect_timeout=1)
    # the answer comes after the write and pool limits would have fired
    async with session.post(f"{server}/slow?delay=0.1", conn_options=options, data=b"x") as resp:
        assert await resp.text() == "ok"


async def test_a_body_the_peer_does_not_read_fires_the_write_limit(
    server: str, session: HttpSession
) -> None:
    """A request body the peer never reads fails with the write limit."""

    async def post() -> None:
        options = _options(write_timeout=FAST)
        async with session.post(f"{server}/sink", conn_options=options, data=b"x" * (256 << 20)):
            pass

    assert await _timeout_of(post) == "write"


async def test_a_silent_websocket_handshake_fires_the_read_limit(
    silent_peer: str, session: HttpSession
) -> None:
    """A websocket handshake left unanswered fails with the read limit."""
    phase = await _timeout_of(
        lambda: session.ws_connect(f"{silent_peer}/ws", conn_options=_options(read_timeout=FAST))
    )
    assert phase == "read"


async def test_frames_the_peer_does_not_read_fire_the_write_limit(
    session: HttpSession, server: str
) -> None:
    """Websocket frames the peer never reads fail with the write limit."""
    # `server` is torn down first: its close ends the stalled socket the session holds
    ws = await session.ws_connect(f"{server}/ws", conn_options=_options(write_timeout=FAST))

    async def flood() -> None:
        while True:
            await ws.send_bytes(b"x" * (1 << 20))

    assert await _timeout_of(flood) == "write"


async def test_receive_timeout_fires_the_read_limit(server: str, session: HttpSession) -> None:
    """A websocket `receive(timeout=)` that expires is a read timeout."""
    ws = await session.ws_connect(f"{server}/echo")
    assert await _timeout_of(lambda: ws.receive(timeout=FAST)) == "read"
    await ws.close()


async def test_a_slow_dns_lookup_fires_the_connect_limit(
    server: str, session: HttpSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving the host name counts against the connect limit."""
    loop = asyncio.get_running_loop()
    resolve = loop.getaddrinfo

    # Any: loop.getaddrinfo's arguments and result, forwarded untouched
    async def stalled_lookup(*args: Any, **kwargs: Any) -> Any:
        await asyncio.Event().wait()  # never answers: only the connect limit ends it
        return await resolve(*args, **kwargs)

    monkeypatch.setattr(loop, "getaddrinfo", stalled_lookup)
    url = server.replace("127.0.0.1", "localhost") + "/hang"

    phase = await _timeout_of(lambda: _get(session, url, _options(connect_timeout=FAST)))

    assert phase == "connect"


async def test_a_session_with_every_connection_busy_fires_the_pool_limit(
    server: str, session: HttpSession
) -> None:
    """Once every pooled connection is held by an unfinished response, the next request fails
    with the pool limit."""
    options = _options(pool_timeout=FAST)

    async with contextlib.AsyncExitStack() as busy:

        async def occupy_another_connection() -> None:
            while True:
                await busy.enter_async_context(
                    session.get(f"{server}/endless", conn_options=options)
                )

        phase = await _timeout_of(occupy_another_connection)

    assert phase == "pool"


# --- errors and messages ----------------------------------------------------------------------


async def test_an_unreachable_host_is_a_connection_error(session: HttpSession) -> None:
    """A refused connection is an `APIConnectionError`, not a timeout."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens once closed
    with pytest.raises(APIConnectionError) as raised:
        await _get(session, f"http://127.0.0.1:{port}/", _options())
    assert not isinstance(raised.value, APITimeoutError)


async def test_an_error_status_is_returned_and_raise_for_status_keeps_the_body(
    server: str, session: HttpSession
) -> None:
    """An error status is returned; `raise_for_status()` raises it with the body and the request id, and the error pickles whole."""
    traced = session.with_request_id("x-trace-id")
    async with traced.get(f"{server}/fail") as response:
        assert (response.status, response.ok) == (500, False)
        with pytest.raises(APIStatusError) as raised:
            await response.raise_for_status()
    restored = pickle.loads(pickle.dumps(raised.value))

    assert (restored.status_code, restored.body, restored.request_id) == (
        500,
        "quota exceeded",
        "tr-1",
    )


async def test_a_rejected_handshake_is_a_status_error(server: str, session: HttpSession) -> None:
    """A websocket handshake answered with an error status is an `APIStatusError`."""
    with pytest.raises(APIStatusError) as raised:
        await session.ws_connect(f"{server}/fail")
    assert raised.value.status_code == 500


async def test_redirects_are_returned_not_followed(server: str, session: HttpSession) -> None:
    """A redirect is returned, not followed, so credentials stay with their host."""
    async with session.get(f"{server}/redirect") as response:
        assert (response.status, response.headers["Location"]) == (302, "/slow?delay=0")
        assert not response.ok
        with pytest.raises(APIStatusError) as raised:
            await response.raise_for_status()
    assert raised.value.status_code == 302


async def test_iter_lines_splits_across_chunks_without_a_line_limit(
    server: str, session: HttpSession
) -> None:
    """`iter_lines()` reassembles lines split across chunks, however long."""
    async with session.get(f"{server}/lines") as response:
        lines = [line async for line in response.iter_lines()]
    assert lines == [b"data: a", b"data: " + b"b" * 100_000, b"tail"]


async def test_a_closed_session_refuses_requests(server: str) -> None:
    """A closed session, and every view of it, refuses new requests."""
    session = HttpSession()
    traced = session.with_request_id("x-trace-id")
    await session.aclose()
    assert session.closed and traced.closed
    with pytest.raises(RuntimeError):
        await _get(session, f"{server}/slow?delay=0", _options())


async def test_websocket_messages_and_close(server: str, session: HttpSession) -> None:
    """Websocket text, binary and close messages arrive typed, with the close code."""
    ws = await session.ws_connect(f"{server}/echo")
    assert ws.headers["x-request-id"] == "req-1"
    await ws.send_json({"a": 1})
    await ws.send_bytes(b"\x01")
    await ws.send_str("close")
    messages = [msg async for msg in ws]
    assert [(m.type, m.data) for m in messages] == [
        (WebSocketMessageType.TEXT, '{"a": 1}'),
        (WebSocketMessageType.BINARY, b"\x01"),
    ]
    assert messages[0].json() == {"a": 1}
    assert (messages[0].text, messages[1].binary) == ('{"a": 1}', b"\x01")
    with pytest.raises(TypeError):
        _ = messages[0].binary
    closing = await ws.receive()
    assert closing.type is WebSocketMessageType.CLOSE
    assert ws.close_code == 4000


@pytest.mark.parametrize(
    ("code", "kind"), [(1008, "auth"), (4401, "auth"), (4402, "quota"), (1011, "server")]
)
async def test_a_provider_close_names_the_kind_its_code_means(
    server: str, session: HttpSession, code: int, kind: str
) -> None:
    """An auth or policy close is final (it used to be retried like any close), a server error
    is retried; the provider's close reason is the error's detail, never its text."""
    ws = await session.ws_connect(f"{server}/echo")
    await ws.send_str(f"close:{code}")
    assert (await ws.receive()).type is WebSocketMessageType.CLOSE

    error = ws.close_error("provider closed")

    assert (str(error), error.kind, error.retryable, error.detail) == (
        f"provider closed (close code {code})",
        kind,
        kind == "server",
        "bye",
    )


async def test_a_provider_close_code_of_its_own_is_read_from_its_table(
    server: str, session: HttpSession
) -> None:
    """A provider's private close code (Vakyam's 4001 for a rejected key) is final once its
    table names it, where the standard rule reads any 4000 code as a dropped connection."""
    ws = await session.ws_connect(f"{server}/echo")
    await ws.send_str("close:4001")
    assert (await ws.receive()).type is WebSocketMessageType.CLOSE

    assert ws.close_error("closed").kind == "connection"
    assert ws.close_error("closed", {4001: "auth"}).kind == "auth"


def _assert_redacted(err: BaseException) -> None:
    assert isinstance(err, APIStatusError) and err.status_code == 401
    chain: BaseException | None = err
    while chain is not None:
        assert SECRET_API_KEY not in repr(chain)
        assert SECRET_API_KEY not in str(chain)
        chain = chain.__cause__


async def test_the_client_redacts_credentials_from_a_rejected_handshake(
    server: str,
) -> None:
    """A rejected handshake's error, whose request headers hold the credential, never shows it.
    https://github.com/livekit/agents/issues/7031"""
    async with http.open() as session:
        with pytest.raises(APIStatusError) as exc_info:
            await session.ws_connect(
                server.replace("http", "ws", 1) + "/reject",
                headers={"Authorization": f"Token {SECRET_API_KEY}"},
            )
    _assert_redacted(exc_info.value)


# --- the job session --------------------------------------------------------------------------


async def test_open_binds_a_session_and_closes_it_on_exit(server: str) -> None:
    """`http.open()` binds `http_session()` for its block and closes it on exit."""
    with pytest.raises(RuntimeError):
        http.http_session()

    async with http.open() as session:
        assert http.http_session() is session
        async with session.get(f"{server}/slow?delay=0") as response:
            assert await response.text() == "ok"

    assert session.closed
    with pytest.raises(RuntimeError):
        http.http_session()


async def test_open_is_reentrant_inner_does_not_close_outer() -> None:
    """A nested `http.open()` yields the outer session and leaves it open.
    https://github.com/livekit/agents/pull/5709"""
    async with http.open() as outer:
        async with http.open() as inner:
            assert inner is outer
        assert http.http_session() is outer


async def test_open_isolated_per_task() -> None:
    """Concurrent tasks each bind their own session. https://github.com/livekit/agents/pull/5709"""
    both = asyncio.Barrier(2)

    async def task() -> tuple[HttpSession, HttpSession]:
        async with http.open() as session:
            await both.wait()
            return session, http.http_session()

    (a, a_bound), (b, b_bound) = await asyncio.gather(task(), task())

    assert a is a_bound and b is b_bound
    assert a is not b


# --- the connection pool ----------------------------------------------------------------------


@dataclass(frozen=True)
class Connection:
    """A pooled provider socket, numbered in the order it was opened."""

    number: int


class Pool:
    """A `ConnectionPool` of `Connection`s, keeping the ones it closed in ``closed``."""

    def __init__(self) -> None:
        self.opened = 0
        self.closed: list[Connection] = []
        self.pool = ConnectionPool[Connection](
            max_session_duration=60, connect_cb=self.connect, close_cb=self.close
        )

    async def connect(self, conn_options: APIConnectOptions) -> Connection:
        self.opened += 1
        return Connection(self.opened)

    async def close(self, connection: Connection) -> None:
        self.closed.append(connection)

    async def get(self) -> Connection:
        return await self.pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)


async def test_a_pool_wait_expires_without_interrupting_the_connecting_caller() -> None:
    """A caller that cannot wait for the pool fails with the pool limit while the caller already
    connecting gets its connection, reusable afterwards."""
    connecting, release = asyncio.Event(), asyncio.Event()

    async def connect(conn_options: APIConnectOptions) -> Connection:
        connecting.set()
        await release.wait()
        return Connection(1)

    pool = ConnectionPool[Connection](connect_cb=connect)
    first = asyncio.create_task(pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS))
    await connecting.wait()

    with pytest.raises(APITimeoutError) as raised:
        await pool.get(conn_options=NO_POOL_WAIT)
    release.set()
    connection = await first
    pool.put(connection)

    assert raised.value.timeout == "pool"
    assert await pool.get(conn_options=NO_POOL_WAIT) is connection
    await pool.aclose()


@pytest.mark.parametrize(
    "failure",
    [
        WSServerHandshakeError(
            RequestInfo(
                url=URL("wss://api.example.com/tts"),
                method="GET",
                headers=CIMultiDictProxy(CIMultiDict({"X-API-Key": SECRET_API_KEY})),
                real_url=URL("wss://api.example.com/tts"),
            ),
            (),
            status=401,
            message="Unauthorized",
        ),
        ConnectionError(f"wss://example.com/ws?api_key={SECRET_API_KEY}"),
    ],
    ids=["handshake-headers", "url-query"],
)
async def test_a_failed_prewarm_logs_no_credentials(
    failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """A prewarm failure is logged without the request headers or URL that carry the API key.
    https://github.com/livekit/agents/issues/6739"""

    async def connect(conn_options: APIConnectOptions) -> Connection:
        raise failure

    pool = ConnectionPool[Connection](connect_cb=connect)

    pool.prewarm()
    await until(lambda: "failed to prewarm" in caplog.text, "the prewarm failure was not logged")

    assert SECRET_API_KEY not in caplog.text


async def test_a_prewarm_after_a_failed_one_connects_and_is_reused() -> None:
    """A failed prewarm does not stop the next one, whose connection the next `get` reuses.
    https://github.com/livekit/agents/pull/6969 https://github.com/livekit/agents/pull/2545"""
    attempts: list[int] = []

    async def connect(conn_options: APIConnectOptions) -> Connection:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise ConnectionError("temporary prewarm failure")
        return Connection(len(attempts))

    pool = ConnectionPool[Connection](connect_cb=connect)
    pool.prewarm()
    await until(lambda: attempts == [1], "the first prewarm never ran")
    await asyncio.sleep(0)  # its failure is handled

    pool.prewarm()
    await until(lambda: attempts == [1, 2], "the second prewarm never ran")

    assert await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS) == Connection(2)
    assert attempts == [1, 2]


async def test_a_connection_removed_after_invalidate_is_closed() -> None:
    """A checked-out connection removed after `invalidate()` is closed.
    https://github.com/livekit/agents/pull/7139"""
    pool = Pool()
    in_use = await pool.get()

    pool.pool.invalidate()
    pool.pool.remove(in_use)

    await until(lambda: pool.closed == [in_use], "the removed connection was never closed")


async def test_invalidate_lets_a_stream_in_flight_finish_on_its_connection() -> None:
    """After `invalidate()` (options changed), a new stream gets a new connection while the one
    in flight keeps its own until returned, then closed.
    https://github.com/livekit/agents/pull/7139"""
    pool = Pool()
    speaking = await pool.get()

    pool.pool.invalidate()
    fresh = await pool.get()
    assert pool.closed == []
    pool.pool.put(speaking)
    pool.pool.put(fresh)

    assert await pool.get() is fresh
    await until(lambda: pool.closed == [speaking], "the retired connection was never closed")


async def test_a_handshake_in_flight_during_invalidate_is_discarded() -> None:
    """A socket negotiated with the old options is closed and never handed to the caller.
    https://github.com/livekit/agents/pull/7139"""
    started, release = asyncio.Event(), asyncio.Event()
    opened: list[Connection] = []
    closed: list[Connection] = []

    async def connect(conn_options: APIConnectOptions) -> Connection:
        opened.append(Connection(len(opened) + 1))
        started.set()
        await release.wait()
        return opened[-1]

    async def close(connection: Connection) -> None:
        closed.append(connection)

    pool = ConnectionPool[Connection](connect_cb=connect, close_cb=close)
    acquiring = asyncio.create_task(pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS))
    await started.wait()

    pool.invalidate()
    release.set()

    assert await acquiring == Connection(2)
    await until(lambda: closed == [Connection(1)], "the stale handshake was never closed")


async def test_aclose_waits_for_a_close_in_flight() -> None:
    """A close runs on the pool's own task from the moment the pool lets the connection go, so
    no caller can abandon it, and `aclose()` waits for it. https://github.com/livekit/agents/pull/7139"""
    closing, finish = asyncio.Event(), asyncio.Event()
    closed: list[Connection] = []

    async def close(connection: Connection) -> None:
        closing.set()
        await finish.wait()
        closed.append(connection)

    pool = ConnectionPool[Connection](connect_cb=Pool().connect, close_cb=close)
    retired = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.put(retired)
    pool.invalidate()
    await closing.wait()

    closing_pool = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)
    assert not closing_pool.done()
    finish.set()
    await closing_pool

    assert closed == [retired]


# --- retry_request ----------------------------------------------------------------------------


@pytest.fixture
async def start_peer() -> AsyncIterator[tuple[str, list[str]]]:
    """``POST /start`` answers 503 with ``Retry-After: 0`` once, then 200; ``POST /stall`` never
    answers. Each request's path is recorded."""
    seen: list[str] = []
    release = asyncio.Event()

    async def start(request: web.Request) -> web.Response:
        seen.append(request.path)
        if seen.count("/start") == 1:
            return web.Response(status=503, headers={"Retry-After": "0"})
        return web.json_response({"session": "s-1"})

    async def stall(request: web.Request) -> web.Response:
        seen.append(request.path)
        await release.wait()
        return web.Response()

    app = web.Application()
    app.router.add_post("/start", start)
    app.router.add_post("/stall", stall)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", seen
    finally:
        release.set()
        await runner.cleanup()


async def _post_json(session: HttpSession, url: str, options: APIConnectOptions) -> Any:
    async with session.post(url, json={}, conn_options=options) as response:
        await response.raise_for_status()
        return await response.json()


async def test_a_session_start_is_retried_after_a_server_error(
    start_peer: tuple[str, list[str]], session: HttpSession
) -> None:
    """A 503 with Retry-After is retried and the second answer is returned."""
    url, seen = start_peer
    options = APIConnectOptions(max_retry=2, retry_interval=0)

    result = await http.retry_request(
        lambda: _post_json(session, f"{url}/start", options), conn_options=options
    )

    assert (result, seen) == ({"session": "s-1"}, ["/start", "/start"])


async def test_a_session_start_whose_answer_timed_out_is_never_sent_again(
    start_peer: tuple[str, list[str]], session: HttpSession
) -> None:
    """A POST the provider may have accepted is not repeated (a second avatar session)."""
    url, seen = start_peer
    options = APIConnectOptions(max_retry=3, retry_interval=0, read_timeout=FAST)

    with pytest.raises(APITimeoutError) as raised:
        await http.retry_request(
            lambda: _post_json(session, f"{url}/stall", options), conn_options=options
        )

    assert (raised.value.timeout, seen) == ("read", ["/stall"])
