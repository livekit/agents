"""Sessions in agent-db: its data plane as an ``Executor``, and the store over it.

The server severs sockets on drain and eviction and marks those failures retryable, so
reconnecting is the normal path here, not an error case.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import random
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, TypeVar

import aiohttp
import jwt

from livekit.protocol import agentdb as pb

from ..log import logger
from ..utils import aio
from .executor import ExecResult, Executor, Row, Statement, StoreError, Value
from .session import LEASE_TTL, _Store

if TYPE_CHECKING:
    from google.protobuf.message import Message

    from livekit.api.twirp_client import TwirpClient

_M = TypeVar("_M", bound="Message")

Wire = pb.AgentDB.Wire

TOKEN_TTL = 6 * 3600.0
MAX_FRAME_BYTES = 16 << 20
"""The server's cap on one inbound frame; results come back under the same bound."""

RETRYABLE_CODES = frozenset({"unavailable"})
"""Codes the server uses for a database that is moving, where the request is sent again."""
RETRY_DELAY = 0.1
RECONNECT_BUDGET = 60.0
"""How long a request waits across reconnects, and the supervisor redials, before failing."""


class _Disconnected(Exception):
    """The socket went away with the request in flight."""


def _statement(sql: str, params: tuple[Value, ...] | list[Value]) -> pb.AgentDB.Wire.Statement:
    values: list[pb.AgentDB.Wire.Value] = []
    for value in params:
        if value is None:
            values.append(Wire.Value(null_value=True))
        elif isinstance(value, bool):
            values.append(Wire.Value(int_value=int(value)))
        elif isinstance(value, int):
            values.append(Wire.Value(int_value=value))
        elif isinstance(value, float):
            values.append(Wire.Value(double_value=value))
        elif isinstance(value, str):
            values.append(Wire.Value(text_value=value))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            values.append(Wire.Value(blob_value=bytes(value)))
        else:
            raise TypeError(f"cannot bind a {type(value).__name__} as a SQL parameter")
    return Wire.Statement(sql=sql, params=values)


class AgentDBExecutor:
    """An ``Executor`` on one agent-db database, over one WebSocket.

    A request cut off by a reconnect is sent again, so every statement run through it must be
    idempotent; a query that already yielded rows fails instead.
    """

    def __init__(
        self,
        *,
        ws_url: str,
        database_id: str,
        token: Callable[[], str],
    ) -> None:
        self._ws_url = ws_url
        self._database_id = database_id
        self._token = token
        self._http_session: aiohttp.ClientSession | None = None

        self._ids = itertools.count(1)
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[int, asyncio.Queue[pb.AgentDB.Wire.ServerMessage | None]] = {}
        # cleared across a reconnect, so a request made then waits instead of failing
        self._ready = asyncio.Event()
        self._closed = False
        self._close_reason: Exception | None = None
        self._ping_interval = 0.0
        self._supervisor: asyncio.Task[None] | None = None
        self._pinger: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Dial and say Hello unless connected; the first connect raises a bad URL or token."""
        if self._closed:
            raise StoreError("closed", "the agent-db connection is closed")
        if self._ready.is_set():
            return
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        token = self._token()
        ws = await self._http_session.ws_connect(
            self._ws_url,
            headers={"Authorization": f"Bearer {token}"},
            max_msg_size=MAX_FRAME_BYTES,
            autoping=True,
        )
        # the handshake is strict request/response, so it happens before any demux loop
        hello = Wire.ClientMessage(
            request_id=next(self._ids),
            hello=Wire.Hello(token=token, database_id=self._database_id),
        )
        try:
            await ws.send_bytes(hello.SerializeToString())
            frame = await ws.receive()
            if frame.type != aiohttp.WSMsgType.BINARY:
                raise StoreError("unavailable", f"socket closed during hello ({frame.type.name})")
            reply = Wire.ServerMessage.FromString(frame.data)
            if reply.WhichOneof("message") == "error":
                raise StoreError(reply.error.code, reply.error.message)
            if reply.WhichOneof("message") != "hello_ok":
                raise StoreError(
                    "protocol", f"unexpected hello reply {reply.WhichOneof('message')}"
                )
        except BaseException:
            await ws.close()
            raise
        self._ws = ws
        self._ping_interval = reply.hello_ok.ping_interval_ms / 1000
        self._ready.set()
        # the supervisor redials through connect() too, so the tasks are made on the first only
        if self._supervisor is None:
            self._supervisor = asyncio.create_task(self._supervise(), name="agentdb_supervise")
            self._pinger = asyncio.create_task(self._ping(), name="agentdb_ping")

    async def _supervise(self) -> None:
        """Own the socket's lifetime: read it, and redial when it dies."""
        while not self._closed:
            assert self._ws is not None
            ws = self._ws
            async for frame in ws:
                if frame.type != aiohttp.WSMsgType.BINARY:
                    continue
                message = Wire.ServerMessage.FromString(frame.data)
                if message.WhichOneof("message") == "pong":
                    continue
                queue = self._pending.get(message.request_id)
                if queue is not None:
                    queue.put_nowait(message)
            if self._closed:
                return

            self._ready.clear()
            for queue in self._pending.values():
                queue.put_nowait(None)
            logger.debug(
                "agent-db socket dropped, reconnecting",
                extra={"database_id": self._database_id, "close_code": ws.close_code},
            )
            deadline = time.monotonic() + RECONNECT_BUDGET
            backoff = 0.1
            while not self._closed:
                try:
                    await self.connect()
                    break
                except Exception as e:
                    if time.monotonic() > deadline:
                        logger.error(
                            "could not reconnect to agent-db",
                            extra={"database_id": self._database_id, "error": str(e)},
                        )
                        self._close_reason = e
                        await self.aclose()
                        return
                await asyncio.sleep(backoff * (1 + random.random() / 4))
                backoff = min(backoff * 2, 5.0)

    async def _ping(self) -> None:
        # the server reads with a timeout, so an idle socket stays open only while it pings
        while not self._closed:
            await asyncio.sleep(self._ping_interval or 5.0)
            if not self._ready.is_set() or self._ws is None:
                continue
            ping = Wire.ClientMessage(
                request_id=next(self._ids), ping=Wire.Ping(timestamp_ms=int(time.time() * 1000))
            )
            with contextlib.suppress(Exception):
                await self._ws.send_bytes(ping.SerializeToString())

    async def _send(self, message: Message) -> None:
        data = message.SerializeToString()
        if len(data) > MAX_FRAME_BYTES:
            # the server drops the socket on an oversized frame, so resending it never ends
            raise StoreError("frame_too_large", f"a {len(data)} byte request is over the limit")
        if self._ws is None or self._ws.closed:
            raise _Disconnected
        try:
            await self._ws.send_bytes(data)
        except (ConnectionError, RuntimeError, aiohttp.ClientError) as e:
            raise _Disconnected from e

    async def _wait_ready(self) -> None:
        if self._closed:
            raise StoreError("closed", "the agent-db connection is closed") from self._close_reason
        waiter = asyncio.ensure_future(self._ready.wait())
        try:
            await waiter
        finally:
            waiter.cancel()
        if self._closed:
            raise StoreError("closed", "the agent-db connection is closed") from self._close_reason

    async def _call(self, build: Callable[[int], Message]) -> pb.AgentDB.Wire.ServerMessage:
        """Send one request and wait for its one reply, again across reconnects."""
        retry_until = time.monotonic() + RECONNECT_BUDGET
        while True:
            await self._wait_ready()
            request_id = next(self._ids)
            queue: asyncio.Queue[pb.AgentDB.Wire.ServerMessage | None] = asyncio.Queue()
            self._pending[request_id] = queue
            try:
                await self._send(build(request_id))
                reply = await queue.get()
            except _Disconnected:
                reply = None
            finally:
                self._pending.pop(request_id, None)
            if reply is not None and reply.WhichOneof("message") != "error":
                return reply
            if reply is not None and reply.error.code not in RETRYABLE_CODES:
                raise StoreError(reply.error.code, reply.error.message)
            if time.monotonic() > retry_until:
                raise StoreError("unavailable", "agent-db did not answer within the budget")
            await asyncio.sleep(RETRY_DELAY)

    async def exec(self, sql: str, *params: Value) -> ExecResult:
        reply = await self._call(
            lambda i: Wire.ClientMessage(request_id=i, exec=_statement(sql, params))
        )
        r = reply.exec_result
        return ExecResult(rows_affected=r.rows_affected, last_insert_id=r.last_insert_id, tip=r.tip)

    async def batch(self, *statements: Statement) -> ExecResult:
        batch = Wire.Batch(statements=[_statement(sql, list(params)) for sql, params in statements])
        reply = await self._call(lambda i: Wire.ClientMessage(request_id=i, batch=batch))
        return ExecResult(tip=reply.exec_result.tip)

    async def query(self, sql: str, *params: Value) -> AsyncIterator[Row]:
        statement = _statement(sql, params)
        retry_until = time.monotonic() + RECONNECT_BUDGET
        yielded = False
        while True:
            await self._wait_ready()
            request_id = next(self._ids)
            queue: asyncio.Queue[pb.AgentDB.Wire.ServerMessage | None] = asyncio.Queue()
            self._pending[request_id] = queue
            done = False
            names: list[str] = []
            try:
                # the query carries the initial window; each batch taken is replenished by one
                await self._send(Wire.ClientMessage(request_id=request_id, query=statement))
                while True:
                    message = await queue.get()
                    if message is None:
                        raise _Disconnected
                    kind = message.WhichOneof("message")
                    if kind == "columns":
                        names = list(message.columns.names)
                    elif kind == "column_batch":
                        credit = Wire.ClientMessage(
                            request_id=request_id, credit=Wire.Credit(batches=1)
                        )
                        with contextlib.suppress(_Disconnected):
                            await self._send(credit)
                        # each row names its own storage class per column, and text and blobs
                        # are sliced by exclusive end offsets
                        batch = message.column_batch
                        columns: list[list[Value]] = []
                        for column in batch.columns:
                            ints, doubles = iter(column.ints), iter(column.doubles)
                            texts, blobs = iter(column.text_ends), iter(column.blob_ends)
                            text_at = blob_at = 0
                            values: list[Value] = []
                            for i in range(batch.rows):
                                kind = column.types[i] if i < len(column.types) else Wire.NULL
                                if kind == Wire.INT:
                                    values.append(next(ints))
                                elif kind == Wire.DOUBLE:
                                    values.append(next(doubles))
                                elif kind == Wire.TEXT:
                                    end = next(texts)
                                    values.append(column.text_data[text_at:end].decode())
                                    text_at = end
                                elif kind == Wire.BLOB:
                                    end = next(blobs)
                                    values.append(bytes(column.blob_data[blob_at:end]))
                                    blob_at = end
                                else:
                                    values.append(None)
                            columns.append(values)
                        for i in range(batch.rows):
                            yielded = True
                            yield {name: columns[c][i] for c, name in enumerate(names)}
                    elif kind == "done":
                        done = True
                        return
                    elif kind == "error":
                        done = True
                        retryable = message.error.code in RETRYABLE_CODES
                        if retryable and not yielded and time.monotonic() < retry_until:
                            raise _Disconnected
                        raise StoreError(message.error.code, message.error.message)
            except _Disconnected:
                if yielded:
                    raise StoreError(
                        "unavailable", "the connection dropped in the middle of a query"
                    ) from None
                if time.monotonic() > retry_until:
                    raise StoreError(
                        "unavailable", "agent-db did not answer within the budget"
                    ) from None
                await asyncio.sleep(RETRY_DELAY)
                continue
            finally:
                self._pending.pop(request_id, None)
                if not done and self._ready.is_set():
                    # the caller stopped reading: the server stops streaming this one
                    with contextlib.suppress(Exception):
                        await self._send(
                            Wire.ClientMessage(request_id=request_id, cancel=Wire.Cancel())
                        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready.set()  # wakes waiters, which then see the executor is closed
        for queue in self._pending.values():
            queue.put_nowait(None)
        if self._pinger is not None:
            await aio.cancel_and_wait(self._pinger)
        if self._ws is not None:
            await self._ws.close()
        if self._supervisor is not None and self._supervisor is not asyncio.current_task():
            await aio.cancel_and_wait(self._supervisor)
        if self._http_session is not None:
            await self._http_session.close()


class AgentDB(_Store):
    """Sessions in agent-db: its management API mints databases, its data plane serves them.

    ``url`` defaults to ``LIVEKIT_AGENTDB_URL`` and the key to ``LIVEKIT_API_KEY``/``_SECRET``;
    ``ws_url`` defaults to ``url`` as ``ws``/``wss`` with ``/db`` appended.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        ws_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: float = LEASE_TTL,
    ) -> None:
        super().__init__(lease_ttl=lease_ttl)
        url = url or os.environ.get("LIVEKIT_AGENTDB_URL")
        api_key = api_key or os.environ.get("LIVEKIT_API_KEY")
        api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        if not url or not api_key or not api_secret:
            raise ValueError(
                "agent-db is not configured: pass url, api_key and api_secret, or set "
                "LIVEKIT_AGENTDB_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
            )
        self._url = url
        self._ws_url = ws_url or re.sub(r"^http", "ws", url.rstrip("/")) + "/db"
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = ""
        self._token_expires_at = 0.0
        self._http_session: aiohttp.ClientSession | None = None
        self._twirp: TwirpClient | None = None

    def _token(self) -> str:
        # one token serves every socket and call until it is near expiry
        if time.time() > self._token_expires_at:
            now = int(time.time())
            claims = {
                "iss": self._api_key,
                "sub": "livekit-agents",
                "nbf": now,
                "exp": now + int(TOKEN_TTL),
                "agent": {"databaseAdmin": True},
            }
            self._access_token = jwt.encode(claims, self._api_secret, algorithm="HS256")
            self._token_expires_at = time.time() + TOKEN_TTL / 2
        return self._access_token

    async def _connect(self, database_id: str) -> Executor:
        executor = AgentDBExecutor(ws_url=self._ws_url, database_id=database_id, token=self._token)
        try:
            await executor.connect()
        except BaseException:
            await executor.aclose()
            raise
        return executor

    async def _request(self, method: str, request: Message, response: type[_M]) -> _M:
        if self._twirp is None:
            from livekit.api.twirp_client import TwirpClient

            self._http_session = aiohttp.ClientSession()
            self._twirp = TwirpClient(self._http_session, self._url, "livekit", failover=False)
        reply = await self._twirp.request(
            "AgentDBService",
            method,
            request,
            {"Authorization": f"Bearer {self._token()}"},
            response,
        )
        assert isinstance(reply, response)
        return reply

    async def create_database(self, *, region: str = "", ttl_seconds: int = 0) -> str:
        """Mint a database and return its id. ``ttl_seconds`` unset means it never expires."""
        request = pb.AgentDB.CreateRequest(region=region, ttl_seconds=ttl_seconds)
        created = await self._request("CreateDatabase", request, pb.AgentDB.CreateResponse)
        return created.database_id

    async def get_database(self, database_id: str) -> pb.AgentDB.AgentDatabase:
        request = pb.AgentDB.GetRequest(database_id=database_id)
        return await self._request("GetDatabase", request, pb.AgentDB.AgentDatabase)

    async def list_databases(
        self, *, page_size: int = 0, page_token: str = ""
    ) -> pb.AgentDB.ListResponse:
        request = pb.AgentDB.ListRequest(page_size=page_size, page_token=page_token)
        return await self._request("ListDatabases", request, pb.AgentDB.ListResponse)

    async def delete_database(self, database_id: str) -> None:
        request = pb.AgentDB.DeleteRequest(database_id=database_id)
        await self._request("DeleteDatabase", request, pb.AgentDB.DeleteResponse)

    async def aclose(self) -> None:
        await super().aclose()
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = self._twirp = None


__all__ = ["AgentDB", "AgentDBExecutor"]
