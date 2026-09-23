"""The agent-db client: its data plane as an ``Executor``, and its management API.

agent-db speaks its own protocol rather than HTTP: one binary protobuf message per WebSocket
frame, ``Hello`` first, requests multiplexed by a client-chosen id, streamed results paced by
credits. The server severs sockets on drain and eviction and marks those failures retryable,
so reconnecting is the normal path here, not an error case.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import random
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import aiohttp
import jwt

from ..log import logger
from ..utils import aio
from ._proto import livekit_agentdb_pb2 as pb
from .executor import ExecResult, Row, Statement, StoreError, Value

if TYPE_CHECKING:
    from google.protobuf.message import Message

Wire = pb.AgentDB.Wire

TOKEN_TTL = 6 * 3600.0
MAX_FRAME_BYTES = 16 << 20
"""The server's cap on one inbound frame; results come back under the same bound."""

RETRYABLE_CODES = frozenset({"unavailable"})
"""Codes the server uses for a database that is moving: the request is sent again, on the
new socket once the old one is severed."""
RETRY_DELAY = 0.1


def access_token(api_key: str, api_secret: str, *, identity: str, ttl: float = TOKEN_TTL) -> str:
    """Sign a token carrying the ``agent.databaseAdmin`` grant both planes require."""
    now = int(time.time())
    claims = {
        "iss": api_key,
        "sub": identity,
        "nbf": now,
        "exp": now + int(ttl),
        "agent": {"databaseAdmin": True},
    }
    return jwt.encode(claims, api_secret, algorithm="HS256")


class _Disconnected(Exception):
    """The socket went away with the request in flight."""


def _to_wire(value: Value) -> pb.AgentDB.Wire.Value:
    if value is None:
        return Wire.Value(null_value=True)
    if isinstance(value, bool):
        return Wire.Value(int_value=int(value))
    if isinstance(value, int):
        return Wire.Value(int_value=value)
    if isinstance(value, float):
        return Wire.Value(double_value=value)
    if isinstance(value, str):
        return Wire.Value(text_value=value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Wire.Value(blob_value=bytes(value))
    raise TypeError(f"cannot bind a {type(value).__name__} as a SQL parameter")


def _statement(sql: str, params: tuple[Value, ...] | list[Value]) -> pb.AgentDB.Wire.Statement:
    return Wire.Statement(sql=sql, params=[_to_wire(p) for p in params])


def _decode_batch(names: list[str], batch: pb.AgentDB.Wire.ColumnBatch) -> list[Row]:
    """Rows from a column batch: each row names its own storage class per column."""
    columns: list[list[Value]] = []
    for column in batch.columns:
        ints, doubles = iter(column.ints), iter(column.doubles)
        texts, blobs = iter(column.text_ends), iter(column.blob_ends)
        text_at = blob_at = 0
        values: list[Value] = []
        for row in range(batch.rows):
            kind = column.types[row] if row < len(column.types) else Wire.NULL
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
    return [{name: columns[i][row] for i, name in enumerate(names)} for row in range(batch.rows)]


class AgentDBExecutor:
    """An ``Executor`` on one agent-db database, over one WebSocket.

    A request interrupted by a reconnect is sent again on the new socket, so the statements
    run through it must be idempotent; every statement the store issues is. A query that has
    already yielded rows cannot be replayed, and fails instead.
    """

    def __init__(
        self,
        *,
        ws_url: str,
        database_id: str,
        token: Callable[[], str],
        http_session: aiohttp.ClientSession | None = None,
        reconnect_budget: float = 60.0,
    ) -> None:
        self._ws_url = ws_url
        self._database_id = database_id
        self._token = token
        self._http_session = http_session
        self._owns_http_session = http_session is None
        self._reconnect_budget = reconnect_budget

        self._ids = itertools.count(1)
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[int, asyncio.Queue[pb.AgentDB.Wire.ServerMessage | None]] = {}
        # set while a Hello has completed on the live socket; cleared across a reconnect so
        # a request made then waits instead of failing
        self._ready = asyncio.Event()
        self._closed = False
        self._close_reason: Exception | None = None
        self._ping_interval = 0.0
        self._supervisor: asyncio.Task[None] | None = None
        self._pinger: asyncio.Task[None] | None = None

    @property
    def database_id(self) -> str:
        return self._database_id

    async def connect(self) -> None:
        """Dial and say Hello. The first connect raises, so a bad URL or token is heard here."""
        await self._connect()
        self._supervisor = asyncio.create_task(self._supervise(), name="agentdb_supervise")
        self._pinger = asyncio.create_task(self._ping(), name="agentdb_ping")

    async def _connect(self) -> None:
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
            if not await self._redial():
                return

    async def _redial(self) -> bool:
        deadline = time.monotonic() + self._reconnect_budget
        backoff = 0.1
        while not self._closed:
            try:
                await self._connect()
                return True
            except Exception as e:
                if time.monotonic() > deadline:
                    logger.error(
                        "could not reconnect to agent-db",
                        extra={"database_id": self._database_id, "error": str(e)},
                    )
                    self._close_reason = e
                    await self.aclose()
                    return False
            await asyncio.sleep(backoff * (1 + random.random() / 4))
            backoff = min(backoff * 2, 5.0)
        return False

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
        if self._ws is None or self._ws.closed:
            raise _Disconnected
        try:
            await self._ws.send_bytes(message.SerializeToString())
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
        retry_until = time.monotonic() + self._reconnect_budget
        while True:
            await self._wait_ready()
            request_id = next(self._ids)
            queue: asyncio.Queue[pb.AgentDB.Wire.ServerMessage | None] = asyncio.Queue()
            self._pending[request_id] = queue
            try:
                await self._send(build(request_id))
                reply = await queue.get()
            except _Disconnected:
                await asyncio.sleep(RETRY_DELAY)
                continue
            finally:
                self._pending.pop(request_id, None)
            if reply is None:
                continue
            if reply.WhichOneof("message") == "error":
                if reply.error.code in RETRYABLE_CODES and time.monotonic() < retry_until:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                raise StoreError(reply.error.code, reply.error.message)
            return reply

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
        retry_until = time.monotonic() + self._reconnect_budget
        yielded = False
        while True:
            await self._wait_ready()
            request_id = next(self._ids)
            queue: asyncio.Queue[pb.AgentDB.Wire.ServerMessage | None] = asyncio.Queue()
            self._pending[request_id] = queue
            done = False
            names: list[str] = []
            try:
                # the server grants the initial window with the query; each consumed batch
                # is replenished by one
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
                        for row in _decode_batch(names, message.column_batch):
                            yielded = True
                            yield row
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
        if self._owns_http_session and self._http_session is not None:
            await self._http_session.close()


class AgentDBService:
    """The management API: create, look up, list and delete databases. Twirp over HTTP."""

    def __init__(
        self,
        url: str,
        *,
        token: Callable[[], str],
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        from livekit.api.twirp_client import TwirpClient

        self._http_session = http_session or aiohttp.ClientSession()
        self._owns_http_session = http_session is None
        self._client = TwirpClient(self._http_session, url, "livekit", failover=False)
        self._token = token

    async def _request(self, method: str, request: Message, response: type[Message]) -> Message:
        return await self._client.request(
            "AgentDBService",
            method,
            request,
            {"Authorization": f"Bearer {self._token()}"},
            response,
        )

    async def create_database(
        self, *, region: str = "", ttl_seconds: int = 0
    ) -> pb.AgentDB.CreateResponse:
        request = pb.AgentDB.CreateRequest(region=region, ttl_seconds=ttl_seconds)
        response = await self._request("CreateDatabase", request, pb.AgentDB.CreateResponse)
        assert isinstance(response, pb.AgentDB.CreateResponse)
        return response

    async def get_database(self, database_id: str) -> pb.AgentDB.AgentDatabase:
        request = pb.AgentDB.GetRequest(database_id=database_id)
        response = await self._request("GetDatabase", request, pb.AgentDB.AgentDatabase)
        assert isinstance(response, pb.AgentDB.AgentDatabase)
        return response

    async def list_databases(
        self, *, page_size: int = 0, page_token: str = ""
    ) -> pb.AgentDB.ListResponse:
        request = pb.AgentDB.ListRequest(page_size=page_size, page_token=page_token)
        response = await self._request("ListDatabases", request, pb.AgentDB.ListResponse)
        assert isinstance(response, pb.AgentDB.ListResponse)
        return response

    async def delete_database(self, database_id: str) -> None:
        request = pb.AgentDB.DeleteRequest(database_id=database_id)
        await self._request("DeleteDatabase", request, pb.AgentDB.DeleteResponse)

    async def aclose(self) -> None:
        if self._owns_http_session:
            await self._http_session.close()


__all__ = ["AgentDBExecutor", "AgentDBService", "access_token"]
