"""A local agentstore that speaks the wire of the Rust storage client (``lk-agents-storage``).

The data plane is the ``AgentDB.Wire`` WebSocket, run on one in-memory SQLite database per
agentstore database with the client's SQL unchanged; the answer streams are catch-up GETs and
live SSE; threads are the public API's REST+JSON under project ``p_``. Any well-formed bearer
token is accepted.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import functools
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

from .livekit_agentdb_pb2 import AgentDB

Wire = AgentDB.Wire

VALUE_TYPES = {
    type(None): Wire.NULL,
    int: Wire.INT,
    float: Wire.DOUBLE,
    str: Wire.TEXT,
    bytes: Wire.BLOB,
}


def failure(status: int, code: str, message: str) -> web.Response:
    """An error body as the public API spells it, its code kept by the client."""
    return web.json_response({"code": code, "message": message}, status=status)


def timestamp(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


class Database:
    """One agentstore database: a SQLite connection and its commit counter, the ``tip``."""

    def __init__(self) -> None:
        self.sql = sqlite3.connect(":memory:", isolation_level=None)
        self.tip = 0

    def answer(self, msg: Any) -> list[Any]:
        """The server messages one client message is answered with, in order."""
        kind = msg.WhichOneof("message")
        if kind == "ping":
            now = int(time.time() * 1000)
            pong = Wire.Pong(last_ping_timestamp_ms=msg.ping.timestamp_ms, timestamp_ms=now)
            return [Wire.ServerMessage(pong=pong)]
        if kind in ("credit", "cancel"):
            return []  # every row goes out at once, so there is nothing to pace or stop
        if kind not in ("exec", "query", "batch"):
            error = Wire.Error(code="protocol", message=f"unsupported message {kind}")
            return [Wire.ServerMessage(error=error)]

        changes = self.sql.total_changes
        try:
            if kind == "batch":
                # every statement or none: the first failure rolls the batch back
                self.sql.execute("BEGIN IMMEDIATE")
                try:
                    statements = enumerate(msg.batch.statements)
                    out = [m for i, st in statements for m in self.results(st, i)]
                except sqlite3.Error:
                    self.sql.execute("ROLLBACK")
                    raise
                self.sql.execute("COMMIT")
            else:
                out = self.results(getattr(msg, kind), 0)
        except sqlite3.Error as e:
            error = Wire.Error(code="query" if kind == "query" else "exec", message=str(e))
            return [Wire.ServerMessage(error=error)]
        if self.sql.total_changes != changes:
            self.tip += 1
        if kind == "exec":
            out[0].exec_result.tip = self.tip
            return out
        rows = sum(m.column_batch.rows for m in out if m.HasField("column_batch"))
        return [*out, Wire.ServerMessage(done=Wire.Done(tip=self.tip, total_rows=rows))]

    def results(self, statement: Any, index: int) -> list[Any]:
        """One statement run: its ``ExecResult``, or its ``Columns`` and rows."""
        params = []
        for p in statement.params:
            kind = p.WhichOneof("value")
            params.append(None if kind in (None, "null_value") else getattr(p, kind))
        cur = self.sql.execute(statement.sql, params)
        rows = cur.fetchall()
        if cur.description is None:
            result = Wire.ExecResult(
                rows_affected=max(cur.rowcount, 0),
                last_insert_id=cur.lastrowid or 0,
                statement=index,
            )
            return [Wire.ServerMessage(exec_result=result)]
        names = [d[0] for d in cur.description]
        out = [Wire.ServerMessage(columns=Wire.Columns(names=names, statement=index))]
        if not rows:
            return out
        # one column-oriented batch, each value tagged with its SQLite type
        batch = Wire.ColumnBatch(rows=len(rows), statement=index)
        for i in range(len(names)):
            column = batch.columns.add()
            types, text, blob = bytearray(), bytearray(), bytearray()
            for row in rows:
                value = row[i]
                types.append(VALUE_TYPES[type(value)])
                if isinstance(value, int):
                    column.ints.append(value)
                elif isinstance(value, float):
                    column.doubles.append(value)
                elif isinstance(value, str):
                    text += value.encode()
                    column.text_ends.append(len(text))
                elif isinstance(value, bytes):
                    blob += value
                    column.blob_ends.append(len(blob))
            column.types, column.text_data, column.blob_data = (
                bytes(types),
                bytes(text),
                bytes(blob),
            )
        return [*out, Wire.ServerMessage(column_batch=batch)]


@dataclass
class Stream:
    """One answer stream: its JSON records in order, and whether it is closed."""

    records: list[bytes] = field(default_factory=list)
    closed: bool = False
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)


served: collections.Counter[str] = collections.Counter()
"""The records each stream served to its readers, by stream id."""


def sse(event: str, data: bytes) -> bytes:
    lines = b"".join(b"data: " + line + b"\n" for line in data.split(b"\n"))
    return b"event: " + event.encode() + b"\n" + lines + b"\n"


class FakeAgentstore:
    """The public API's agentstore surface: threads, the data plane and the streams."""

    def __init__(self) -> None:
        self.databases: dict[str, Database] = {}
        self.threads: dict[str, dict[str, Any]] = {}
        self.streams: dict[str, Stream] = {}
        self.app = web.Application(middlewares=[self.bearer])
        self.app.add_routes(
            [
                web.post("/v0/projects/p_/threads", self.create_thread),
                web.get("/v0/projects/p_/threads/{id}", self.get_thread),
                web.patch("/v0/projects/p_/threads/{id}", self.update_thread),
                web.get("/v0/agentstore/db", self.data_plane),
                web.post("/v0/agentstore/streams", self.create_stream),
                web.post("/v0/agentstore/streams/{id}", self.append),
                web.get("/v0/agentstore/streams/{id}", self.read),
            ]
        )

    @web.middleware
    async def bearer(self, request: web.Request, handler: Any) -> web.StreamResponse:
        """Any token of three dot-separated parts, from the header or the WebSocket upgrade."""
        auth = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if (auth or request.query.get("access_token", "")).count(".") != 2:
            return failure(401, "unauthenticated", "no bearer token")
        response: web.StreamResponse = await handler(request)
        return response

    # -- threads ----------------------------------------------------------------------------

    async def create_thread(self, request: web.Request) -> web.Response:
        body = await request.json()
        database_id = "DB_" + secrets.token_hex(6)
        self.databases[database_id] = Database()
        now = time.time()
        thread = {
            "threadId": "AT_" + secrets.token_hex(6),
            "scope": body.get("scope", ""),
            "attributes": body.get("attributes", {}),
            "createdAt": timestamp(now),
            "databaseId": database_id,
        }
        self.threads[thread["threadId"]] = thread
        return self.touch(thread, body.get("idleTtl"))

    async def get_thread(self, request: web.Request) -> web.Response:
        if (thread := self.threads.get(request.match_info["id"])) is None:
            return failure(404, "not_found", "thread not found")
        return web.json_response({"thread": thread})

    async def update_thread(self, request: web.Request) -> web.Response:
        if (thread := self.threads.get(request.match_info["id"])) is None:
            return failure(404, "not_found", "thread not found")
        body = await request.json()
        for key, value in body.get("attributes", {}).items():
            if value:
                thread["attributes"][key] = value
            else:
                thread["attributes"].pop(key, None)
        return self.touch(thread, body.get("idleTtl"))

    def touch(self, thread: dict[str, Any], idle_ttl: str | None) -> web.Response:
        """Marks the thread active, with its expiry pushed out by its idle TTL when it has one."""
        now = time.time()
        thread["lastActive"] = timestamp(now)
        if idle_ttl:
            thread["idleTtl"] = idle_ttl
        if "idleTtl" in thread:
            thread["expiresAt"] = timestamp(now + float(thread["idleTtl"].rstrip("s")))
        return web.json_response({"thread": thread})

    # -- the data plane ---------------------------------------------------------------------

    async def data_plane(self, request: web.Request) -> web.WebSocketResponse:
        """``Hello`` names the database; every later frame is answered in the order it came."""
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        database: Database | None = None
        async for frame in ws:
            if frame.type != WSMsgType.BINARY:
                continue
            msg = Wire.ClientMessage.FromString(frame.data)
            if database is not None:
                replies = database.answer(msg)
            elif msg.HasField("hello") and msg.hello.database_id in self.databases:
                database = self.databases[msg.hello.database_id]
                hello_ok = Wire.HelloOk(tip=database.tip, ping_interval_ms=0)
                replies = [Wire.ServerMessage(hello_ok=hello_ok)]
            else:
                error = Wire.Error(code="not_found", message="database not found")
                replies = [Wire.ServerMessage(error=error)]
            for reply in replies:
                reply.request_id = msg.request_id
                await ws.send_bytes(reply.SerializeToString())
            if database is None:
                break
        await ws.close()
        return ws

    # -- the answer streams -----------------------------------------------------------------

    async def create_stream(self, request: web.Request) -> web.Response:
        body = await request.json()
        stream_id = "AST_" + secrets.token_hex(6)
        self.streams[stream_id] = Stream()
        expires_at = int(time.time()) + int(body.get("ttl_seconds", 86400))
        return web.json_response({"stream_id": stream_id, "expires_at": expires_at})

    async def append(self, request: web.Request) -> web.Response:
        """Appends the body as one record; ``Stream-Closed: true`` closes the stream."""
        if (stream := self.streams.get(request.match_info["id"])) is None:
            return failure(404, "not_found", "stream not found")
        record = await request.read()
        if record and stream.closed:
            return failure(409, "failed_precondition", "stream closed")
        if record:
            stream.records.append(record)
        stream.closed |= request.headers.get("Stream-Closed", "").lower() == "true"
        async with stream.changed:
            stream.changed.notify_all()
        return web.Response(headers={"Stream-Offset": str(len(stream.records))})

    async def read(self, request: web.Request) -> web.StreamResponse:
        """Records from ``offset`` (``-1`` the start, else a record index): a JSON array up to
        the tail, or with ``live=sse`` ``data`` and ``control`` events until the stream closes."""
        if (stream := self.streams.get(request.match_info["id"])) is None:
            return failure(404, "not_found", "stream not found")
        offset = request.query.get("offset", "-1")
        start = 0 if offset == "-1" else int(offset)
        if request.query.get("live") != "sse":
            headers = {"Stream-Next-Offset": str(len(stream.records)), "Stream-Up-To-Date": "true"}
            if stream.closed:
                headers["Stream-Closed"] = "true"
            served[request.match_info["id"]] += len(stream.records[start:])
            body = b"[" + b",".join(stream.records[start:]) + b"]"
            return web.Response(body=body, headers=headers, content_type="application/json")

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        while True:
            async with stream.changed:
                await stream.changed.wait_for(
                    lambda n=start: len(stream.records) > n or stream.closed
                )
            records, closed = stream.records[start:], stream.closed
            start += len(records)
            served[request.match_info["id"]] += len(records)
            if records:
                await response.write(sse("data", b"[" + b",".join(records) + b"]"))
            control = {"streamNextOffset": str(start), "upToDate": True, "streamClosed": closed}
            await response.write(sse("control", json.dumps(control).encode()))
            if closed:
                return response


@functools.cache
def serve() -> str:
    """The fake's base URL; the first call starts it on a loop and thread of its own."""
    started: concurrent.futures.Future[str] = concurrent.futures.Future()

    def run() -> None:
        loop = asyncio.new_event_loop()
        runner = web.AppRunner(FakeAgentstore().app, access_log=None)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        started.set_result(f"http://127.0.0.1:{runner.addresses[0][1]}")
        loop.run_forever()

    threading.Thread(target=run, name="fake-agentstore", daemon=True).start()
    return started.result(timeout=10)
