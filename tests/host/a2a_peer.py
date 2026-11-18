"""A local A2A peer (HTTP/SSE JSON-RPC): it records what it is sent and either completes each
task or holds it open until cancelled."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import pytest_asyncio

URI = "https://livekit.io/a2a/ext/agents/v1"

Json = Any  # parsed JSON-RPC: whatever `json.loads` returns


class Peer:
    """``hold=True`` keeps each task working until it is cancelled or the peer closes;
    ``fail`` ends each task failed with that text."""

    def __init__(self) -> None:
        self.url = ""
        self.hold = False
        self.fail: str | None = None
        self.auth_required = False
        self.question: str | None = None
        self.direct_reply: Json | None = None
        self.task_snapshot: Json | None = None
        self.new_task_id = "remote-task"
        """Task returned for unaddressed work; a continuation keeps its addressed ID."""
        self.events: Callable[[Json], Sequence[Json | asyncio.Event]] | None = None
        """Scripted stream events per message; an ``asyncio.Event`` among them holds the stream
        open there until it is set or the peer closes."""
        self.response_gate: asyncio.Event | None = None
        """An optional boundary holding a stream after its first admission event."""
        self.gated_after = 0
        """The index of the event the ``response_gate`` holds the stream after."""
        self.input_schema: dict[str, Json] | None = None
        self.messages: list[Json] = []
        """Each message a ``SendStreamingMessage`` carried."""
        self.methods: list[str] = []
        self.cancelled: list[str] = []
        """The id of each task a ``CancelTask`` named."""
        self.cancel_reply = "canceled"
        """How a ``CancelTask`` ends: ``canceled``; ``refused`` (not cancelable); ``lost``, which
        cancels the task but drops the reply; or ``lost_unconfirmed``, which drops the reply and
        leaves the task working. A later ``GetTask`` reads the task's state."""
        self._task_states: dict[str, Json] = {}
        self.received = asyncio.Event()
        self._release = asyncio.Event()
        self._connections: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()

    async def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._connections.add(task)
        self._writers.add(writer)
        try:
            await self._serve(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            self._writers.discard(writer)
            self._connections.discard(task)

    async def close(self) -> None:
        self._release.set()
        if self.response_gate is not None:
            self.response_gate.set()
        for writer in tuple(self._writers):
            writer.close()
        await asyncio.gather(*tuple(self._connections))

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        method = (await reader.readline()).decode().split()[0]
        headers: dict[str, str] = {}
        while (line := await reader.readline()) != b"\r\n":
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        body = bytearray()
        if headers.get("transfer-encoding") == "chunked":
            while size := int((await reader.readline()).split(b";", 1)[0], 16):
                body.extend(await reader.readexactly(size))
                await reader.readexactly(2)
            await reader.readline()
        elif length := int(headers.get("content-length", "0")):
            body.extend(await reader.readexactly(length))
        if method == "GET":
            await self._respond(writer, "application/json", json.dumps(self._card()).encode())
            return
        rpc: Json = json.loads(body)
        self.methods.append(rpc["method"])
        params = rpc["params"]
        if rpc["method"] == "CancelTask":
            self.cancelled.append(params["id"])
            result = {"id": params["id"], "contextId": params["contextId"]}
            if self.cancel_reply == "refused":
                error = {"code": -32002, "message": "This task can no longer be canceled"}
                body = json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "error": error})
                await self._respond(writer, "application/json", body.encode())
                return
            state = "WORKING" if self.cancel_reply == "lost_unconfirmed" else "CANCELED"
            result["status"] = {"state": f"TASK_STATE_{state}"}
            self._task_states[params["id"]] = result
            if self.cancel_reply == "canceled":
                await self._respond(writer, "application/json", _envelope(rpc, result))
            return
        if rpc["method"] == "GetTask" and params.get("id") in self._task_states:
            result = self._task_states[params["id"]]
            await self._respond(writer, "application/json", _envelope(rpc, result))
            return
        if rpc["method"] == "GetTask" and self.task_snapshot is not None:
            await self._respond(writer, "application/json", _envelope(rpc, self.task_snapshot))
            return
        if rpc["method"] not in ("SendStreamingMessage", "SendMessage"):
            await self._respond(
                writer,
                "application/json",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": rpc["id"],
                        "error": {"code": -32601, "message": "Method not supported by this peer"},
                    }
                ).encode(),
            )
            return
        message = params["message"]
        self.messages.append(message)
        self.received.set()
        if self.events is not None:
            scripted = list(self.events(message))
            if self.response_gate is not None:
                scripted.insert(self.gated_after + 1, self.response_gate)
            if not any(isinstance(event, asyncio.Event) for event in scripted):
                payload = b"".join(b"data: " + _envelope(rpc, e) + b"\n\n" for e in scripted)
                await self._respond(writer, "text/event-stream", payload)
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            for event in scripted:
                if isinstance(event, asyncio.Event):
                    await self._until(event)
                    continue
                data = b"data: " + _envelope(rpc, event) + b"\n\n"
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            return
        if self.direct_reply is not None:
            await self._respond(
                writer, "application/json", _envelope(rpc, {"message": self.direct_reply})
            )
            return
        context = message.get("contextId", "remote-context")
        task = {"id": message.get("taskId", self.new_task_id), "contextId": context}
        events: list[Json] = [
            {"task": {**task, "status": {"state": "TASK_STATE_SUBMITTED"}}},
            _status(task, "WORKING", "Checking"),
        ]
        if self.hold:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            for event in events:
                data = b"data: " + _envelope(rpc, event) + b"\n\n"
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()
            await self._release.wait()
            return
        if self.auth_required:
            events.append(_status(task, "AUTH_REQUIRED", "Sign in to continue."))
        elif self.question is not None and "taskId" not in message:
            events.append(_status(task, "INPUT_REQUIRED", self.question))
        elif self.fail is not None:
            events.append(_status(task, "FAILED", self.fail))
        else:
            answer = {"artifactId": "answer", "parts": [{"text": "Confirmed"}]}
            events.append(
                {
                    "artifactUpdate": {
                        "taskId": task["id"],
                        "contextId": context,
                        "artifact": answer,
                    }
                }
            )
            events.append(_status(task, "COMPLETED", ""))
        stream = b"".join(b"data: " + _envelope(rpc, event) + b"\n\n" for event in events)
        await self._respond(writer, "text/event-stream", stream)

    async def _until(self, gate: asyncio.Event) -> None:
        """Waits for ``gate``; closing the peer releases every held stream."""
        waits = [asyncio.ensure_future(gate.wait()), asyncio.ensure_future(self._release.wait())]
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()

    def _card(self) -> Json:
        extension: Json = {"uri": URI}
        if self.input_schema is not None:
            extension["params"] = {"inputSchema": self.input_schema}
        return {
            "name": "coverage_expert",
            "description": "Checks account coverage.",
            "version": "1",
            "supportedInterfaces": [
                {"url": self.url + "/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
            ],
            "capabilities": {"streaming": True, "extensions": [extension]},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [],
        }

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, kind: str, body: bytes) -> None:
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()


def _envelope(rpc: Json, result: Json) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": rpc["id"], "result": result}).encode()


def _status(task: dict[str, str], state: str, text: str) -> Json:
    message: Json = {
        "messageId": f"status-{task['id']}-{state}",
        "role": "ROLE_AGENT",
        "parts": [{"text": text}],
    }
    if state == "WORKING":
        # This peer's "Checking" is explicit conversational progress, like ctx.update().
        message["metadata"] = {f"{URI}/progress": True}
    status = {"state": f"TASK_STATE_{state}", "message": message}
    return {
        "statusUpdate": {"taskId": task["id"], "contextId": task["contextId"], "status": status}
    }


@contextlib.asynccontextmanager
async def serving() -> AsyncIterator[Peer]:
    """One peer on its own port, so its URL is a distinct endpoint."""
    peer = Peer()
    server = await asyncio.start_server(peer.accept, "127.0.0.1", 0)
    peer.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/expert"
    try:
        yield peer
    finally:
        server.close()
        await peer.close()
        await server.wait_closed()


@pytest_asyncio.fixture
async def peer() -> AsyncIterator[Peer]:
    async with serving() as peer:
        yield peer
