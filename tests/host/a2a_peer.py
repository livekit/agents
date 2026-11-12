"""A local A2A peer (HTTP/SSE JSON-RPC): it records what it is sent and either completes each
task or holds it open until cancelled."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
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
        self.messages: list[Json] = []
        """Each message a ``SendStreamingMessage`` carried."""
        self.cancelled: list[str] = []
        """The id of each task a ``CancelTask`` named."""
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
        params = rpc["params"]
        if rpc["method"] == "CancelTask":
            self.cancelled.append(params["id"])
            result = {"id": params["id"], "contextId": params["contextId"]}
            result["status"] = {"state": "TASK_STATE_CANCELED"}
            await self._respond(writer, "application/json", _envelope(rpc, result))
            return
        message = params["message"]
        self.messages.append(message)
        self.received.set()
        context = message.get("contextId", "remote-context")
        task = {"id": "remote-task", "contextId": context}
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
        if self.fail is not None:
            events.append(_status(task, "FAILED", self.fail))
        else:
            answer = {"artifactId": "answer", "parts": [{"text": "Confirmed"}]}
            events.append(
                {
                    "artifactUpdate": {
                        "taskId": "remote-task",
                        "contextId": context,
                        "artifact": answer,
                    }
                }
            )
            events.append(_status(task, "COMPLETED", ""))
        stream = b"".join(b"data: " + _envelope(rpc, event) + b"\n\n" for event in events)
        await self._respond(writer, "text/event-stream", stream)

    def _card(self) -> Json:
        return {
            "name": "coverage_expert",
            "description": "Checks account coverage.",
            "version": "1",
            "supportedInterfaces": [
                {"url": self.url + "/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
            ],
            "capabilities": {"streaming": True, "extensions": [{"uri": URI}]},
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
    message = {"messageId": f"status-{state}", "role": "ROLE_AGENT", "parts": [{"text": text}]}
    status = {"state": f"TASK_STATE_{state}", "message": message}
    return {
        "statusUpdate": {"taskId": task["id"], "contextId": task["contextId"], "status": status}
    }


@pytest_asyncio.fixture
async def peer() -> AsyncIterator[Peer]:
    peer = Peer()
    server = await asyncio.start_server(peer.accept, "127.0.0.1", 0)
    peer.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/expert"
    try:
        yield peer
    finally:
        server.close()
        await peer.close()
        await server.wait_closed()
