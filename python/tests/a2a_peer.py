"""An owned local A2A HTTP/SSE peer: it records each request and answers by scenario."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio

URI = "https://livekit.io/a2a/ext/agents/v1"


class Peer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any] | None]] = []
        self.scenario = "question"
        self.capable = True
        self.structured = False
        self.url = ""
        self.sockets: set[asyncio.StreamWriter] = set()
        self.tasks: set[asyncio.Task[None]] = set()
        self.release = asyncio.Event()
        self.canceled = asyncio.Event()

    async def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        self.sockets.add(writer)
        try:
            await self.serve(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            self.sockets.discard(writer)
            self.tasks.discard(task)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        method, _, _ = (await reader.readline()).decode().split()
        headers: dict[str, str] = {}
        while (line := await reader.readline()) != b"\r\n":
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        body = bytearray()
        if headers.get("transfer-encoding") == "chunked":
            while size := int((await reader.readline()).split(b";", 1)[0], 16):
                body.extend(await reader.readexactly(size))
                assert await reader.readexactly(2) == b"\r\n"
            assert await reader.readline() == b"\r\n"
        elif length := int(headers.get("content-length", "0")):
            body.extend(await reader.readexactly(length))
        rpc = json.loads(body) if body else None
        self.requests.append((method, headers, rpc))
        if method == "GET":
            if self.scenario == "stall-card":
                await self.release.wait()
                return
            extensions = []
            if self.capable:
                extension: dict[str, Any] = {"uri": URI}
                if self.structured:
                    extension["params"] = {
                        "toolName": "check_coverage",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"account": {"type": "integer"}},
                            "required": ["account"],
                            "additionalProperties": False,
                        },
                    }
                extensions.append(extension)
            card = {
                "name": "coverage_expert",
                "description": "Checks account coverage.",
                "version": "1",
                "supportedInterfaces": [
                    {
                        "url": self.url + "/rpc",
                        "protocolBinding": "JSONRPC",
                        "protocolVersion": "1.0",
                    }
                ],
                "capabilities": {"streaming": True, "extensions": extensions},
                "defaultInputModes": ["text/plain"],
                "defaultOutputModes": ["text/plain"],
                "skills": [],
            }
            await self.respond(writer, "application/json", json.dumps(card).encode())
            return
        assert rpc is not None
        params = rpc["params"]
        if rpc["method"] == "CancelTask":
            self.canceled.set()
            result = {
                "id": params["id"],
                "contextId": params["contextId"],
                "status": {"state": "TASK_STATE_CANCELED"},
            }
            await self.respond(writer, "application/json", self.envelope(rpc, result))
            return
        message = params.get("message", {})
        context = message.get("contextId", params.get("contextId", "remote-context"))
        task_id = message.get("taskId", params.get("id", "remote-task"))
        events = [
            {
                "task": {
                    "id": task_id,
                    "contextId": context,
                    "status": {"state": "TASK_STATE_SUBMITTED"},
                }
            }
        ]
        events.append(self.status(task_id, context, "WORKING", "Checking café"))
        if self.scenario in ("wait", "partial-stall"):
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            for event in events:
                data = b"data: " + self.envelope(rpc, event) + b"\n\n"
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()
            await self.release.wait()
            return
        if self.scenario == "flood":
            events.extend(
                self.status(task_id, context, "WORKING", f"progress {i}") for i in range(40)
            )
        if self.scenario == "failed":
            events.append(self.status(task_id, context, "FAILED", "coverage unavailable"))
        elif self.scenario == "canceled":
            events.append(self.status(task_id, context, "CANCELED", "stopped"))
        elif self.scenario == "question" and not message.get("taskId"):
            events.append(self.status(task_id, context, "INPUT_REQUIRED", "Which account?"))
        else:
            events.append(
                {
                    "artifactUpdate": {
                        "taskId": task_id,
                        "contextId": context,
                        "artifact": {
                            "artifactId": "answer",
                            "name": "answer",
                            "parts": [{"text": "Confirmed "}, {"data": {"receipt": 42}}],
                        },
                    }
                }
            )
            final = self.status(task_id, context, "COMPLETED", "")
            final["statusUpdate"]["metadata"] = {URI + "/verbatim": "Confirmed "}
            events.append(final)
        wire = b"".join(b"data: " + self.envelope(rpc, event) + b"\n\n" for event in events)
        await self.respond(writer, "text/event-stream", wire)

    @staticmethod
    def envelope(rpc: dict[str, Any], result: object) -> bytes:
        return json.dumps(
            {"jsonrpc": "2.0", "id": rpc["id"], "result": result}, ensure_ascii=False
        ).encode()

    @staticmethod
    def status(task: str, context: str, state: str, text: str) -> dict[str, Any]:
        return {
            "statusUpdate": {
                "taskId": task,
                "contextId": context,
                "status": {
                    "state": "TASK_STATE_" + state,
                    "message": {
                        "messageId": "status-" + state,
                        "role": "ROLE_AGENT",
                        "parts": [{"text": text}],
                    },
                },
            }
        }

    @staticmethod
    async def respond(writer: asyncio.StreamWriter, kind: str, body: bytes) -> None:
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        )
        for offset in range(0, len(body), 7):
            writer.write(body[offset : offset + 7])
            await writer.drain()


@pytest_asyncio.fixture
async def peer() -> AsyncIterator[Peer]:
    peer = Peer()
    server = await asyncio.start_server(peer.accept, "127.0.0.1", 0)
    assert server.sockets
    peer.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/expert"
    try:
        yield peer
    finally:
        # wait_closed() waits for open connections: release the stalled handlers first
        server.close()
        peer.release.set()
        for writer in tuple(peer.sockets):
            writer.close()
        await asyncio.gather(*tuple(peer.tasks))
        await server.wait_closed()
