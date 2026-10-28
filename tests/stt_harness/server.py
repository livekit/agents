"""A local HTTP/WebSocket peer with independent input and output tasks."""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from .protocol import Provider, Transcript


@dataclass(frozen=True)
class Segment:
    audio: bytes
    text: str


@dataclass(frozen=True)
class Reply:
    data: dict[str, Any] | bytes | str | None
    final: bool = False
    close_code: int = 1000


class Connection:
    def __init__(self, server: STTServer, request: web.Request, ws: web.WebSocketResponse):
        self.server = server
        self.request = request
        assert request.transport is not None
        self.transport = request.transport
        self.ws = ws
        self.protocol = server.provider.protocol(self)
        self.audio = bytearray()
        self.base_offset = len(server.received_audio)
        self.final_offset = 0
        self.final_index = 0
        self.input_ended = False
        self.disconnected = asyncio.Event()
        self.replies: asyncio.Queue[Reply] = asyncio.Queue()
        self.messages: list[aiohttp.WSMessage] = []
        self.pings = 0
        self.pongs = 0
        self.audio_received = asyncio.Event()
        self.pong_received = asyncio.Event()

    def send(self, data: dict[str, Any] | bytes | str, *, final: bool = False) -> None:
        self.replies.put_nowait(Reply(data, final))

    async def acknowledge(self, data: dict[str, Any]) -> None:
        """A receipt acknowledgement may overtake delayed processing results."""
        await self.ws.send_json(data)

    def close(self, code: int = 1000) -> None:
        self.replies.put_nowait(Reply(None, close_code=code))

    def abort(self) -> None:
        self.transport.abort()

    async def accept_audio(self, data: bytes) -> None:
        assert self.protocol.configured, "audio arrived before configuration"
        assert not self.input_ended, "audio arrived after end of input"
        self.audio.extend(data)
        self.server.received_audio.extend(data)
        expected = self.server.expected_audio
        if expected is not None:
            assert expected.startswith(self.server.received_audio), (
                "audio bytes differ from the input"
            )
        self.server.audio_received.set()
        self.audio_received.set()
        if self.server.fault_after_audio:
            if self.server.fault_after_audio == "abort":
                self.abort()
            else:
                self.close()
            return
        if self.server.processing_stalled:
            return
        boundaries = []
        offset = 0
        for segment in self.server.segments or ():
            offset += len(segment.audio)
            boundaries.append(offset)
        if self.server.automatic_finals and any(
            self.base_offset + self.final_offset < boundary <= len(self.server.received_audio)
            for boundary in boundaries
        ):
            await self.finalize()
        else:
            await self.protocol.send_transcript(self.result(final=False))

    def result(self, *, final: bool) -> Transcript:
        end = len(self.audio)
        offset = 0
        words = []
        for segment in self.server.segments or ():
            start = offset
            offset += len(segment.audio)
            if self.base_offset + self.final_offset < offset <= self.base_offset + end:
                words.append(segment.text)
            elif not final and start < self.base_offset + end < offset:
                words.append(segment.text[:1])
                break
        result = Transcript(
            text=" ".join(words),
            final=final,
            start=self.final_offset / (2 * self.server.provider.sample_rate),
            end=end / (2 * self.server.provider.sample_rate),
            index=self.final_index,
        )
        if final:
            self.final_offset = end
            self.final_index += 1
        return result

    async def finalize(self) -> None:
        self.server.final_requested.set()
        if self.final_offset < len(self.audio):
            await self.protocol.send_transcript(self.result(final=True))

    def end_input(self, *, require_all: bool = True) -> None:
        assert not self.input_ended, "duplicate end-of-input request"
        expected = self.server.expected_audio
        if expected is not None and require_all and not self.server.allow_partial_input:
            assert bytes(self.server.received_audio) == expected, (
                f"end of input before all audio arrived ({len(self.server.received_audio)}/{len(expected)} bytes)"
            )
        self.input_ended = True
        self.server.input_ended.set()

    async def write(self) -> None:
        while True:
            reply = await self.replies.get()
            try:
                if reply.final:
                    await self.server.release_results.wait()
                if reply.data is None:
                    await self.ws.close(code=reply.close_code)
                    return
                if isinstance(reply.data, dict):
                    await self.ws.send_json(reply.data)
                elif isinstance(reply.data, bytes):
                    await self.ws.send_bytes(reply.data)
                else:
                    await self.ws.send_str(reply.data)
            finally:
                self.replies.task_done()


class STTServer:
    def __init__(self, provider: Provider, *, segments: tuple[Segment, ...] | None = None):
        self.provider = provider
        self.segments = segments
        self.expected_audio = None if segments is None else b"".join(s.audio for s in segments)
        self.connections: list[Connection] = []
        self.received_audio = bytearray()
        self.errors: list[BaseException] = []
        self.connected = asyncio.Event()
        self.connections_opened: asyncio.Queue[Connection] = asyncio.Queue()
        self.audio_received = asyncio.Event()
        self.input_ended = asyncio.Event()
        self.final_requested = asyncio.Event()
        self.release_results = asyncio.Event()
        self.release_results.set()
        self.answer_pings = True
        self.automatic_finals = False
        self.allow_partial_input = False
        self.fault_after_audio: str | None = None
        self.processing_stalled = False
        self.ignore_client_messages = False
        self.app = web.Application()
        self.app.router.add_route("*", "/{path:.*}", self.handle)
        self.http = TestServer(self.app)
        self.ssl_context = None
        self._certificates = None
        self._client_sessions: list[aiohttp.ClientSession] = []

    @property
    def url(self) -> str:
        return str(self.http.make_url("/"))[:-1]

    @property
    def ws_url(self) -> str:
        return self.url.replace("http", "ws", 1)

    def client_session(self) -> aiohttp.ClientSession:
        session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context))
        self._client_sessions.append(session)
        return session

    async def bootstrap(self, request: web.Request) -> web.Response:
        if self.provider.bootstrap is not None:
            return await self.provider.bootstrap(self, request)
        raise AssertionError(f"unexpected HTTP request: {request.method} {request.path}")

    async def handle(self, request: web.Request) -> web.StreamResponse:
        connection = None
        writer = None
        try:
            if request.headers.get("Upgrade", "").lower() != "websocket":
                return await self.bootstrap(request)
            ws = web.WebSocketResponse(autoping=False, autoclose=True, timeout=0.2)
            await ws.prepare(request)
            connection = Connection(self, request, ws)
            self.connections.append(connection)
            writer = asyncio.create_task(connection.write())

            def writer_done(task):
                if not task.cancelled() and (error := task.exception()) is not None:
                    if not isinstance(error, (ConnectionResetError, aiohttp.ClientConnectionError)):
                        self.errors.append(error)
                    connection.abort()

            writer.add_done_callback(writer_done)
            await connection.protocol.open(request)
            self.connected.set()
            self.connections_opened.put_nowait(connection)
            async for message in ws:
                connection.messages.append(message)
                if message.type == aiohttp.WSMsgType.PING:
                    connection.pings += 1
                    if self.answer_pings:
                        await ws.pong(message.data)
                elif message.type == aiohttp.WSMsgType.PONG:
                    connection.pongs += 1
                    connection.pong_received.set()
                elif message.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    if not self.ignore_client_messages:
                        await connection.protocol.receive(message)
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise ws.exception() or ConnectionError("server WebSocket receive failed")
            return ws
        except (ConnectionResetError, aiohttp.ClientConnectionError):
            # The client is allowed to abort, including during a final response.
            return connection.ws if connection is not None else web.Response(status=499)
        except Exception as error:
            self.errors.append(error)
            if connection is not None:
                await connection.ws.close(code=1002, message=b"test protocol violation")
                return connection.ws
            return web.Response(status=500)
        finally:
            try:
                if writer is not None:
                    writer.cancel()
                    await asyncio.gather(writer, return_exceptions=True)
            finally:
                if connection is not None:
                    connection.disconnected.set()

    async def __aenter__(self) -> STTServer:
        ssl_options = {}
        if self.provider.tls:
            from .tls import create_ssl_contexts

            self._certificates = tempfile.TemporaryDirectory()
            server_context, self.ssl_context = create_ssl_contexts(self._certificates.name)
            ssl_options["ssl"] = server_context
        await self.http.start_server(**ssl_options)
        return self

    async def __aexit__(self, *args: object) -> None:
        for session in self._client_sessions:
            await session.close()
        for connection in self.connections:
            await connection.ws.close()
        await self.http.close()
        if self._certificates is not None:
            self._certificates.cleanup()
        self.check()

    async def wait_for_audio(self, minimum_bytes: int) -> None:
        while len(self.received_audio) < minimum_bytes:
            self.audio_received.clear()
            await self.audio_received.wait()
        self.check()

    def check(self) -> None:
        if self.errors:
            raise AssertionError(f"server protocol failed: {self.errors!r}") from self.errors[0]


def json_message(message: aiohttp.WSMessage) -> dict[str, Any]:
    assert message.type == aiohttp.WSMsgType.TEXT, "expected a JSON text frame"
    value = json.loads(message.data)
    assert isinstance(value, dict), "expected a JSON object"
    return value
