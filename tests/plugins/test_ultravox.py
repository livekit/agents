"""Ultravox realtime plugin: the call protocol through a local peer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import http, llm
from livekit.plugins.ultravox.realtime import RealtimeModel
from tests.fakes import until

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


class _Peer:
    def __init__(self) -> None:
        self.calls: list[Wire] = []
        self.sockets: asyncio.Queue[web.WebSocketResponse] = asyncio.Queue()
        self.received: asyncio.Queue[Wire] = asyncio.Queue()
        self.join_url = ""

    async def create_call(self, request: web.Request) -> web.Response:
        self.calls.append(await request.json())
        return web.json_response({"joinUrl": self.join_url})

    async def join(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.sockets.put_nowait(socket)
        async for message in socket:
            if message.type == web.WSMsgType.TEXT:
                self.received.put_nowait(json.loads(message.data))
        return socket

    async def next_received(self, kind: str) -> Wire:
        while True:
            event = await asyncio.wait_for(self.received.get(), 1)
            if event["type"] == kind:
                return event


async def test_typed_tool_arguments_transcripts_and_results_cross_the_protocol() -> None:
    """Tool schemas and arguments keep their JSON types; transcripts and results are forwarded.

    https://github.com/livekit/agents/issues/3713 https://github.com/livekit/agents/pull/3339
    """

    @llm.function_tool
    async def find_items(tags: list[str], limit: int, include_value: bool) -> str:
        """Find stock items."""
        return ""

    peer = _Peer()
    app = web.Application()
    app.router.add_post("/api/calls", peer.create_call)
    app.router.add_get("/join", peer.join)
    async with TestServer(app) as server, http.open():
        peer.join_url = str(server.make_url("/join").with_scheme("ws"))
        model = RealtimeModel(api_key="key", base_url=str(server.make_url("/api")))
        session = model.session()
        transcripts: list[str] = []
        generations: list[llm.GenerationCreatedEvent] = []
        session.on(
            "input_audio_transcription_completed", lambda ev: transcripts.append(ev.transcript)
        )
        session.on("generation_created", generations.append)
        try:
            await asyncio.wait_for(peer.sockets.get(), 1)
            await session.update_tools([find_items])
            socket = await asyncio.wait_for(peer.sockets.get(), 1)

            [tool] = peer.calls[-1]["selectedTools"]
            schemas = {
                p["name"]: p["schema"]["type"] for p in tool["temporaryTool"]["dynamicParameters"]
            }
            assert schemas == {"tags": "array", "limit": "integer", "include_value": "boolean"}

            await socket.send_json(
                {
                    "type": "transcript",
                    "role": "user",
                    "medium": "voice",
                    "text": "low stock?",
                    "final": True,
                    "ordinal": 1,
                }
            )
            arguments = {"tags": ["low_stock"], "limit": 10, "include_value": True, "note": None}
            await socket.send_json(
                {
                    "type": "client_tool_invocation",
                    "toolName": "find_items",
                    "invocationId": "inv-1",
                    "parameters": arguments,
                }
            )
            await until(lambda: generations, "the tool call opened a generation")
            calls = [call async for call in generations[0].function_stream]
            assert [(c.call_id, c.name, json.loads(c.arguments)) for c in calls] == [
                ("inv-1", "find_items", arguments)
            ]
            assert transcripts == ["low stock?"]

            chat_ctx = llm.ChatContext.empty()
            chat_ctx.insert(calls[0])
            chat_ctx.insert(
                llm.FunctionCallOutput(
                    call_id="inv-1", name="find_items", output='{"count": 3}', is_error=False
                )
            )
            await session.update_chat_ctx(chat_ctx)
            result = await peer.next_received("client_tool_result")
            assert result["invocationId"] == "inv-1"
            assert result["result"] == '{"count": 3}'
            assert result["agentReaction"] == "speaks"
        finally:
            await session.aclose()
            await model.aclose()
