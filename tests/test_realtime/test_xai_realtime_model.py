"""Provider declarations cross the native realtime boundary without losing validation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai.types.responses.file_search_tool import RankingOptions
from openai.types.responses.web_search_tool import Filters, UserLocation
from openai.types.shared import ComparisonFilter

from livekit.agents import Agent, APIConnectOptions, VoiceSession, llm
from livekit.agents.voice.events import ErrorEvent
from livekit.plugins.google.tools import GoogleSearch
from livekit.plugins.openai import realtime as openai_realtime
from livekit.plugins.openai.tools import (
    FileSearch as OpenAIFileSearch,
    WebSearch as OpenAIWebSearch,
)
from livekit.plugins.xai import realtime
from livekit.plugins.xai.tools import FileSearch, WebSearch, XAITool, XSearch

pytestmark = pytest.mark.unit


@pytest.fixture
async def peer() -> AsyncIterator[tuple[str, asyncio.Queue[list[dict[str, object]]]]]:
    received: asyncio.Queue[list[dict[str, object]]] = asyncio.Queue()

    async def connect(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {"id": "test"}})
        async for message in ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                payload = message.json()
                if payload["type"] == "session.update" and "tools" in payload["session"]:
                    await received.put(payload["session"]["tools"])
        return ws

    app = web.Application()
    app.router.add_get("/{tail:.*}", connect)
    async with TestServer(app) as server:
        yield str(server.make_url("/realtime").with_scheme("ws")), received


@pytest.mark.parametrize(
    "provider_tool",
    [
        WebSearch(),
        XSearch(allowed_x_handles=["livekit"]),
        FileSearch(vector_store_ids=["store"], max_num_results=3),
    ],
)
async def test_provider_tools_and_nested_functions_reach_the_native_realtime_peer(
    provider_tool: XAITool,
    peer: tuple[str, asyncio.Queue[list[dict[str, object]]]],
) -> None:
    url, received = peer

    @llm.function_tool
    async def lookup() -> str:
        """Look up a booking."""
        return "found"

    model = realtime.RealtimeModel(
        api_key="test", base_url=url, conn_options=APIConnectOptions(max_retry=0)
    )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(
            Agent(
                instructions="Help with bookings.",
                tools=[provider_tool, llm.Toolset(id="nested", tools=[lookup])],
            )
        )
        async with asyncio.timeout(2):
            tools = await received.get()
        assert provider_tool.to_dict() in tools
        assert [tool["name"] for tool in tools if tool["type"] == "function"] == ["lookup"]


@pytest.mark.parametrize("kind", ["function", "raw", "nested"])
async def test_reserved_names_emit_an_error_without_sending_the_invalid_tools(
    kind: str,
    peer: tuple[str, asyncio.Queue[list[dict[str, object]]]],
) -> None:
    url, received = peer
    errors: asyncio.Queue[ErrorEvent] = asyncio.Queue()

    async def search() -> str:
        return "found"

    if kind == "raw":
        tool: llm.Tool = llm.function_tool(
            raw_schema={
                "name": "collections_search",
                "description": "Search records.",
                "parameters": {"type": "object", "properties": {}},
            }
        )(search)
    else:
        tool = llm.function_tool(name="collections_search")(search)
        if kind == "nested":
            tool = llm.Toolset(id="nested", tools=[tool])
    model = realtime.RealtimeModel(
        api_key="test", base_url=url, conn_options=APIConnectOptions(max_retry=0)
    )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        session.on("error", errors.put_nowait)
        await session.start(Agent(instructions="Search records.", tools=[FileSearch(), tool]))
        async with asyncio.timeout(2):
            event = await errors.get()
        assert isinstance(event.error, llm.RealtimeModelError)
        assert not event.error.recoverable
        assert "Rename or remove" in str(event.error.error)
        assert received.empty()


@pytest.mark.parametrize(
    "kind,provider_tool",
    [
        ("xai", OpenAIWebSearch()),
        ("azure", OpenAIWebSearch()),
        (
            "xai",
            OpenAIWebSearch(
                user_location=UserLocation(type="approximate", city="Paris"),
                filters=Filters(allowed_domains=["example.com"]),
            ),
        ),
        (
            "xai",
            OpenAIFileSearch(
                vector_store_ids=["store"],
                filters=ComparisonFilter(type="eq", key="country", value="FR"),
                ranking_options=RankingOptions(score_threshold=0.4),
            ),
        ),
        ("xai", GoogleSearch()),
        ("openai", WebSearch()),
        ("openai", OpenAIWebSearch()),
        ("openai", GoogleSearch()),
    ],
)
async def test_realtime_ignores_foreign_and_responses_only_provider_tools(
    kind: str,
    provider_tool: llm.ProviderTool,
    peer: tuple[str, asyncio.Queue[list[dict[str, object]]]],
) -> None:
    url, received = peer

    @llm.function_tool
    async def lookup() -> str:
        return "found"

    if kind == "azure":
        model = openai_realtime.RealtimeModel.with_azure(
            api_key="test", base_url=url, azure_deployment="test"
        )
    else:
        constructor = realtime.RealtimeModel if kind == "xai" else openai_realtime.RealtimeModel
        model = constructor(
            api_key="test", base_url=url, conn_options=APIConnectOptions(max_retry=0)
        )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Help.", tools=[provider_tool, lookup]))
        async with asyncio.timeout(2):
            tools = await received.get()
        assert [tool["type"] for tool in tools] == ["function"]
        assert tools[0]["name"] == "lookup"


async def test_realtime_refuses_per_reply_tools_without_mutating_its_toolset(
    peer: tuple[str, asyncio.Queue[list[dict[str, object]]]],
) -> None:
    url, received = peer
    provider = WebSearch()

    @llm.function_tool
    async def lookup() -> str:
        return "found"

    model = realtime.RealtimeModel(
        api_key="test", base_url=url, conn_options=APIConnectOptions(max_retry=0)
    )
    async with VoiceSession(llm=model, vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Help.", tools=[provider, lookup]))
        async with asyncio.timeout(2):
            await received.get()
            handle = session.generate_reply(tools=[])
            await handle
        assert handle.exception() is not None
        assert "not supported on a realtime model" in str(handle.exception())
        assert received.empty()
