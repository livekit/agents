"""MCP elicitation, tested end to end against an in-memory FastMCP server (no network)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from mcp import McpError
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import BaseModel

from livekit.agents.llm.mcp import (
    MCPElicitationContext,
    MCPElicitationResult,
    MCPServer,
    MCPServerHTTP,
    MCPTool,
)
from livekit.agents.llm.tool_context import ToolError

pytestmark = pytest.mark.unit


class _Seat(BaseModel):
    seat: str


def _make_fastmcp() -> FastMCP:
    server = FastMCP("elicitation-test")

    @server.tool()
    async def pick_seat(ctx: Context) -> str:  # type: ignore[type-arg]
        result = await ctx.elicit("Which seat would you like?", schema=_Seat)
        if result.action == "accept":
            return f"booked seat {result.data.seat}"
        return f"no seat: {result.action}"

    @server.tool()
    async def link_account(ctx: Context) -> str:  # type: ignore[type-arg]
        result = await ctx.elicit_url(
            "Sign in to link your account", url="https://example.com/oauth", elicitation_id="e1"
        )
        return f"link: {result.action}"

    @server.tool()
    async def client_supports_elicitation(ctx: Context) -> str:  # type: ignore[type-arg]
        params = ctx.session.client_params
        assert params is not None
        return str(params.capabilities.elicitation is not None)

    @server.tool()
    async def pick_two_seats(ctx: Context) -> str:  # type: ignore[type-arg]
        seats = []
        for leg in ("outbound", "return"):
            result = await ctx.elicit(f"Seat for the {leg} flight?", schema=_Seat)
            assert result.action == "accept"
            seats.append(result.data.seat)
        return f"booked seats {' '.join(seats)}"

    @server.tool()
    async def hang() -> str:
        await asyncio.sleep(30)
        return "done"

    @server.tool()
    def ping() -> str:
        return "pong"

    return server


class _InMemoryMCPServer(MCPServer):
    """MCPServer connected to a FastMCP server over in-memory streams."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("client_session_timeout_seconds", 10)
        super().__init__(**kwargs)
        self._server = _make_fastmcp()._mcp_server

    @asynccontextmanager
    async def client_streams(self) -> AsyncIterator[Any]:  # type: ignore[override]
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    lambda: self._server.run(
                        server_streams[0],
                        server_streams[1],
                        self._server.create_initialization_options(),
                    )
                )
                yield client_streams
                tg.cancel_scope.cancel()


@asynccontextmanager
async def _connected(**kwargs: Any) -> AsyncIterator[dict[str, MCPTool]]:
    server = _InMemoryMCPServer(**kwargs)
    await server.initialize()
    try:
        yield {tool.info.name: tool for tool in await server.list_tools()}
    finally:
        await server.aclose()


async def _call(tool: MCPTool) -> str:
    # the default resolver returns the single content item as JSON
    return str(json.loads(await tool({}))["text"])


async def test_form_elicitation_accept() -> None:
    received: list[MCPElicitationContext] = []

    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        received.append(ctx)
        return MCPElicitationResult(action="accept", content={"seat": "12A"})

    async with _connected(elicitation_handler=handler) as tools:
        assert await _call(tools["pick_seat"]) == "booked seat 12A"

    assert len(received) == 1
    ctx = received[0]
    assert ctx.mode == "form"
    assert ctx.message == "Which seat would you like?"
    assert ctx.url is None
    assert ctx.requested_schema is not None
    assert ctx.requested_schema["properties"]["seat"]["type"] == "string"
    assert isinstance(ctx.server, _InMemoryMCPServer)


@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_form_elicitation_decline_or_cancel(action: str) -> None:
    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        return MCPElicitationResult(action=action)  # type: ignore[arg-type]

    async with _connected(elicitation_handler=handler) as tools:
        assert await _call(tools["pick_seat"]) == f"no seat: {action}"


async def test_url_elicitation() -> None:
    received: list[MCPElicitationContext] = []

    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        received.append(ctx)
        return MCPElicitationResult(action="accept")

    async with _connected(elicitation_handler=handler) as tools:
        assert await _call(tools["link_account"]) == "link: accept"

    ctx = received[0]
    assert ctx.mode == "url"
    assert ctx.url == "https://example.com/oauth"
    assert ctx.requested_schema is None


async def test_handler_timeout_answers_cancel(caplog: pytest.LogCaptureFixture) -> None:
    async def slow_handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        await asyncio.sleep(5)
        return MCPElicitationResult(action="accept", content={"seat": "1A"})

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        async with _connected(elicitation_handler=slow_handler, elicitation_timeout=0.1) as tools:
            assert await _call(tools["pick_seat"]) == "no seat: cancel"
            # the session keeps working after a timed out elicitation
            assert await _call(tools["ping"]) == "pong"

    assert "MCP elicitation timed out" in caplog.text


async def test_handler_error_fails_the_tool_call(caplog: pytest.LogCaptureFixture) -> None:
    async def broken_handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        raise RuntimeError("frontend unreachable")

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        async with _connected(elicitation_handler=broken_handler) as tools:
            # the server receives an INTERNAL_ERROR, so the tool call reports an error
            with pytest.raises(ToolError, match="Elicitation handler failed"):
                await tools["pick_seat"]({})
            assert await _call(tools["ping"]) == "pong"

    assert "MCP elicitation handler failed" in caplog.text


async def test_capability_advertised_only_with_handler() -> None:
    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        return MCPElicitationResult(action="cancel")

    async with _connected() as tools:
        assert await _call(tools["client_supports_elicitation"]) == "False"
        # without a handler the SDK's default callback rejects the request
        with pytest.raises(ToolError, match="Elicitation not supported"):
            await tools["pick_seat"]({})

    async with _connected(elicitation_handler=handler) as tools:
        assert await _call(tools["client_supports_elicitation"]) == "True"


async def test_slow_answer_outlives_session_read_timeout() -> None:
    # the user answers after the session read timeout; the tool call must still succeed
    async def slow_handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        await asyncio.sleep(0.6)
        return MCPElicitationResult(action="accept", content={"seat": "3C"})

    async with _connected(
        elicitation_handler=slow_handler,
        client_session_timeout_seconds=0.3,
        elicitation_timeout=2,
    ) as tools:
        assert await _call(tools["pick_seat"]) == "booked seat 3C"


async def test_elicitation_time_not_charged_to_tool_call() -> None:
    # two answers, each slower than the session read timeout but within
    # elicitation_timeout: time spent waiting on the user doesn't count
    seats = iter(["12A", "14C"])

    async def slow_handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        await asyncio.sleep(1.0)
        return MCPElicitationResult(action="accept", content={"seat": next(seats)})

    # the call takes ~2s in total, longer than read timeout + one elicitation_timeout
    async with _connected(
        elicitation_handler=slow_handler,
        client_session_timeout_seconds=0.3,
        elicitation_timeout=1.5,
    ) as tools:
        assert await _call(tools["pick_two_seats"]) == "booked seats 12A 14C"


async def test_hung_tool_still_times_out_with_handler() -> None:
    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        return MCPElicitationResult(action="cancel")

    async with _connected(elicitation_handler=handler, client_session_timeout_seconds=0.3) as tools:
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(McpError, match="Timed out while waiting for response to tool 'hang'"):
            await tools["hang"]({})
        assert loop.time() - start < 2
        # the session keeps working after the timeout
        assert await _call(tools["ping"]) == "pong"


async def test_handler_raised_timeout_is_an_error(caplog: pytest.LogCaptureFixture) -> None:
    # a TimeoutError from inside the handler (e.g. a timed out RPC) is a failure, not "cancel"
    async def rpc_timeout_handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        raise asyncio.TimeoutError()

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        async with _connected(elicitation_handler=rpc_timeout_handler) as tools:
            with pytest.raises(ToolError, match="Elicitation handler failed"):
                await tools["pick_seat"]({})

    assert "MCP elicitation handler failed" in caplog.text


def test_http_warns_when_elicitation_outlives_sse_read_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        return MCPElicitationResult(action="cancel")

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        MCPServerHTTP("http://localhost/mcp", elicitation_handler=handler)  # 60s < 300s
        MCPServerHTTP("http://localhost/mcp", elicitation_timeout=None)  # no handler
    assert "sse_read_timeout" not in caplog.text

    for kwargs in ({"elicitation_timeout": None}, {"sse_read_timeout": 30}):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            MCPServerHTTP("http://localhost/mcp", elicitation_handler=handler, **kwargs)  # type: ignore[arg-type]
        assert "elicitation_timeout should be lower than sse_read_timeout" in caplog.text
