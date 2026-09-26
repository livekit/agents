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
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import BaseModel

from livekit.agents.llm.mcp import (
    MCPElicitationContext,
    MCPElicitationHandler,
    MCPElicitationResult,
    MCPServer,
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


def test_warns_when_session_timeout_is_shorter(caplog: pytest.LogCaptureFixture) -> None:
    async def handler(ctx: MCPElicitationContext) -> MCPElicitationResult:
        return MCPElicitationResult(action="cancel")

    typed_handler: MCPElicitationHandler = handler
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        _InMemoryMCPServer(elicitation_handler=typed_handler, client_session_timeout_seconds=5)
    assert "elicitation_timeout should be lower" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        _InMemoryMCPServer(
            elicitation_handler=typed_handler,
            client_session_timeout_seconds=120,
            elicitation_timeout=60,
        )
        _InMemoryMCPServer(client_session_timeout_seconds=5)  # no handler, no warning
    assert "elicitation_timeout should be lower" not in caplog.text
