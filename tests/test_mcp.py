from __future__ import annotations

import anyio
import asyncio
import pytest

pytest.importorskip("mcp")

import mcp.types
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, METHOD_NOT_FOUND, ErrorData

from livekit.agents.llm import mcp as lk_mcp
from livekit.agents.llm.tool_context import ToolError

pytestmark = pytest.mark.unit


def test_is_connection_dead():
    assert lk_mcp._is_connection_dead(anyio.ClosedResourceError())
    assert lk_mcp._is_connection_dead(anyio.BrokenResourceError())
    assert lk_mcp._is_connection_dead(
        McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))
    )
    # a normal protocol error is not a dead connection
    assert not lk_mcp._is_connection_dead(
        McpError(ErrorData(code=METHOD_NOT_FOUND, message="Method not found"))
    )
    assert not lk_mcp._is_connection_dead(ValueError("boom"))


class _DeadClient:
    """Fake ClientSession whose transport has died (server process gone)."""

    async def call_tool(self, name: str, arguments: dict) -> object:
        raise anyio.ClosedResourceError


class _ErrorResultClient:
    """Fake ClientSession that returns a normal tool error result."""

    async def call_tool(self, name: str, arguments: dict) -> mcp.types.CallToolResult:
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text="tool blew up")],
            isError=True,
        )


class _DeadListToolsClient:
    """Fake ClientSession whose transport has died while listing tools."""

    async def list_tools(self) -> object:
        raise anyio.ClosedResourceError


def _tool(server: lk_mcp.MCPServer):
    return server._make_function_tool("ping", None, {"type": "object", "properties": {}}, None)


async def test_dead_connection_fails_loudly_and_flips_initialized():
    # a session that connected and then died underneath us must not keep raising raw
    # anyio.ClosedResourceError forever: it should fail loudly and stop reporting as live.
    server = lk_mcp.MCPServerStdio(command="unused", args=[])
    server._client = _DeadClient()  # type: ignore[assignment]
    assert server.initialized is True

    with pytest.raises(ToolError):
        await _tool(server)(raw_arguments={})

    assert server.initialized is False


async def test_list_tools_dead_connection_tears_down_and_raises():
    server = lk_mcp.MCPServerStdio(command="unused", args=[])
    server._client = _DeadListToolsClient()  # type: ignore[assignment]
    server._cache_dirty = True

    with pytest.raises(RuntimeError, match="MCPServer connection failed"):
        await server.list_tools()

    assert server.initialized is False


async def test_initialize_waits_for_previous_dead_connection_cleanup():
    server = lk_mcp.MCPServerStdio(command="unused", args=[])

    task_started = asyncio.Event()
    task_continue = asyncio.Event()

    async def old_task() -> None:
        task_started.set()
        await task_continue.wait()

    server._client = None
    server._ready_fut = asyncio.get_running_loop().create_future()
    server._ready_fut.set_result(None)
    server._client_task = asyncio.create_task(old_task())

    await task_started.wait()

    async def fake_run_client(self, ready_fut: asyncio.Future[None]) -> None:
        ready_fut.set_result(None)

    server._run_client = fake_run_client.__get__(server, type(server))  # type: ignore[assignment]

    initialize_task = asyncio.create_task(server.initialize())
    await asyncio.sleep(0.01)
    assert not initialize_task.done()

    task_continue.set()
    await initialize_task

    assert server._client_task is not None
    assert server._client_task.done()


async def test_tool_error_result_still_propagates():
    # a normal isError result must still surface as a ToolError and must NOT tear the
    # connection down (i.e. it is not mistaken for a dead transport).
    server = lk_mcp.MCPServerStdio(command="unused", args=[])
    server._client = _ErrorResultClient()  # type: ignore[assignment]

    with pytest.raises(ToolError) as exc_info:
        await _tool(server)(raw_arguments={})

    assert "tool blew up" in str(exc_info.value)
    assert server.initialized is True
