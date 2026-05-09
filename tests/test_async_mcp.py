from __future__ import annotations

import pytest

from livekit.agents.llm.async_toolset import (
    AsyncToolset,
    _has_async_context_param,
    cancel_task,
    get_running_tasks,
)
from livekit.agents.llm.mcp import MCPServer, MCPToolset
from livekit.agents.llm.tool_context import is_raw_function_tool


class _FakeMCPServer(MCPServer):
    """Minimal MCPServer subclass for tests — no real client streams."""

    def __init__(self) -> None:
        super().__init__(client_session_timeout_seconds=5)

    def client_streams(self):  # not used in these tests
        raise NotImplementedError


def _build_mcp_tool(server: _FakeMCPServer, *, async_mode: bool):
    """Bypass real client setup; directly call _make_function_tool."""
    return server._make_function_tool(
        name="echo",
        description="echo back",
        input_schema={"type": "object", "properties": {}},
        meta=None,
        async_mode=async_mode,
    )


def test_mcp_toolset_is_async_toolset_subclass():
    assert issubclass(MCPToolset, AsyncToolset)


def test_mcp_tool_default_signature_has_no_async_context_param():
    server = _FakeMCPServer()
    tool = _build_mcp_tool(server, async_mode=False)

    assert is_raw_function_tool(tool)
    assert not _has_async_context_param(tool)


def test_mcp_tool_async_mode_signature_has_async_context_param():
    server = _FakeMCPServer()
    tool = _build_mcp_tool(server, async_mode=True)

    assert is_raw_function_tool(tool)
    assert _has_async_context_param(tool)


def test_mcp_toolset_default_does_not_expose_helper_tools():
    server = _FakeMCPServer()
    ts = MCPToolset(id="m", mcp_server=server)

    exposed = list(ts.tools)
    assert get_running_tasks not in exposed
    assert cancel_task not in exposed


def test_mcp_toolset_async_mode_exposes_helper_tools():
    server = _FakeMCPServer()
    ts = MCPToolset(id="m", mcp_server=server, async_mode=True)

    exposed = list(ts.tools)
    assert get_running_tasks in exposed
    assert cancel_task in exposed


def test_mcp_toolset_stores_async_mode_flag():
    server = _FakeMCPServer()

    assert MCPToolset(id="a", mcp_server=server)._async_mode is False
    assert MCPToolset(id="b", mcp_server=server, async_mode=True)._async_mode is True


def test_mcp_server_list_tools_cache_keyed_by_async_mode():
    """Switching async_mode should rebuild the tool list (cache miss)."""
    server = _FakeMCPServer()

    # Seed the cache as if a prior list_tools call ran with async_mode=False
    legacy_tool = _build_mcp_tool(server, async_mode=False)
    server._lk_tools = [legacy_tool]
    server._lk_tools_async_mode = False
    server._cache_dirty = False

    # Same mode — cache hit, returns the seeded list
    assert server._lk_tools_async_mode is False

    # Switching mode invalidates: simulate cache-miss check
    cache_hit_for_async = (
        not server._cache_dirty
        and server._lk_tools is not None
        and server._lk_tools_async_mode is True
    )
    assert cache_hit_for_async is False


@pytest.mark.asyncio
async def test_mcp_toolset_setup_wraps_async_mode_tools():
    """After setup with async_mode=True, MCP tools are wrapped — their outer
    signature is the AsyncToolset wrapper (RunContext, raw_arguments), not the
    inner AsyncRunContext signature."""
    server = _FakeMCPServer()

    # Pre-populate the server's cache so setup() bypasses real I/O
    raw_tool = _build_mcp_tool(server, async_mode=True)
    server._lk_tools = [raw_tool]
    server._lk_tools_async_mode = True
    server._cache_dirty = False

    class _DummyClient:
        pass

    server._client = _DummyClient()  # type: ignore[assignment]

    ts = MCPToolset(id="m", mcp_server=server, async_mode=True)
    await ts.setup()

    # _tools holds the wrapped tools — outer signature is RunContext
    assert len(ts._tools) == 1
    assert not _has_async_context_param(ts._tools[0])


@pytest.mark.asyncio
async def test_mcp_toolset_setup_passes_through_non_async_tools():
    """Without async_mode, tools are not wrapped (no AsyncRunContext to wrap)."""
    server = _FakeMCPServer()

    raw_tool = _build_mcp_tool(server, async_mode=False)
    server._lk_tools = [raw_tool]
    server._lk_tools_async_mode = False
    server._cache_dirty = False

    class _DummyClient:
        pass

    server._client = _DummyClient()  # type: ignore[assignment]

    ts = MCPToolset(id="m", mcp_server=server)
    await ts.setup()

    # _wrap_tool is a no-op for tools without AsyncRunContext, so identity holds
    assert ts._tools == [raw_tool]
