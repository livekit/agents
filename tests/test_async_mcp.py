from __future__ import annotations

from typing import Any, get_type_hints

import pytest

from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.llm.mcp import MCPServer, MCPToolOptions, MCPToolset
from livekit.agents.llm.tool_context import ToolFlag, is_raw_function_tool
from livekit.agents.llm.utils import is_context_type
from livekit.agents.voice.tool_executor import has_cancellable_tool

pytestmark = pytest.mark.unit


class _FakeToolDesc:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "echo back"
        self.inputSchema: dict[str, Any] = {"type": "object", "properties": {}}
        self.meta = None


class _FakeListToolsResult:
    def __init__(self, names: list[str]) -> None:
        self.tools = [_FakeToolDesc(n) for n in names]


class _FakeClient:
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.list_tools_calls = 0

    async def list_tools(self) -> _FakeListToolsResult:
        self.list_tools_calls += 1
        return _FakeListToolsResult(self._names)


class _FakeMCPServer(MCPServer):
    """Minimal MCPServer with a fake client — no real streams or network."""

    def __init__(self, tool_names: list[str] | None = None) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self._client = _FakeClient(tool_names or ["echo"])  # type: ignore[assignment]

    def client_streams(self):  # type: ignore[no-untyped-def]  # not used in these tests
        raise NotImplementedError


def _has_run_context_param(tool: object) -> bool:
    hints = get_type_hints(tool)
    return any(is_context_type(h, allow_subclasses=True) for h in hints.values())


def _build_mcp_tool(server: _FakeMCPServer, options: MCPToolOptions):
    return server._make_function_tool(
        name="echo",
        description="echo back",
        input_schema={"type": "object", "properties": {}},
        meta=None,
        options=options,
    )


def test_mcp_toolset_is_async_toolset_subclass() -> None:
    assert issubclass(MCPToolset, AsyncToolset)


def test_default_options_build_a_plain_blocking_tool() -> None:
    tool = _build_mcp_tool(_FakeMCPServer(), MCPToolOptions())

    assert is_raw_function_tool(tool)
    assert not _has_run_context_param(tool)
    assert tool.info.flags == ToolFlag.NONE
    assert tool.info.on_duplicate == "allow"


def test_forward_progress_adds_run_context_param() -> None:
    blocking = _build_mcp_tool(_FakeMCPServer(), MCPToolOptions(forward_progress=False))
    progress = _build_mcp_tool(_FakeMCPServer(), MCPToolOptions(forward_progress=True))

    assert not _has_run_context_param(blocking)
    assert _has_run_context_param(progress)


def test_flags_and_on_duplicate_pass_through_independently() -> None:
    # flags / on_duplicate are decoupled from forward_progress
    tool = _build_mcp_tool(
        _FakeMCPServer(),
        MCPToolOptions(
            flags=ToolFlag.CANCELLABLE | ToolFlag.IGNORE_ON_ENTER,
            on_duplicate="reject",
            forward_progress=False,
        ),
    )

    assert ToolFlag.CANCELLABLE in tool.info.flags
    assert ToolFlag.IGNORE_ON_ENTER in tool.info.flags
    assert tool.info.on_duplicate == "reject"
    # cancellable but still blocking — no ctx param
    assert not _has_run_context_param(tool)


def test_resolve_options_prefers_named_then_blocking_default() -> None:
    booking = MCPToolOptions(flags=ToolFlag.CANCELLABLE, forward_progress=True)
    ts = MCPToolset(id="m", mcp_server=_FakeMCPServer(), tool_options={"book_flight": booking})

    assert ts._resolve_options("book_flight") is booking
    # unlisted tools fall back to a plain blocking call
    assert ts._resolve_options("get_weather") == MCPToolOptions()


@pytest.mark.asyncio
async def test_setup_applies_per_tool_options() -> None:
    ts = MCPToolset(
        id="m",
        mcp_server=_FakeMCPServer(["get_weather", "book_flight"]),
        tool_options={
            "book_flight": MCPToolOptions(
                flags=ToolFlag.CANCELLABLE, on_duplicate="confirm", forward_progress=True
            ),
        },
    )
    await ts.setup()

    by_name = {t.info.name: t for t in ts._tools}
    assert ToolFlag.CANCELLABLE in by_name["book_flight"].info.flags
    assert by_name["book_flight"].info.on_duplicate == "confirm"
    assert _has_run_context_param(by_name["book_flight"])

    assert by_name["get_weather"].info.flags == ToolFlag.NONE
    assert not _has_run_context_param(by_name["get_weather"])

    assert has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_setup_default_is_all_blocking() -> None:
    ts = MCPToolset(id="m", mcp_server=_FakeMCPServer(["get_weather", "book_flight"]))
    await ts.setup()

    assert len(ts._tools) == 2
    assert not has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_list_tools_caches_raw_fetch() -> None:
    server = _FakeMCPServer(["echo"])

    await server.list_tools()
    await server.list_tools()
    assert server._client.list_tools_calls == 1  # raw descriptors cached

    server.invalidate_cache()
    await server.list_tools()
    assert server._client.list_tools_calls == 2  # re-fetched after invalidation


@pytest.mark.asyncio
async def test_list_tools_rebuilds_per_options() -> None:
    server = _FakeMCPServer(["echo"])

    blocking = await server.list_tools(resolve_options=lambda _name: MCPToolOptions())
    cancellable = await server.list_tools(
        resolve_options=lambda _name: MCPToolOptions(
            flags=ToolFlag.CANCELLABLE, forward_progress=True
        )
    )

    assert blocking[0] is not cancellable[0]
    assert ToolFlag.CANCELLABLE not in blocking[0].info.flags
    assert ToolFlag.CANCELLABLE in cancellable[0].info.flags
