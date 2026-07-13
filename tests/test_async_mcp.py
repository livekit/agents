from __future__ import annotations

from typing import Any, get_type_hints

import pytest

from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.llm.mcp import MCPServer, MCPToolset
from livekit.agents.llm.tool_context import DuplicateMode, ToolFlag, is_raw_function_tool
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

    async def list_tools(self) -> _FakeListToolsResult:
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


def _build_mcp_tool(
    server: _FakeMCPServer, *, nonblocking: bool, on_duplicate: DuplicateMode = "confirm"
):
    return server._make_function_tool(
        name="echo",
        description="echo back",
        input_schema={"type": "object", "properties": {}},
        meta=None,
        nonblocking=nonblocking,
        on_duplicate=on_duplicate,
    )


def test_mcp_toolset_is_async_toolset_subclass() -> None:
    assert issubclass(MCPToolset, AsyncToolset)


def test_blocking_tool_has_no_run_context_and_is_not_cancellable() -> None:
    tool = _build_mcp_tool(_FakeMCPServer(), nonblocking=False)

    assert is_raw_function_tool(tool)
    assert not _has_run_context_param(tool)
    assert ToolFlag.CANCELLABLE not in tool.info.flags
    # blocking tools always fall back to "allow" — no duplicate management
    assert tool.info.on_duplicate == "allow"


def test_nonblocking_tool_has_run_context_and_is_cancellable() -> None:
    tool = _build_mcp_tool(_FakeMCPServer(), nonblocking=True, on_duplicate="reject")

    assert is_raw_function_tool(tool)
    assert _has_run_context_param(tool)
    assert ToolFlag.CANCELLABLE in tool.info.flags
    assert tool.info.on_duplicate == "reject"


def test_mcp_toolset_stores_nonblocking_tools_flag() -> None:
    server = _FakeMCPServer()

    assert MCPToolset(id="a", mcp_server=server)._nonblocking_tools is True
    assert (
        MCPToolset(id="b", mcp_server=server, nonblocking_tools=False)._nonblocking_tools is False
    )
    assert MCPToolset(
        id="c", mcp_server=server, nonblocking_tools=["book_flight", "search"]
    )._nonblocking_tools == frozenset({"book_flight", "search"})


@pytest.mark.asyncio
async def test_setup_default_exposes_cancellable_tools() -> None:
    ts = MCPToolset(id="m", mcp_server=_FakeMCPServer(["echo"]))  # nonblocking_tools=True
    await ts.setup()

    assert len(ts._tools) == 1
    assert _has_run_context_param(ts._tools[0])
    assert has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_setup_blocking_has_no_cancellable_tools() -> None:
    ts = MCPToolset(id="m", mcp_server=_FakeMCPServer(["echo"]), nonblocking_tools=False)
    await ts.setup()

    assert len(ts._tools) == 1
    assert not _has_run_context_param(ts._tools[0])
    assert not has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_setup_empty_list_is_all_blocking() -> None:
    ts = MCPToolset(id="m", mcp_server=_FakeMCPServer(["echo"]), nonblocking_tools=[])
    await ts.setup()

    assert not has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_setup_named_list_selects_nonblocking_tools() -> None:
    ts = MCPToolset(
        id="m",
        mcp_server=_FakeMCPServer(["get_weather", "book_flight"]),
        nonblocking_tools=["book_flight"],
    )
    await ts.setup()

    by_name = {t.info.name: t for t in ts._tools}
    assert ToolFlag.CANCELLABLE in by_name["book_flight"].info.flags
    assert _has_run_context_param(by_name["book_flight"])
    assert ToolFlag.CANCELLABLE not in by_name["get_weather"].info.flags
    assert not _has_run_context_param(by_name["get_weather"])
    assert has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_list_tools_cache_keyed_by_nonblocking_tools() -> None:
    server = _FakeMCPServer(["echo"])

    blocking = await server.list_tools(nonblocking_tools=False)
    # same key → cache hit → same list object
    assert (await server.list_tools(nonblocking_tools=False)) is blocking

    # different key → cache miss → rebuilt with different behavior
    nonblocking = await server.list_tools(nonblocking_tools=True)
    assert nonblocking is not blocking
    assert ToolFlag.CANCELLABLE in nonblocking[0].info.flags
    assert ToolFlag.CANCELLABLE not in blocking[0].info.flags


@pytest.mark.asyncio
async def test_list_tools_cache_keyed_by_on_duplicate() -> None:
    server = _FakeMCPServer(["echo"])

    confirm = await server.list_tools(nonblocking_tools=True, on_duplicate="confirm")
    reject = await server.list_tools(nonblocking_tools=True, on_duplicate="reject")

    assert reject is not confirm
    assert reject[0].info.on_duplicate == "reject"
