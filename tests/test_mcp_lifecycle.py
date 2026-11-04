from __future__ import annotations

import asyncio

import pytest

from livekit.agents import _core
from livekit.agents.llm import mcp
from livekit.agents.llm.mcp import MCPServer, MCPServerHTTP, MCPServerStdio, MCPToolset
from livekit.agents.llm.tool_context import RawFunctionTool, function_tool

pytestmark = pytest.mark.unit


@function_tool
async def lookup() -> str:
    return "found"


class Server(MCPServer):
    def __init__(self) -> None:
        self.connected = False
        self.closes = 0
        self.reachable = True
        self.listing: asyncio.Event = asyncio.Event()

    @property
    def initialized(self) -> bool:
        return self.connected

    @property
    def _retry_in(self) -> float:
        return 0.0

    async def initialize(self) -> None:
        self.connected = True

    async def list_tools(self, **kwargs: object) -> list[RawFunctionTool]:
        self.listing.set()
        if not self.reachable:
            raise ConnectionError("server unreachable")
        return [lookup]  # type: ignore[list-item]

    async def aclose(self) -> None:
        self.closes += 1
        self.connected = False


# Catches: one unreachable MCP server failing the activation, which on a call ends the call.
@pytest.mark.asyncio
async def test_an_unreachable_server_leaves_the_toolset_empty_until_it_answers() -> None:
    server = Server()
    server.reachable = False
    toolset = MCPToolset(id="mcp", mcp_server=server)
    await toolset.setup()
    assert list(toolset.tools) == []
    server.listing.clear()
    server.reachable = True
    async with asyncio.timeout(5):
        while not toolset.tools:
            await asyncio.sleep(0)
    assert list(toolset.tools) == [lookup]
    await toolset.aclose()


@pytest.mark.asyncio
async def test_a_cancelled_setup_is_not_swallowed() -> None:
    class Blocked(Server):
        async def list_tools(self, **kwargs: object) -> list[RawFunctionTool]:
            self.listing.set()
            await asyncio.Event().wait()
            raise AssertionError

    server = Blocked()
    setup = asyncio.create_task(MCPToolset(id="mcp", mcp_server=server).setup())
    await server.listing.wait()
    setup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await setup


# Catches: closing the connection after every run, so each backend request pays a handshake.
@pytest.mark.asyncio
async def test_the_connection_outlives_the_last_run_until_it_idles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp, "MCP_IDLE_CLOSE_SECONDS", 0.05)
    server = Server()
    toolset = MCPToolset(id="mcp", mcp_server=server)
    await toolset.setup()
    await toolset.aclose()
    await toolset.setup()
    await asyncio.sleep(0.1)
    assert server.closes == 0, "a run inside the idle window reuses the connection"
    await toolset.aclose()
    await asyncio.sleep(0.1)
    assert server.closes == 1
    assert list(toolset.tools) == []


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 1e300])
@pytest.mark.parametrize("field", ["init_timeout", "read_timeout", "request_timeout"])
def test_invalid_mcp_timeout_fails_at_construction(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        MCPServerHTTP("http://127.0.0.1:1/mcp", **{field: bad})  # type: ignore[arg-type]
    if field != "read_timeout":
        with pytest.raises(ValueError, match=field):
            MCPServerStdio("never-launched", [], **{field: bad})  # type: ignore[arg-type]


def test_an_invalid_tool_timeout_fails_when_its_options_resolve() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _core.resolve_mcp_tool_options(mcp.MCPToolOptions(timeout=-1.0))
