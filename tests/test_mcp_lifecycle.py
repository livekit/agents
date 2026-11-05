from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import _core
from livekit.agents.llm import mcp
from livekit.agents.llm.mcp import MCPServer, MCPServerHTTP, MCPServerStdio, MCPToolset
from livekit.agents.llm.tool_context import RawFunctionTool, function_tool
from livekit.agents.voice.events import RunContext

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


class Holder:
    """A run or a call as ``RunContext`` reads it: its application context."""

    def __init__(self, token: str) -> None:
        self.context = type("Member", (), {"token": token})()


def run_ctx(token: str) -> RunContext[Any]:
    return RunContext(function_call=None, run=Holder(token))  # type: ignore[arg-type]


def call_ctx(holder: Holder) -> RunContext[Any]:
    return RunContext(function_call=None, session=holder)  # type: ignore[arg-type]


def per_user() -> MCPToolset:
    server = MCPServerHTTP("http://127.0.0.1:1/mcp", auth=lambda ctx: ctx.context.token)
    return MCPToolset(id="mcp", mcp_server=server)


# Catches: one user's MCP session, and the tools and data behind it, serving another user.
def test_per_run_credentials_share_a_toolset_only_between_equal_credentials() -> None:
    toolset = per_user()
    alice, again, bob = (toolset.for_run(run_ctx(t)) for t in ("alice", "alice", "bob"))
    assert alice is again
    assert alice is not bob and alice is not toolset


# Catches: a refreshed token mid-call swapping in a toolset whose tools were never set up.
def test_a_call_keeps_its_toolset_when_its_token_changes() -> None:
    toolset = per_user()
    call = Holder("first")
    kept = toolset.for_run(call_ctx(call))
    call.context.token = "refreshed"
    assert toolset.for_run(call_ctx(call)) is kept


@pytest.mark.asyncio
async def test_a_changed_token_reconnects_before_the_next_request() -> None:
    connects: list[str | None] = []

    class Session:
        connected = True

        async def connect_http(self, url: str, headers: object, auth: str | None) -> None:
            connects.append(auth)

    call = Holder("first")
    child = per_user().for_run(call_ctx(call))
    assert isinstance(child, MCPToolset)
    server = child._mcp_server
    assert isinstance(server, MCPServerHTTP)
    server._session = Session()  # type: ignore[assignment]
    await server.initialize()
    await server._refresh()
    call.context.token = "refreshed"
    await server._refresh()
    assert connects == ["first", "refreshed"]


def test_a_filtered_per_run_toolset_filters_each_run() -> None:
    toolset = per_user()
    view = toolset.filtered(lambda tool: True)
    child = view.for_run(run_ctx("alice"))
    assert child is not view
    assert child is view.for_run(run_ctx("alice"))
