from __future__ import annotations

import pytest

from livekit.agents.llm.mcp import MCPServer, MCPServerHTTP, MCPServerStdio, MCPTool, MCPToolset

pytestmark = pytest.mark.unit


class Server(MCPServer):
    def __init__(self) -> None:
        self.connected = False
        self.closes = 0
        self.fail_discovery = True

    @property
    def initialized(self) -> bool:
        return self.connected

    async def initialize(self) -> None:
        self.connected = True

    async def list_tools(self, **kwargs: object) -> list[MCPTool]:
        if self.fail_discovery:
            raise RuntimeError("discovery failed after connection")
        return []

    async def aclose(self) -> None:
        self.closes += 1
        self.connected = False


@pytest.mark.asyncio
async def test_failed_discovery_closes_the_acquired_mcp_session() -> None:
    server = Server()
    toolset = MCPToolset(id="mcp", mcp_server=server)
    with pytest.raises(RuntimeError, match="discovery failed"):
        await toolset.setup()
    assert server.closes == 1
    assert not server.initialized
    server.fail_discovery = False
    await toolset.setup()
    await toolset.aclose()
    assert server.closes == 2


@pytest.mark.asyncio
async def test_parent_close_failure_still_closes_the_mcp_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.llm.tool_context import AsyncToolset

    async def fail_close(self: AsyncToolset) -> None:
        raise RuntimeError("parent close failed")

    server = Server()
    server.fail_discovery = False
    toolset = MCPToolset(id="mcp", mcp_server=server)
    await toolset.setup()
    monkeypatch.setattr(AsyncToolset, "aclose", fail_close)
    with pytest.raises(RuntimeError, match="parent close failed"):
        await toolset.aclose()
    assert server.closes == 1
    assert not server.initialized


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 1e300])
@pytest.mark.parametrize("field", ["timeout", "sse_read_timeout", "client_session_timeout_seconds"])
def test_invalid_mcp_timeout_fails_at_construction(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        MCPServerHTTP("http://127.0.0.1:1/mcp", **{field: bad})
    if field == "client_session_timeout_seconds":
        with pytest.raises(ValueError, match=field):
            MCPServerStdio("never-launched", [], client_session_timeout_seconds=bad)
