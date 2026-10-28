from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from livekit.agents.llm.mcp import (
    MCPServer,
    MCPServerHTTP,
    MCPServerStdio,
    MCPToolOptions,
    MCPToolset,
)
from livekit.agents.llm.tool_context import RawFunctionTool

pytestmark = pytest.mark.e2e


@pytest.fixture
async def server(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[MCPServer]:
    script = (
        Path(__file__).resolve().parents[1] / "rust/crates/lk-agents-runtime/tests/mcp/server.py"
    )
    mode = request.param
    if mode == "stdio":
        server = MCPServerStdio(sys.executable, [str(script), mode, str(tmp_path)])
        try:
            yield server
        finally:
            await server.aclose()
        return

    process = await asyncio.create_subprocess_exec(sys.executable, str(script), mode, str(tmp_path))
    try:
        async with asyncio.timeout(10):
            while not (tmp_path / "url").exists():
                if process.returncode is not None:
                    raise RuntimeError(f"MCP peer exited with {process.returncode}")
                await asyncio.sleep(0.01)
        server = MCPServerHTTP((tmp_path / "url").read_text())
        try:
            yield server
        finally:
            await server.aclose()
    finally:
        if process.returncode is None:
            process.terminate()
        await process.wait()


@pytest.mark.parametrize("server", ["stdio", "http", "sse"], indirect=True)
@pytest.mark.asyncio
async def test_mcp_toolset_discovers_and_calls_real_server(server: MCPServer) -> None:
    toolset = MCPToolset(
        id="mcp",
        mcp_server=server,
        tool_options={"first": MCPToolOptions(on_duplicate="reject")},
    )
    try:
        await toolset.setup()
        tools = [tool for tool in toolset.tools if isinstance(tool, RawFunctionTool)]
        assert [tool.info.name for tool in tools] == ["first", "second"]
        assert tools[0].info.on_duplicate == "reject"
        assert await tools[0]({}) == '{"type":"text","text":"hello"}'
    finally:
        await toolset.aclose()
    assert not server.initialized
