from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from livekit.agents.llm.mcp import (
    MCPServer,
    MCPServerHTTP,
    MCPServerStdio,
    MCPToolOptions,
    MCPToolset,
)
from livekit.agents.llm.tool_context import RawFunctionTool
from livekit.agents.voice.events import RunContext

pytestmark = pytest.mark.e2e


@pytest.fixture
async def server(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[MCPServer]:
    script = (
        Path(__file__).resolve().parents[1] / "rust/crates/lk-agents-runtime/fixtures/mcp_server.py"
    )
    mode = request.param
    if mode == "stdio":
        server: MCPServer = MCPServerStdio(sys.executable, [str(script), mode, str(tmp_path)])
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


@pytest.mark.parametrize("server", ["stdio", "http"], indirect=True)
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
        assert await tools[0]({}) == "hello", "a lone text block reaches the model as its text"
    finally:
        await toolset.aclose()
    assert server.initialized, "the connection stays open for the next run"
    await server.aclose()
    assert not server.initialized


@pytest.mark.parametrize("server", ["http"], indirect=True)
@pytest.mark.asyncio
async def test_a_per_run_bearer_token_reaches_the_server(server: MCPServer, tmp_path: Path) -> None:
    assert isinstance(server, MCPServerHTTP)
    per_run = MCPServerHTTP(server.url, auth=lambda ctx: ctx.context)
    ctx = RunContext[Any](function_call=None, run=type("Run", (), {"context": "secret"})())
    toolset = MCPToolset(id="mcp", mcp_server=per_run).for_run(ctx)
    try:
        await toolset.setup()
        assert [tool.info.name for tool in toolset.tools] == ["first", "second"]  # type: ignore[union-attr]
    finally:
        await toolset.aclose()
        assert isinstance(toolset, MCPToolset)
        await toolset._mcp_server.aclose()
    assert (tmp_path / "authorization").read_text() == "Bearer secret"
