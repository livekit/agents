from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import mcp.types
import pytest

from livekit.agents.llm.mcp import (
    MCPServerStdio,
    MCPToolResultContext,
    _default_tool_result_resolver,
)

pytestmark = pytest.mark.unit


def _resolve(result: mcp.types.CallToolResult) -> str | None:
    return _default_tool_result_resolver(
        MCPToolResultContext(tool_name="turn_on_light", arguments={}, result=result)
    )


def test_empty_content_with_structured_content_returns_it() -> None:
    result = mcp.types.CallToolResult(content=[], structured_content={"result": None})
    out = _resolve(result)
    assert out is not None
    assert json.loads(out) == {"result": None}


def test_empty_content_without_anything_returns_none() -> None:
    assert _resolve(mcp.types.CallToolResult(content=[])) is None


# A real stdio server: a tool annotated `-> None` comes back as content=[] with
# structured_content={"result": None}. The call succeeded, so it must not surface
# as a ToolError, which would prompt the model to run the side effect again.
_SERVER = textwrap.dedent(
    """
    from mcp.server.mcpserver import MCPServer

    app = MCPServer("lights")
    calls = 0

    @app.tool()
    def turn_on_light(room: str) -> None:
        global calls
        calls += 1

    @app.tool()
    def light_calls() -> int:
        return calls

    app.run(transport="stdio")
    """
)


async def test_tool_returning_none_over_stdio_is_not_an_error(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    script.write_text(_SERVER)

    server = MCPServerStdio(command=sys.executable, args=[str(script)])
    await server.initialize()
    try:
        tools = {t.info.name: t for t in await server.list_tools()}

        out = await tools["turn_on_light"](raw_arguments={"room": "kitchen"})
        assert out is not None
        assert json.loads(out) == {"result": None}

        count = await tools["light_calls"](raw_arguments={})
        assert json.loads(count)["text"] == "1"
    finally:
        await server.aclose()
