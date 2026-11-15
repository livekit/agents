"""Type checks for the public MCP API, run by ``.github/check_types.py``.

Each ``type: ignore[code]`` marks an error the type checker must report; strict mode reports an
ignore that is no longer needed, so the file fails both when a check is lost and when a new
error appears.
"""

from dataclasses import dataclass
from typing import assert_type

from pydantic import JsonValue

from livekit.agents import RunContext
from livekit.agents.llm import Toolset
from livekit.agents.llm.mcp import (
    MCPServerHTTP,
    MCPServerStdio,
    MCPTool,
    MCPToolOptions,
    MCPToolset,
)


@dataclass
class Caller:
    token: str
    tenant: str


def token(ctx: RunContext[Caller]) -> str:
    return ctx.context.token


def headers(ctx: RunContext[Caller]) -> dict[str, str]:
    return {"x-tenant": ctx.context.tenant}


async def listings(http: MCPServerHTTP[Caller], stdio: MCPServerStdio) -> None:
    for server in (http, stdio):
        tools = await server.list_tools()
        assert_type(tools, list[MCPTool])
        tool = tools[0]
        assert_type(tool.name, str)
        assert_type(tool.description, str | None)
        assert_type(tool.input_schema, dict[str, JsonValue])
        assert_type(tool.meta, dict[str, JsonValue] | None)


def per_run_credentials(ctx: RunContext[Caller]) -> None:
    server = MCPServerHTTP("https://mcp.example.com/mcp", auth=token, headers=headers)
    assert_type(server, MCPServerHTTP[Caller])
    toolset = MCPToolset(
        id="mcp",
        mcp_server=server,
        tool_options={"book": MCPToolOptions(timeout=120, report_progress=True)},
    )
    assert_type(toolset, MCPToolset[Caller])
    assert_type(toolset.for_run(ctx), Toolset)

    MCPServerHTTP("https://mcp.example.com/mcp", auth=42)  # type: ignore[arg-type]
    MCPToolOptions(timeout="soon")  # type: ignore[typeddict-item]
