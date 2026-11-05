"""Real MCP SDK peer for the runtime's HTTP and stdio lifecycle contract."""

import asyncio
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError

mode, event_dir = sys.argv[1:3]
events = Path(event_dir)
events.mkdir(parents=True, exist_ok=True)


async def list_tools(ctx, params):
    second = params is not None and params.cursor == "second"
    return types.ListToolsResult(
        tools=[types.Tool(name="second" if second else "first", inputSchema={"type": "object"})],
        nextCursor=None if second else "second",
    )


async def call_tool(ctx, params):
    if params.name == "wait":
        await anyio.Path(events / "called").write_text("entered")
        try:
            await anyio.sleep_forever()
        finally:
            await anyio.Path(events / "settled").write_text("settled")
    if params.name == "rpc_error":
        raise MCPError(-32602, "invalid tool arguments", data={"field": "seat"})
    return types.CallToolResult(content=[types.TextContent(type="text", text="hello")])


server = Server("mcp-lifecycle", version="1", on_list_tools=list_tools, on_call_tool=call_tool)


async def stdio():
    await anyio.Path(events / "pid").write_text(str(os.getpid()))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def http():
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route

    manager = StreamableHTTPSessionManager(server, json_response=True)

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    class Endpoint:
        async def __call__(self, scope, receive, send):
            # the credentials the client sent, for the bearer-token check
            for name, value in scope["headers"]:
                if name == b"authorization":
                    (events / "authorization").write_bytes(value)
            await manager.handle_request(scope, receive, send)

    app = Starlette(
        routes=[
            Route("/mcp", Endpoint(), methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    (events / "url").write_text(f"http://127.0.0.1:{sock.getsockname()[1]}/mcp")
    uvicorn.Server(uvicorn.Config(app, log_level="error")).run(sockets=[sock])


if mode == "stdio":
    asyncio.run(stdio())
else:
    http()
