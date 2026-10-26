"""MCP servers as tool sources. The client sessions and their transports are the core's
(``_core.McpSession``); a server with its own ``client_streams()`` needs the ``mcp`` package."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from typing_extensions import Self

from .. import _core
from .._generated.mcp import MCPToolOptions
from ..voice.events import RunContext
from .async_toolset import AsyncToolset
from .tool_context import RawFunctionTool, function_tool

__all__ = [
    "MCPServer",
    "MCPServerHTTP",
    "MCPServerStdio",
    "MCPTool",
    "MCPToolOptions",
    "MCPToolResultContext",
    "MCPToolResultResolver",
    "MCPToolset",
]

MCPTool = RawFunctionTool


@dataclass
class MCPToolResultContext:
    """What an ``MCPToolResultResolver`` is called with."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    """The ``CallToolResult``, as its wire JSON."""


MCPToolResultResolver = Callable[[MCPToolResultContext], Any | Awaitable[Any]]


class _Progress:
    """A ``report_progress`` tool's progress messages, as ``ctx.update()`` calls."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def _core_update(self, message: str) -> None:
        await self._ctx.update(message)


class _Streams:
    """A ``client_streams()`` context as the core's transport (``mcp`` SDK streams)."""

    def __init__(self, cm: AbstractAsyncContextManager[Any]) -> None:
        self._cm = cm
        self._read: Any = None
        self._write: Any = None

    async def _core_start(self) -> None:
        streams = await self._cm.__aenter__()
        self._read, self._write = streams[0], streams[1]

    async def _core_send(self, message: dict[str, Any]) -> None:
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCMessage
        from pydantic import TypeAdapter

        parsed = TypeAdapter(JSONRPCMessage).validate_python(message)
        await self._write.send(SessionMessage(parsed))

    async def _core_receive(self) -> dict[str, Any] | None:
        import anyio

        try:
            message = await self._read.receive()
        except (anyio.EndOfStream, anyio.ClosedResourceError):
            return None
        if isinstance(message, Exception):
            raise message
        return message.message.model_dump(by_alias=True, exclude_none=True, mode="json")  # type: ignore[no-any-return]

    async def _core_close(self) -> None:
        await self._cm.__aexit__(None, None, None)


class MCPServer:
    """An MCP server; subclasses connect it (``initialize``) or give its ``client_streams()``."""

    def __init__(
        self,
        *,
        client_session_timeout_seconds: float,
        tool_result_resolver: MCPToolResultResolver | None = None,
    ) -> None:
        self._session = _core.McpSession(client_session_timeout_seconds)
        self._tool_result_resolver = tool_result_resolver

    @property
    def initialized(self) -> bool:
        return self._session.connected  # type: ignore[no-any-return]

    def invalidate_cache(self) -> None:
        self._session.invalidate_cache()

    async def initialize(self) -> None:
        await self._session.connect_host(_Streams(self.client_streams()))

    def client_streams(self) -> AbstractAsyncContextManager[Any]:
        raise NotImplementedError

    async def list_tools(
        self, *, tool_options: dict[str, MCPToolOptions] | None = None
    ) -> list[MCPTool]:
        options = tool_options or {}
        return [
            self._make_function_tool(
                tool["name"],
                tool.get("description"),
                tool.get("inputSchema") or {},
                tool.get("_meta"),
                options=_core.resolve_mcp_tool_options(options.get(tool["name"])),
            )
            for tool in await self._session.list_tools()
        ]

    def _make_function_tool(
        self,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
        meta: dict[str, Any] | None,
        *,
        options: dict[str, Any],
    ) -> MCPTool:
        async def _call(raw_arguments: dict[str, Any], progress: _Progress | None) -> Any:
            result = await self._session.call_tool(name, raw_arguments, progress)
            if self._tool_result_resolver is None:
                return _core.mcp_render_result(name, result)
            _core.mcp_check_result(result)
            resolved = self._tool_result_resolver(
                MCPToolResultContext(tool_name=name, arguments=raw_arguments, result=result)
            )
            return await resolved if asyncio.iscoroutine(resolved) else resolved

        impl: Callable[..., Awaitable[Any]]
        if options["report_progress"]:
            # progress becomes ctx.update(); the first update frees the reply loop
            async def impl(ctx: RunContext, raw_arguments: dict[str, Any]) -> Any:
                return await _call(raw_arguments, _Progress(ctx))
        else:

            async def impl(raw_arguments: dict[str, Any]) -> Any:
                return await _call(raw_arguments, None)

        return function_tool(
            impl,
            raw_schema=_core.mcp_raw_schema(name, description, input_schema, meta),
            flags=options["flags"],
            on_duplicate=options["on_duplicate"],
            duplicate_scope=options["duplicate_scope"],
        )

    async def aclose(self) -> None:
        await self._session.close()


class MCPServerHTTP(MCPServer):
    """An MCP server over HTTP: streamable HTTP, or SSE (deprecated).

    Args:
        url: The server's URL.
        transport_type: ``"sse"`` or ``"streamable_http"``; by default streamable HTTP when the
            URL path ends with ``/mcp``, else SSE.
        allowed_tools: The tool names exposed; all when ``None``.
        headers: HTTP headers sent with every request.
        timeout: The connect timeout, in seconds.
        sse_read_timeout: The read timeout, in seconds.
        client_session_timeout_seconds: The timeout of one request, in seconds.
        tool_result_resolver: Turns a tool's result into what the model reads.
    """

    def __init__(
        self,
        url: str,
        transport_type: Literal["sse", "streamable_http"] | None = None,
        allowed_tools: list[str] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        client_session_timeout_seconds: float = 5,
        *,
        tool_result_resolver: MCPToolResultResolver | None = None,
    ) -> None:
        super().__init__(
            client_session_timeout_seconds=client_session_timeout_seconds,
            tool_result_resolver=tool_result_resolver,
        )
        self.url = url
        self.headers = headers or {}
        self._timeout = timeout
        self._sse_read_timeout = sse_read_timeout
        self._transport = _core.mcp_http_transport(url, transport_type)
        self._allowed_tools = list(allowed_tools or [])
        self._session.allowed_tools = self._allowed_tools

    async def initialize(self) -> None:
        await self._session.connect_http(
            self.url,
            {k: str(v) for k, v in self.headers.items()},
            self._transport,
            self._timeout,
            self._sse_read_timeout,
        )

    def __repr__(self) -> str:
        allowed = f", allowed_tools={self._allowed_tools}" if self._allowed_tools else ""
        return f"MCPServerHTTP(url={self.url}, transport={self._transport}{allowed})"


class MCPServerStdio(MCPServer):
    """An MCP server run as a child process, over its stdio."""

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        client_session_timeout_seconds: float = 5,
        *,
        tool_result_resolver: MCPToolResultResolver | None = None,
    ) -> None:
        super().__init__(
            client_session_timeout_seconds=client_session_timeout_seconds,
            tool_result_resolver=tool_result_resolver,
        )
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

    async def initialize(self) -> None:
        cwd = str(self.cwd) if self.cwd is not None else None
        await self._session.connect_stdio(self.command, self.args, self.env or {}, cwd)

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"


class MCPToolset(AsyncToolset):
    """An ``MCPServer``'s tools for an ``Agent``; ``setup()`` connects and fetches them.

    Per-tool behavior is set with ``MCPToolOptions`` in ``tool_options`` (keyed by tool name);
    unlisted tools are plain blocking calls.

    Example::

        MCPToolset(
            id="mcp",
            mcp_server=MCPServerHTTP(url="...", client_session_timeout_seconds=120),
            tool_options={
                "book_flight": MCPToolOptions(
                    flags=ToolFlag.CANCELLABLE, on_duplicate="confirm", report_progress=True
                ),
            },
        )
    """

    def __init__(
        self,
        *,
        id: str,
        mcp_server: MCPServer,
        tool_options: dict[str, MCPToolOptions] | None = None,
    ) -> None:
        super().__init__(id=id)
        self._mcp_server = mcp_server
        self._tool_options = dict(tool_options or {})
        self._initialized = False
        self._lock = asyncio.Lock()

    async def setup(self, *, reload: bool = False) -> Self:
        """Connect the server when it is not, then fetch its tools; a no-op once done unless
        ``reload``."""
        await super().setup()
        async with self._lock:
            plan = _core.mcp_plan_setup(reload, self._initialized, self._mcp_server.initialized)
            if plan == "skip":
                return self
            if plan == "connect_then_list":
                await self._mcp_server.initialize()
            elif plan == "invalidate_then_list":
                self._mcp_server.invalidate_cache()
            self._tools = list(await self._mcp_server.list_tools(tool_options=self._tool_options))
            self._initialized = True
            return self

    def filter_tools(self, filter_fn: Callable[[MCPTool], bool]) -> Self:
        """Filter the toolset's tools in-place using a predicate."""
        self._tools = [t for t in self._tools if isinstance(t, MCPTool) and filter_fn(t)]
        return self

    async def aclose(self) -> None:
        try:
            await super().aclose()
            await self._mcp_server.aclose()
        finally:
            self._initialized = False
            self._tools = []
