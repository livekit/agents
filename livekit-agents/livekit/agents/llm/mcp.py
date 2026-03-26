# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import Self

from .tool_context import Toolset

try:
    import httpx
    import mcp.types
    from mcp import ClientSession, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.streamable_http import GetSessionIdCallback, streamable_http_client
    from mcp.shared.message import SessionMessage
except ImportError as e:
    raise ImportError(
        "The 'mcp' package is required to run the MCP server integration but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[mcp]'"
    ) from e


from .tool_context import (
    RawFunctionTool,
    ToolError,
    function_tool,
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

MCPTool = RawFunctionTool


@dataclass
class MCPToolResultContext:
    """Context passed to an MCPToolResultResolver callback."""

    tool_name: str
    arguments: dict[str, Any]
    result: mcp.types.CallToolResult


MCPToolResultResolver = Callable[[MCPToolResultContext], Any | Awaitable[Any]]


def _default_tool_result_resolver(ctx: MCPToolResultContext) -> str:
    # TODO(theomonnom): handle images & binary messages
    if len(ctx.result.content) == 1:
        return str(ctx.result.content[0].model_dump_json())
    elif len(ctx.result.content) > 1:
        return json.dumps([item.model_dump() for item in ctx.result.content])

    raise ToolError(
        f"Tool '{ctx.tool_name}' completed without producing a result. "
        "This might indicate an issue with internal processing."
    )


class MCPServer(ABC):
    def __init__(
        self,
        *,
        client_session_timeout_seconds: float,
        tool_result_resolver: MCPToolResultResolver | None = None,
    ) -> None:
        self._client: ClientSession | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._read_timeout = client_session_timeout_seconds
        self._tool_result_resolver: MCPToolResultResolver = (
            tool_result_resolver or _default_tool_result_resolver
        )

        self._cache_dirty = True
        self._lk_tools: list[MCPTool] | None = None

    @property
    def initialized(self) -> bool:
        return self._client is not None

    def invalidate_cache(self) -> None:
        self._cache_dirty = True

    async def initialize(self) -> None:
        try:
            streams = await self._exit_stack.enter_async_context(self.client_streams())
            receive_stream, send_stream = streams[0], streams[1]
            self._client = await self._exit_stack.enter_async_context(
                ClientSession(
                    receive_stream,
                    send_stream,
                    read_timeout_seconds=timedelta(seconds=self._read_timeout)
                    if self._read_timeout
                    else None,
                )
            )
            await self._client.initialize()  # type: ignore[union-attr]
        except Exception:
            await self.aclose()
            raise

    async def list_tools(self) -> list[MCPTool]:
        if self._client is None:
            raise RuntimeError("MCPServer isn't initialized")

        if not self._cache_dirty and self._lk_tools is not None:
            return self._lk_tools

        tools = await self._client.list_tools()
        lk_tools = [
            self._make_function_tool(tool.name, tool.description, tool.inputSchema, tool.meta)
            for tool in tools.tools
        ]

        self._lk_tools = lk_tools
        self._cache_dirty = False
        return lk_tools

    def _make_function_tool(
        self,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
        meta: dict[str, Any] | None,
    ) -> MCPTool:
        async def _tool_called(raw_arguments: dict[str, Any]) -> Any:
            # In case (somehow), the tool is called after the MCPServer aclose.
            if self._client is None:
                raise ToolError(
                    "Tool invocation failed: internal service is unavailable. "
                    "Please check that the MCPServer is still running."
                )

            tool_result = await self._client.call_tool(name, raw_arguments)

            if tool_result.isError:
                error_str = "\n".join(
                    part.text if hasattr(part, "text") else str(part)
                    for part in tool_result.content
                )
                raise ToolError(error_str)

            ctx = MCPToolResultContext(tool_name=name, arguments=raw_arguments, result=tool_result)
            resolved = self._tool_result_resolver(ctx)
            if asyncio.iscoroutine(resolved):
                resolved = await resolved
            return resolved

        raw_schema = {
            "name": name,
            "description": description,
            "parameters": input_schema,
        }
        if meta:
            raw_schema["meta"] = meta

        return function_tool(_tool_called, raw_schema=raw_schema)

    async def aclose(self) -> None:
        try:
            await self._exit_stack.aclose()
        finally:
            self._client = None
            self._lk_tools = None

    @abstractmethod
    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
        | tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback,
        ]
    ]: ...


class MCPServerHTTP(MCPServer):
    """
    HTTP-based MCP server with configurable transport type and tool filtering.

    Args:
        url: The URL of the MCP server
        transport_type: Explicit transport type - "sse" or "streamable_http".
            If None, transport type is auto-detected from URL path:
            - URLs ending with 'sse' use Server-Sent Events (SSE) transport
            - URLs ending with 'mcp' use streamable HTTP transport
            - For other URLs, defaults to SSE transport for backward compatibility
        allowed_tools: Optional list of tool names to filter. If provided, only
            tools whose names are in this list will be available. If None, all
            tools from the server will be available.
        headers: Optional HTTP headers to include in requests
        timeout: Connection timeout in seconds (default: 5)
        sse_read_timeout: SSE read timeout in seconds (default: 300)
        client_session_timeout_seconds: Client session timeout in seconds (default: 5)

    Note: SSE transport is being deprecated in favor of streamable HTTP transport.
    See: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206
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
        self.headers = headers
        self._timeout = timeout
        self._sse_read_timeout = sse_read_timeout
        self._allowed_tools = set(allowed_tools) if allowed_tools else None

        # Determine transport type: explicit > URL-based detection
        if transport_type is not None:
            if transport_type not in ("sse", "streamable_http"):
                raise ValueError(
                    f"transport_type must be 'sse' or 'streamable_http', got '{transport_type}'"
                )
            self._use_streamable_http = transport_type == "streamable_http"
        else:
            # Fall back to URL-based detection for backward compatibility
            self._use_streamable_http = self._should_use_streamable_http(url)

    def _should_use_streamable_http(self, url: str) -> bool:
        """
        Determine transport type based on URL path (for backward compatibility).

        Returns True for streamable HTTP if URL ends with 'mcp',
        False for SSE if URL ends with 'sse' or for backward compatibility.
        """
        parsed_url = urlparse(url)
        path_lower = parsed_url.path.lower().rstrip("/")
        return path_lower.endswith("/mcp")

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
        | tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback,
        ]
    ]:
        if self._use_streamable_http:

            @asynccontextmanager
            async def _streamable_http_with_client():  # type: ignore[no-untyped-def]
                async with httpx.AsyncClient(
                    headers=self.headers or {},
                    timeout=httpx.Timeout(self._timeout, read=self._sse_read_timeout),
                ) as http_client:
                    async with streamable_http_client(
                        url=self.url, http_client=http_client
                    ) as streams:
                        yield streams

            return _streamable_http_with_client()  # type: ignore[return-value]
        else:
            return sse_client(  # type: ignore[no-any-return]
                url=self.url,
                headers=self.headers,
                timeout=self._timeout,
                sse_read_timeout=self._sse_read_timeout,
            )

    async def list_tools(self) -> list[MCPTool]:
        """
        List tools from the MCP server, filtered by allowed_tools if specified.
        """
        all_tools = await super().list_tools()

        # If no filter is set, return all tools
        if self._allowed_tools is None:
            return all_tools

        # Filter tools by name
        return self._filter_tools(all_tools)

    def _filter_tools(self, tools: list[MCPTool]) -> list[MCPTool]:
        """
        Filter tools by allowed_tools if specified.
        """
        if self._allowed_tools is None:
            return tools

        filtered_tools: list[MCPTool] = []
        for tool in tools:
            # Get tool name based on tool type
            if is_function_tool(tool):
                tool_name = get_function_info(tool).name
            elif is_raw_function_tool(tool):
                tool_name = get_raw_function_info(tool).name
            else:
                # Fallback: skip tools we can't identify
                continue

            if tool_name in self._allowed_tools:
                filtered_tools.append(tool)  # type: ignore[arg-type]

        return filtered_tools

    def __repr__(self) -> str:
        transport_type = "streamable_http" if self._use_streamable_http else "sse"
        allowed_str = f", allowed_tools={list(self._allowed_tools)}" if self._allowed_tools else ""
        return f"MCPServerHTTP(url={self.url}, transport={transport_type}{allowed_str})"


class MCPServerStdio(MCPServer):
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

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        return stdio_client(  # type: ignore[no-any-return]
            StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"


class MCPToolset(Toolset):
    """A toolset that exposes tools from a Model Context Protocol (MCP) server.

    MCPToolset wraps an ``MCPServer`` instance and makes its tools available for
    use by an ``Agent``. On ``setup()``, it connects to the MCP server (if not
    already connected), fetches the available tools, and caches them locally.
    """

    def __init__(self, *, id: str, mcp_server: MCPServer) -> None:
        super().__init__(id=id)
        self._mcp_server = mcp_server
        self._initialized = False
        self._lock = asyncio.Lock()

    async def setup(self, *, reload: bool = False) -> Self:
        """Initialize the MCP server connection and fetch available tools.

        If the MCP server is not yet connected, this will call
        ``MCPServer.initialize()``. Subsequent calls are no-ops unless
        ``reload=True``.

        Args:
            reload: If ``True``, invalidate the tool cache and re-fetch
                tools from the MCP server even if already initialized.
        """
        await super().setup()
        async with self._lock:
            if not reload and self._initialized:
                return self

            if not self._mcp_server.initialized:
                await self._mcp_server.initialize()
            elif reload:
                self._mcp_server.invalidate_cache()

            tools = await self._mcp_server.list_tools()
            self._tools = tools
            self._initialized = True
            return self

    def filter_tools(self, filter_fn: Callable[[MCPTool], bool]) -> Self:
        """Filter the toolset's tools in-place using a predicate."""
        self._tools = [
            tool for tool in self._tools if isinstance(tool, MCPTool) and filter_fn(tool)
        ]
        return self

    async def aclose(self) -> None:
        try:
            await super().aclose()
            await self._mcp_server.aclose()
        finally:
            self._initialized = False
            self._tools = []
