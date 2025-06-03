# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

try:
    from mcp import ClientSession, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
    from mcp.shared.message import SessionMessage
except ImportError as e:
    raise ImportError(
        "The 'mcp' package is required to run the MCP server integration but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[mcp]'"
    ) from e


from .tool_context import RawFunctionTool, ToolError, function_tool

MCPTool = RawFunctionTool


class MCPServer(ABC):
    def __init__(self, *, client_session_timeout_seconds: float) -> None:
        self._client: ClientSession | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._read_timeout = client_session_timeout_seconds

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
            self._initialized = True
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
            self._make_function_tool(tool.name, tool.description, tool.inputSchema)
            for tool in tools.tools
        ]

        self._lk_tools = lk_tools
        self._cache_dirty = False
        return lk_tools

    def _make_function_tool(
        self, name: str, description: str | None, input_schema: dict[str, Any]
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
                error_str = "\n".join(str(part) for part in tool_result.content)
                raise ToolError(error_str)

            # TODO(theomonnom): handle images & binary messages
            if len(tool_result.content) == 1:
                return tool_result.content[0].model_dump_json()
            elif len(tool_result.content) > 1:
                return json.dumps([item.model_dump() for item in tool_result.content])

            raise ToolError(
                f"Tool '{name}' completed without producing a result. "
                "This might indicate an issue with internal processing."
            )

        return function_tool(
            _tool_called,
            raw_schema={"name": name, "description": description, "parameters": input_schema},
        )

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
    HTTP-based MCP server to detect transport type based on URL path.

    - URLs ending with 'sse' use Server-Sent Events (SSE) transport
    - URLs ending with 'mcp' use streamable HTTP transport
    - For other URLs, defaults to SSE transport for backward compatibility

    Note: SSE transport is being deprecated in favor of streamable HTTP transport.
    See: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.url = url
        self.headers = headers
        self._timeout = timeout
        self._sse_read_timeout = sse_read_timeout
        self._use_streamable_http = self._should_use_streamable_http(url)

    def _should_use_streamable_http(self, url: str) -> bool:
        """
        Determine transport type based on URL path.

        Returns True for streamable HTTP if URL ends with 'mcp',
        False for SSE if URL ends with 'sse' or for backward compatibility.
        """
        parsed_url = urlparse(url)
        path_lower = parsed_url.path.lower().rstrip("/")
        return path_lower.endswith("mcp")

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
            return streamablehttp_client(  # type: ignore[no-any-return]
                url=self.url,
                headers=self.headers,
                timeout=timedelta(seconds=self._timeout),
                sse_read_timeout=timedelta(seconds=self._sse_read_timeout),
            )
        else:
            return sse_client(  # type: ignore[no-any-return]
                url=self.url,
                headers=self.headers,
                timeout=self._timeout,
                sse_read_timeout=self._sse_read_timeout,
            )

    def __repr__(self) -> str:
        transport_type = "streamable_http" if self._use_streamable_http else "sse"
        return f"MCPServerHTTP(url={self.url}, transport={transport_type})"


class MCPServerStdio(MCPServer):
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
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
