import json
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

try:
    from mcp import ClientSession, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.types import JSONRPCMessage
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

    @property
    def initialized(self) -> bool:
        return self._client is not None

    async def initialize(self) -> None:
        try:
            receive_stream, send_stream = await self._exit_stack.enter_async_context(
                self.client_streams()
            )
            self._client = await self._exit_stack.enter_async_context(
                ClientSession(
                    receive_stream,
                    send_stream,
                    read_timeout_seconds=timedelta(seconds=self._read_timeout)
                    if self._read_timeout
                    else None,
                )
            )
            await self._client.initialize()
            self._initialized = True
        except Exception:
            await self.aclose()
            raise

    async def list_tools(self) -> list[MCPTool]:
        if self._client is None:
            raise RuntimeError("MCPServer isn't initialized")

        tools = await self._client.list_tools()

        def make_function_tool(name: str):
            async def _tool_called(raw_arguments: dict) -> Any:
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

            return _tool_called

        return [
            function_tool(
                make_function_tool(tool.name),
                raw_schema={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            )
            for tool in tools.tools
        ]

    async def aclose(self) -> None:
        try:
            await self._exit_stack.aclose()
        finally:
            self._client = None

    @abstractmethod
    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]: ...


class MCPServerHTTP(MCPServer):
    # SSE is going to get replaced soon: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206

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
        self._see_read_timeout = sse_read_timeout

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self._timeout,
            sse_read_timeout=self._see_read_timeout,
        )

    def __repr__(self) -> str:
        return f"MCPServerHTTP(url={self.url})"


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
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return stdio_client(
            StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"
