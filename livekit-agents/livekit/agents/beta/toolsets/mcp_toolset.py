import asyncio
from collections.abc import Callable

from typing_extensions import Self

from ...llm.mcp import MCPServer, MCPTool
from ...llm.tool_context import Toolset


class MCPToolset(Toolset):
    def __init__(self, *, id: str, mcp_server: MCPServer) -> None:
        super().__init__(id=id)
        self._mcp_server = mcp_server
        self._initialized = False
        self._lock = asyncio.Lock()

    async def setup(self, *, reload: bool = False) -> Self:
        async with self._lock:
            if not reload and self._initialized:
                return self

            if not self._mcp_server.initialized:
                await self._mcp_server.initialize()

            tools = await self._mcp_server.list_tools()
            self._tools = tools
            self._initialized = True
            return self

    def filter_tools(self, filter_fn: Callable[[MCPTool], bool]) -> Self:
        self._tools = list(filter(filter_fn, self._tools))
        return self

    async def aclose(self) -> None:
        await self._mcp_server.aclose()
