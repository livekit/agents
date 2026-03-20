import asyncio
from collections.abc import Callable

from typing_extensions import Self

from ..mcp import MCPServer, MCPTool
from ..tool_context import Toolset


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
