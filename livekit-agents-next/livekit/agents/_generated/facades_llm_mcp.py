# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class MCPServerStdio:
    """`MCPServerStdio`: an MCP server run as a child process, over its stdio; restarted if it exits."""

    def __init__(
        self,
        command: str,
        args: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        init_timeout: float = 5.0,
        request_timeout: float | None = 30.0,
    ) -> None:
        """Args:
            command: The program to run.
            args: Its arguments.
            env: Environment variables, over a minimal inherited set.
            cwd: Its working directory.
            init_timeout: Seconds to start it, complete the handshake and list the tools.
            request_timeout: Seconds one request may go without a reply or progress; ``None`` for
                no limit. ``MCPToolOptions(timeout=)`` sets one tool's.
        """
        self._native = _core.McpSession(init_timeout, request_timeout)
        self._command = command
        self._args = args
        self._env = env
        self._cwd = cwd

    @property
    def initialized(self) -> bool:
        """Whether the server is connected."""
        return self._native.connected

    async def initialize(self) -> None:
        """Start the server and complete the MCP handshake."""
        await self._native.connect_stdio(self._command, self._args, self._env, self._cwd)

    async def list_tools(self) -> object:
        """The server's tool listing, every page."""
        return await self._native.list_tools()

    async def aclose(self) -> None:
        await self._native.close()


class MCPServerHTTP:
    """`MCPServerHTTP`'s connection: an MCP server over Streamable HTTP. The handwritten subclass
    takes the credentials, which can be functions of a run, and connects with them.
    """

    def __init__(
        self,
        url: str,
        *,
        init_timeout: float = 5.0,
        read_timeout: float | None = 300.0,
        request_timeout: float | None = 30.0,
    ) -> None:
        """Args:
            url: The server's MCP endpoint.
            init_timeout: Seconds to connect, complete the handshake and list the tools.
            read_timeout: Seconds a reply stream may stay idle; ``None`` for no limit.
            request_timeout: Seconds one request may go without a reply or progress; ``None`` for
                no limit. ``MCPToolOptions(timeout=)`` sets one tool's.
        """
        self._native = _core.McpSession(init_timeout, request_timeout, read_timeout)
        self._url = url

    @property
    def initialized(self) -> bool:
        """Whether the server is connected."""
        return self._native.connected

    async def list_tools(self) -> object:
        """The server's tool listing, every page."""
        return await self._native.list_tools()

    async def aclose(self) -> None:
        await self._native.close()


MCPServerStdio.__module__ = "livekit.agents.llm.mcp"
