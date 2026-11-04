"""MCP servers as tool sources. The client sessions and their transports are the core's
(``_core.McpSession``)."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Self, cast

from pydantic import JsonValue

from .. import _core
from .._generated.mcp import MCP_IDLE_CLOSE_SECONDS, MCPToolOptions, ResolvedToolOptions
from ..log import logger
from ..utils import aio
from ..voice.events import RunContext
from .tool_context import AsyncToolset, RawFunctionTool, function_tool

# the documented surface (docs/tools.md "Toolsets")
__all__ = [
    "MCPServerHTTP",
    "MCPServerStdio",
    "MCPToolOptions",
    "MCPToolset",
]


class _Progress:
    """A ``report_progress`` tool's progress messages, as ``ctx.update()`` calls."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def _core_update(self, _invocation: object, message: str) -> None:
        await self._ctx.update(message)


class MCPServer:
    """An MCP server: a session that reconnects on its own after losing its connection."""

    def __init__(
        self,
        *,
        init_timeout: float,
        request_timeout: float | None,
        read_timeout: float | None = None,
    ) -> None:
        self._session = _core.McpSession(init_timeout, request_timeout, read_timeout)

    @property
    def initialized(self) -> bool:
        return self._session.connected

    @property
    def _retry_in(self) -> float:
        """Seconds until the session's backoff lets it try to connect again."""
        return self._session.retry_in

    async def initialize(self) -> None:
        raise NotImplementedError

    async def list_tools(
        self, *, tool_options: dict[str, MCPToolOptions] | None = None
    ) -> list[RawFunctionTool]:
        options = tool_options or {}
        tools = cast(list[dict[str, Any]], await self._session.list_tools())
        return [
            self._make_function_tool(
                tool["name"],
                tool.get("description"),
                tool.get("inputSchema") or {},
                tool.get("_meta"),
                options=_core.resolve_mcp_tool_options(options.get(tool["name"])),
            )
            for tool in tools
        ]

    def _make_function_tool(
        self,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
        meta: dict[str, Any] | None,
        *,
        options: ResolvedToolOptions,
    ) -> RawFunctionTool:
        timeout = options["timeout"]

        async def _call(raw_arguments: dict[str, Any], progress: _Progress | None) -> Any:
            result = await self._session.call_tool(
                name,
                raw_arguments,
                timeout,
                progress,
                (asyncio.get_running_loop(), contextvars.copy_context()),
            )
            return _core.mcp_render_result(name, result)

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
            raw_schema=cast(
                dict[str, JsonValue], _core.mcp_raw_schema(name, description, input_schema, meta)
            ),
            flags=options["flags"],
            on_duplicate=options["on_duplicate"],
            duplicate_scope=options["duplicate_scope"],
            retries=options["retries"],
        )

    async def aclose(self) -> None:
        await self._session.close()


class MCPServerHTTP(MCPServer):
    """An MCP server over Streamable HTTP.

    Args:
        url: The server's MCP endpoint.
        headers: HTTP headers sent with every request.
        init_timeout: Seconds to connect, complete the handshake and list the tools.
        read_timeout: Seconds a reply stream may stay idle; ``None`` for no limit.
        request_timeout: Seconds one request may go without a reply or progress; ``None`` for
            no limit. ``MCPToolOptions(timeout=)`` sets one tool's.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        init_timeout: float = 5,
        read_timeout: float | None = 300,
        request_timeout: float | None = 30,
    ) -> None:
        super().__init__(
            init_timeout=init_timeout, request_timeout=request_timeout, read_timeout=read_timeout
        )
        self.url = url
        self.headers = headers or {}

    async def initialize(self) -> None:
        await self._session.connect_http(self.url, [(k, str(v)) for k, v in self.headers.items()])

    def __repr__(self) -> str:
        return f"MCPServerHTTP(url={self.url})"


class MCPServerStdio(MCPServer):
    """An MCP server run as a child process, over its stdio; restarted if it exits.

    Args:
        command: The program to run.
        args: Its arguments.
        env: Environment variables, over a minimal inherited set.
        cwd: Its working directory.
        init_timeout: Seconds to start it, complete the handshake and list the tools.
        request_timeout: Seconds one request may go without a reply or progress; ``None`` for
            no limit. ``MCPToolOptions(timeout=)`` sets one tool's.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        init_timeout: float = 5,
        request_timeout: float | None = 30,
    ) -> None:
        super().__init__(init_timeout=init_timeout, request_timeout=request_timeout)
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

    async def initialize(self) -> None:
        cwd = str(self.cwd) if self.cwd is not None else None
        await self._session.connect_stdio(
            self.command, self.args, list((self.env or {}).items()), cwd
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"


class MCPToolset(AsyncToolset):
    """An ``MCPServer``'s tools for an ``Agent``; ``setup()`` connects and fetches them.

    A server that cannot be reached leaves the toolset without tools instead of failing the run
    or the call; it keeps trying in the background, and later runs and activations get the tools
    once it answers. The connection stays open for a while after the last run, for the next.

    Per-tool behavior is set with ``MCPToolOptions`` in ``tool_options`` (keyed by tool name);
    unlisted tools are plain blocking calls.

    Example::

        MCPToolset(
            id="mcp",
            mcp_server=MCPServerHTTP("https://mcp.example.com/mcp"),
            tool_options={
                "book_flight": MCPToolOptions(
                    on_duplicate="confirm", report_progress=True, timeout=120
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
        self._listed = False
        self._lock = asyncio.Lock()
        self._recovery: asyncio.Task[None] | None = None
        self._idle_close: asyncio.TimerHandle | None = None
        self._closing: asyncio.Task[None] | None = None

    async def setup(self) -> Self:
        """Fetch the server's tools, connecting it when it is not; without them while the server
        cannot be reached."""
        await super().setup()
        async with self._lock:
            if self._idle_close is not None:
                self._idle_close.cancel()
                self._idle_close = None
            if self._listed or self._recovery is not None:
                return self
            try:
                await self._list()
            except Exception as error:
                logger.warning(
                    "MCP server unavailable, continuing without its tools",
                    extra={"mcp_server": repr(self._mcp_server), "error": str(error)},
                )
                self._recovery = asyncio.create_task(self._recover())
            return self

    async def _list(self) -> None:
        if not self._mcp_server.initialized:
            await self._mcp_server.initialize()
        self._tools = list(await self._mcp_server.list_tools(tool_options=self._tool_options))
        self._listed = True

    async def _recover(self) -> None:
        """Retry the listing at the session's own backoff until the server answers."""
        while True:
            await asyncio.sleep(self._mcp_server._retry_in)
            try:
                await self._list()
            except Exception:
                continue
            self._recovery = None
            return

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            if self._recovery is not None:
                await aio.cancel_and_wait(self._recovery)
                self._recovery = None
            self._idle_close = asyncio.get_running_loop().call_later(
                MCP_IDLE_CLOSE_SECONDS, self._close_idle
            )

    def _close_idle(self) -> None:
        self._idle_close = None
        self._listed = False
        self._tools = []
        # Holds the close until it settles; a later setup reconnects.
        self._closing = asyncio.create_task(self._mcp_server.aclose())
