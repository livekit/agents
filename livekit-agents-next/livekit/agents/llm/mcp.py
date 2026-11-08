"""MCP servers as tool sources (docs/tools.md, "MCP servers").

The Rust core does the MCP work: the protocol, the stdio and Streamable HTTP transports, listing
and paginating tools, rendering their schemas and results, the reconnect backoff, recovering an
unreachable server in the background and closing an idle connection (``_core.McpSession`` and
``_core._core_McpToolsetLifecycle``). ``MCPServerStdio`` is generated over it whole, and so is
the connection half of ``MCPServerHTTP``.

What stays here is what only the host can do:

- credentials that come from a run (``headers=``/``auth=`` functions of the ``RunContext``),
  which are application code;
- one toolset per set of run credentials, kept by a call for its whole length;
- each MCP tool as a function tool, because tools are host objects.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, Self, cast

from pydantic import JsonValue

from .. import _core
from .._generated import facades_llm_mcp as _facades, public_llm_mcp as _public
from .._generated.mcp import ResolvedToolOptions
from .._generated.public_llm_mcp import *  # noqa: F403
from ..voice.events import RunContext
from .tool_context import AsyncToolset, RawFunctionTool, Toolset, function_tool

# the documented surface (docs/tools.md "Toolsets")
__all__ = [
    "MCPServer",
    "MCPServerHTTP",
    "MCPToolset",
]
__all__ += _public.__all__


HeadersFor = Callable[[RunContext[Any]], dict[str, str]]
"""A run's HTTP headers, from its ``RunContext``."""
TokenFor = Callable[[RunContext[Any]], "str | None"]
"""A run's bearer token, from its ``RunContext``."""
_Credentials = tuple[tuple[tuple[str, str], ...], "str | None"]
"""Headers (sorted name/value pairs) and a bearer token, comparable and hashable."""


# --- servers ----------------------------------------------------------------------------------


class MCPServerHTTP(_facades.MCPServerHTTP):
    """An MCP server over Streamable HTTP.

    Args:
        url: The server's MCP endpoint.
        headers: HTTP headers sent with every request, or a function of the run's
            ``RunContext`` returning them.
        auth: A bearer token, or a function of the run's ``RunContext`` returning it, e.g.
            ``lambda ctx: ctx.context.mcp_token``.
        init_timeout: Seconds to connect, complete the handshake and list the tools.
        read_timeout: Seconds a reply stream may stay idle; ``None`` for no limit.
        request_timeout: Seconds one request may go without a reply or progress; ``None`` for
            no limit. ``MCPToolOptions(timeout=)`` sets one tool's.

    With a function, each run connects with its own credentials: runs whose credentials are
    the same share a connection, others never do. A changed result reconnects before the next
    request, so a call survives a token refresh. The functions must not block.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | HeadersFor | None = None,
        auth: str | TokenFor | None = None,
        init_timeout: float = 5,
        read_timeout: float | None = 300,
        request_timeout: float | None = 30,
    ) -> None:
        super().__init__(
            url,
            init_timeout=init_timeout,
            read_timeout=read_timeout,
            request_timeout=request_timeout,
        )
        self._headers = headers
        self._auth = auth
        self._timeouts = (init_timeout, read_timeout, request_timeout)
        # set on a run's copy (``_for_run``): its credentials, resolved again before a request
        self._run_credentials: Callable[[], _Credentials] | None = None
        self._connected = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def _per_run(self) -> bool:
        return callable(self._headers) or callable(self._auth)

    def _credentials(self, ctx: RunContext[Any] | None) -> _Credentials:
        """The headers and token to connect with; ``ctx`` is required when either is per run."""
        headers, auth = self._headers, self._auth
        if callable(headers):
            assert ctx is not None
            headers = headers(ctx)
        if callable(auth):
            assert ctx is not None
            auth = auth(ctx)
        return tuple(sorted((name, str(value)) for name, value in (headers or {}).items())), auth

    def _for_run(self, ctx: RunContext[Any]) -> MCPServerHTTP:
        """A copy of this server that connects with ``ctx``'s credentials."""
        init_timeout, read_timeout, request_timeout = self._timeouts
        copy = MCPServerHTTP(
            self._url,
            init_timeout=init_timeout,
            read_timeout=read_timeout,
            request_timeout=request_timeout,
        )
        copy._run_credentials = lambda: self._credentials(ctx)
        return copy

    async def initialize(self) -> None:
        """Connect and complete the MCP handshake."""
        if self._run_credentials is not None:
            await self._connect(self._run_credentials())
        elif self._per_run:
            raise RuntimeError("an MCPServerHTTP with per-run credentials connects per run")
        else:
            await self._connect(self._credentials(None))

    async def _connect(self, credentials: _Credentials) -> None:
        headers, auth = credentials
        await self._native.connect_http(self._url, dict(headers), auth)
        self._connected = True

    async def _refresh(self) -> None:
        """Before each request: the core keeps the connection when the run's credentials are
        unchanged and replaces it when they changed. A closed server is not reopened."""
        if self._run_credentials is not None and self._connected:
            await self._connect(self._run_credentials())

    async def aclose(self) -> None:
        self._connected = False
        await super().aclose()

    def __repr__(self) -> str:
        return f"MCPServerHTTP(url={self._url})"


MCPServer = MCPServerHTTP | _facades.MCPServerStdio
"""An MCP server an ``MCPToolset`` takes its tools from."""


# --- the toolset ------------------------------------------------------------------------------


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
        # the core decides when to list, recover and close; it calls back _core_list and
        # _core_close_idle
        self._lifecycle = _core._core_McpToolsetLifecycle(_describe(mcp_server), mcp_server._native)
        # per-run credentials: one toolset per credential set, kept by a call for its length
        self._by_credentials: dict[str, MCPToolset] = {}
        self._by_call: weakref.WeakKeyDictionary[object, MCPToolset] = weakref.WeakKeyDictionary()
        self._release: Callable[[], object] | None = None

    @property
    def _per_run(self) -> bool:
        server = self._mcp_server
        return isinstance(server, MCPServerHTTP) and server._per_run

    def for_run(self, ctx: RunContext[Any]) -> Toolset:
        """This toolset, or with per-run credentials the one for ``ctx``'s: shared by the runs
        whose credentials are the same, and kept by a call for its whole length."""
        if not self._per_run:
            return self
        call = ctx.live_session
        if call is not None and (kept := self._by_call.get(call)) is not None:
            return kept
        server = cast(MCPServerHTTP, self._mcp_server)
        # keyed by a digest, so the dict never holds the credentials themselves
        key = hashlib.sha256(repr(server._credentials(ctx)).encode()).hexdigest()
        if (toolset := self._by_credentials.get(key)) is None:
            toolset = MCPToolset(
                id=self.id, mcp_server=server._for_run(ctx), tool_options=self._tool_options
            )
            toolset._tool_handling = self._tool_handling
            toolset._release = functools.partial(self._by_credentials.pop, key, None)
            self._by_credentials[key] = toolset
        if call is not None:
            self._by_call[call] = toolset
        return toolset

    async def setup(self) -> Self:
        """Fetch the server's tools, connecting it when it is not; without them while the server
        cannot be reached, which is listed again in the background at the session's backoff.
        A per-run toolset lists nothing itself: each run's copy does."""
        await super().setup()
        if not self._per_run:
            await self._lifecycle.setup(self)
        return self

    async def aclose(self) -> None:
        """Release the toolset: a recovery stops, and the connection closes after
        ``MCP_IDLE_CLOSE_SECONDS`` unless a run sets the toolset up again first."""
        try:
            await super().aclose()
        finally:
            await self._lifecycle.release(self)

    async def _core_list(self) -> None:
        server = self._mcp_server
        if not server.initialized:
            await server.initialize()
        await _refresh(server)
        listed = cast(list[dict[str, Any]], await server.list_tools())
        self._tools = [
            _function_tool(
                server,
                tool,
                _core.resolve_mcp_tool_options(self._tool_options.get(tool["name"])),
            )
            for tool in listed
        ]

    async def _core_close_idle(self) -> None:
        self._tools = []
        if self._release is not None:
            self._release()
        await self._mcp_server.aclose()


# --- tools ------------------------------------------------------------------------------------


def _function_tool(
    server: MCPServer, tool: dict[str, Any], options: ResolvedToolOptions
) -> RawFunctionTool:
    """``tool`` (an MCP listing entry) as a function tool calling ``server``. With
    ``report_progress``, the tool takes the ``RunContext`` so its progress reaches
    ``ctx.update()``."""
    name = tool["name"]

    async def call(arguments: dict[str, Any], progress: RunContext | None) -> Any:
        await _refresh(server)
        result = await server._native.call_tool(
            name,
            arguments,
            options["timeout"],
            progress,
            (asyncio.get_running_loop(), contextvars.copy_context()),
        )
        return _core.mcp_render_result(name, result)

    impl: Callable[..., Awaitable[Any]]
    if options["report_progress"]:

        async def impl(ctx: RunContext, raw_arguments: dict[str, Any]) -> Any:
            return await call(raw_arguments, ctx)

    else:

        async def impl(raw_arguments: dict[str, Any]) -> Any:
            return await call(raw_arguments, None)

    schema = _core.mcp_raw_schema(
        name, tool.get("description"), tool.get("inputSchema") or {}, tool.get("_meta")
    )
    return function_tool(
        impl,
        raw_schema=cast(dict[str, JsonValue], schema),
        flags=options["flags"],
        on_duplicate=options["on_duplicate"],
        duplicate_scope=options["duplicate_scope"],
        retries=options["retries"],
    )


async def _refresh(server: MCPServer) -> None:
    if isinstance(server, MCPServerHTTP):
        await server._refresh()


def _describe(server: MCPServer) -> str:
    """The server as logs name it."""
    if isinstance(server, MCPServerHTTP):
        return repr(server)
    return f"MCPServerStdio(command={server._command}, args={list(server._args)})"
