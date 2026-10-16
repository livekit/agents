from __future__ import annotations

import asyncio
import contextlib
from typing import Any, get_type_hints

import mcp.types
import pytest
from mcp.shared.exceptions import MCPError

from livekit.agents.llm import mcp as mcp_module
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.llm.mcp import (
    MCPServer,
    MCPToolOptions,
    MCPToolset,
    _resolve_tool_options,
)
from livekit.agents.llm.tool_context import ToolError, ToolFlag, is_raw_function_tool
from livekit.agents.llm.utils import is_context_type
from livekit.agents.voice.tool_executor import has_cancellable_tool

pytestmark = pytest.mark.unit


class _FakeToolDesc:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "echo back"
        self.input_schema: dict[str, Any] = {"type": "object", "properties": {}}
        self.meta = None


class _FakeListToolsResult:
    def __init__(self, names: list[str]) -> None:
        self.tools = [_FakeToolDesc(n) for n in names]


class _FakeClient:
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.list_tools_calls = 0

    async def list_tools(self) -> _FakeListToolsResult:
        self.list_tools_calls += 1
        return _FakeListToolsResult(self._names)


class _FakeMCPServer(MCPServer):
    """Minimal MCPServer with a fake client — no real streams or network."""

    def __init__(self, tool_names: list[str] | None = None) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self._client = _FakeClient(tool_names or ["echo"])  # type: ignore[assignment]

    def client_streams(self):  # type: ignore[no-untyped-def]  # not used in these tests
        raise NotImplementedError


def _has_run_context_param(tool: object) -> bool:
    hints = get_type_hints(tool)
    return any(is_context_type(h, allow_subclasses=True) for h in hints.values())


def _build_mcp_tool(server: _FakeMCPServer, options: MCPToolOptions):
    return server._make_function_tool(
        name="echo",
        description="echo back",
        input_schema={"type": "object", "properties": {}},
        meta=None,
        options=_resolve_tool_options(options),
    )


def test_mcp_toolset_is_async_toolset_subclass() -> None:
    assert issubclass(MCPToolset, AsyncToolset)


def test_default_options_build_a_plain_blocking_tool() -> None:
    tool = _build_mcp_tool(_FakeMCPServer(), MCPToolOptions())

    assert is_raw_function_tool(tool)
    assert not _has_run_context_param(tool)
    assert tool.info.flags == ToolFlag.NONE
    assert tool.info.on_duplicate == "allow"


def test_report_progress_adds_run_context_param() -> None:
    blocking = _build_mcp_tool(_FakeMCPServer(), MCPToolOptions(report_progress=False))
    progress = _build_mcp_tool(_FakeMCPServer(), MCPToolOptions(report_progress=True))

    assert not _has_run_context_param(blocking)
    assert _has_run_context_param(progress)


def test_flags_and_on_duplicate_pass_through_independently() -> None:
    # flags / on_duplicate are decoupled from report_progress
    tool = _build_mcp_tool(
        _FakeMCPServer(),
        MCPToolOptions(
            flags=ToolFlag.CANCELLABLE | ToolFlag.IGNORE_ON_ENTER,
            on_duplicate="reject",
            report_progress=False,
        ),
    )

    assert ToolFlag.CANCELLABLE in tool.info.flags
    assert ToolFlag.IGNORE_ON_ENTER in tool.info.flags
    assert tool.info.on_duplicate == "reject"
    # cancellable but still blocking — no ctx param
    assert not _has_run_context_param(tool)


@pytest.mark.asyncio
async def test_list_tools_applies_named_options_else_blocking() -> None:
    server = _FakeMCPServer(["get_weather", "book_flight"])
    tools = await server.list_tools(
        tool_options={
            "book_flight": MCPToolOptions(flags=ToolFlag.CANCELLABLE, report_progress=True),
        }
    )

    by_name = {t.info.name: t for t in tools}
    assert ToolFlag.CANCELLABLE in by_name["book_flight"].info.flags
    assert _has_run_context_param(by_name["book_flight"])
    # unlisted tools fall back to a plain blocking call
    assert by_name["get_weather"].info.flags == ToolFlag.NONE
    assert not _has_run_context_param(by_name["get_weather"])


@pytest.mark.asyncio
async def test_setup_applies_per_tool_options() -> None:
    ts = MCPToolset(
        id="m",
        mcp_server=_FakeMCPServer(["get_weather", "book_flight"]),
        tool_options={
            "book_flight": MCPToolOptions(
                flags=ToolFlag.CANCELLABLE, on_duplicate="confirm", report_progress=True
            ),
        },
    )
    await ts.setup()

    by_name = {t.info.name: t for t in ts._tools}
    assert ToolFlag.CANCELLABLE in by_name["book_flight"].info.flags
    assert by_name["book_flight"].info.on_duplicate == "confirm"
    assert _has_run_context_param(by_name["book_flight"])

    assert by_name["get_weather"].info.flags == ToolFlag.NONE
    assert not _has_run_context_param(by_name["get_weather"])

    assert has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_setup_default_is_all_blocking() -> None:
    ts = MCPToolset(id="m", mcp_server=_FakeMCPServer(["get_weather", "book_flight"]))
    await ts.setup()

    assert len(ts._tools) == 2
    assert not has_cancellable_tool(ts.tools)


@pytest.mark.asyncio
async def test_list_tools_caches_raw_fetch() -> None:
    server = _FakeMCPServer(["echo"])

    await server.list_tools()
    await server.list_tools()
    assert server._client.list_tools_calls == 1  # raw descriptors cached

    server.invalidate_cache()
    await server.list_tools()
    assert server._client.list_tools_calls == 2  # re-fetched after invalidation


@pytest.mark.asyncio
async def test_list_tools_rebuilds_per_options() -> None:
    server = _FakeMCPServer(["echo"])

    blocking = await server.list_tools()
    cancellable = await server.list_tools(
        tool_options={"echo": MCPToolOptions(flags=ToolFlag.CANCELLABLE, report_progress=True)}
    )

    assert blocking[0] is not cancellable[0]
    assert ToolFlag.CANCELLABLE not in blocking[0].info.flags
    assert ToolFlag.CANCELLABLE in cancellable[0].info.flags


@pytest.mark.asyncio
async def test_a_connect_nobody_waits_for_is_stopped_rather_than_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the caller of initialize() used to cancel the connect's own future, and the
    connect then failed setting a result on it."""
    import asyncio
    import contextlib

    from livekit.agents.llm import mcp as mcp_module

    handshaking = asyncio.Event()
    answer = asyncio.Event()

    class _SlowSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _SlowSession:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

        async def initialize(self) -> None:
            handshaking.set()
            # the server answers regardless of whether anyone is still waiting
            await asyncio.shield(answer.wait())

    class _Server(MCPServer):
        def __init__(self) -> None:
            super().__init__(client_session_timeout_seconds=5)

        @contextlib.asynccontextmanager
        async def client_streams(self):  # type: ignore[no-untyped-def,override]
            yield (None, None)

    monkeypatch.setattr(mcp_module, "ClientSession", _SlowSession)
    server = _Server()
    connecting = asyncio.create_task(server.initialize())
    await asyncio.wait_for(handshaking.wait(), timeout=5)
    connecting.cancel()
    answer.set()
    with contextlib.suppress(asyncio.CancelledError):
        await connecting

    client_task = server._client_task
    assert client_task is not None
    await asyncio.wait([client_task], timeout=5)
    assert client_task.cancelled() or client_task.exception() is None
    assert not server.initialized


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", ["first", "second"])
async def test_a_caller_giving_up_leaves_the_connect_to_the_one_still_waiting(
    monkeypatch: pytest.MonkeyPatch, cancelled: str
) -> None:
    """Two callers share one connect, so cancelling either must not fail the other."""
    import asyncio
    import contextlib

    from livekit.agents.llm import mcp as mcp_module

    handshaking = asyncio.Event()
    answer = asyncio.Event()

    class _SlowSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _SlowSession:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

        async def initialize(self) -> None:
            handshaking.set()
            await answer.wait()

    class _Server(MCPServer):
        def __init__(self) -> None:
            super().__init__(client_session_timeout_seconds=5)

        @contextlib.asynccontextmanager
        async def client_streams(self):  # type: ignore[no-untyped-def,override]
            yield (None, None)

    monkeypatch.setattr(mcp_module, "ClientSession", _SlowSession)
    server = _Server()
    first = asyncio.create_task(server.initialize())
    await asyncio.wait_for(handshaking.wait(), timeout=5)
    second = asyncio.create_task(server.initialize())
    await asyncio.sleep(0)

    gives_up, stays = (first, second) if cancelled == "first" else (second, first)
    gives_up.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gives_up

    answer.set()
    await asyncio.wait_for(stays, timeout=5)
    assert server.initialized
    await server.aclose()


@pytest.mark.asyncio
async def test_a_cancelled_call_propagates() -> None:
    """Cancelling a tool call raises CancelledError.

    MCP SDK 2 sends ``notifications/cancelled`` when the task awaiting
    ``ClientSession.call_tool`` is cancelled. This wrapper used to synthesize
    that notification from the private ``_request_id`` attribute, which v2 removed.
    """
    import asyncio

    calling = asyncio.Event()
    sent: list[Any] = []

    class _HangingClient(_FakeClient):
        async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
            calling.set()
            await asyncio.Event().wait()

        async def send_notification(self, notification: Any) -> None:
            sent.append(notification)

    server = _FakeMCPServer()
    server._client = _HangingClient(["echo"])  # type: ignore[assignment]
    tool = _build_mcp_tool(server, MCPToolOptions())

    call = asyncio.create_task(tool({}))
    await asyncio.wait_for(calling.wait(), timeout=5)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert sent == []


class _DyingSession:
    """Stands in for ``ClientSession``: connects fine, then the transport is gone.

    ``call_tool`` on tool ``"slow"`` waits for ``release`` before failing, so a test can keep a
    call in flight while another one discovers the death.
    """

    release: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _DyingSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
        if name == "slow":
            await type(self).release.wait()
        raise MCPError(mcp.types.CONNECTION_CLOSED, "Connection closed")

    async def list_tools(self) -> Any:
        raise MCPError(mcp.types.CONNECTION_CLOSED, "Connection closed")


class _ConnectableServer(MCPServer):
    def __init__(self) -> None:
        super().__init__(client_session_timeout_seconds=5)

    @contextlib.asynccontextmanager
    async def client_streams(self):  # type: ignore[no-untyped-def,override]
        yield (None, None)


def _tool_named(server: MCPServer, name: str):  # type: ignore[no-untyped-def]
    return server._make_function_tool(
        name=name,
        description="x",
        input_schema={"type": "object", "properties": {}},
        meta=None,
        options=_resolve_tool_options(MCPToolOptions()),
    )


@pytest.mark.asyncio
async def test_a_dead_connection_fails_the_call_clearly_and_is_no_longer_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the server process dies, nothing woke the parked connection task, so ``initialized``
    stayed True and every later call raised the SDK's bare ``MCPError``."""
    monkeypatch.setattr(mcp_module, "ClientSession", _DyingSession)
    server = _ConnectableServer()
    await server.initialize()
    assert server.initialized
    client_task = server._client_task
    assert client_task is not None

    with pytest.raises(ToolError, match="internal service is unavailable"):
        await _tool_named(server, "echo")({})

    assert not server.initialized
    await asyncio.wait_for(client_task, timeout=5)  # the connection task unwound

    # later calls keep failing the same clear way
    with pytest.raises(ToolError, match="internal service is unavailable"):
        await _tool_named(server, "echo")({})


@pytest.mark.asyncio
async def test_an_mcp_error_that_is_not_a_closed_connection_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MethodNotFoundSession(_DyingSession):
        async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
            raise MCPError(mcp.types.METHOD_NOT_FOUND, "Method not found")

    monkeypatch.setattr(mcp_module, "ClientSession", _MethodNotFoundSession)
    server = _ConnectableServer()
    await server.initialize()

    with pytest.raises(MCPError, match="Method not found"):
        await _tool_named(server, "echo")({})

    assert server.initialized
    await server.aclose()


@pytest.mark.asyncio
async def test_listing_tools_on_a_dead_connection_is_no_longer_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_module, "ClientSession", _DyingSession)
    server = _ConnectableServer()
    await server.initialize()

    with pytest.raises(RuntimeError, match="internal service is unavailable"):
        await server.list_tools()

    assert not server.initialized


@pytest.mark.asyncio
async def test_a_call_failing_after_the_connection_was_torn_down_does_not_break_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls are in flight when the server dies. The first to notice tears the connection
    down; the second fails later. It must not leave ``_closing_ev`` set, or the next
    ``initialize()`` would close the connection it just opened."""
    _DyingSession.release = asyncio.Event()
    monkeypatch.setattr(mcp_module, "ClientSession", _DyingSession)
    server = _ConnectableServer()
    await server.initialize()

    in_flight = asyncio.create_task(_tool_named(server, "slow")({}))
    await asyncio.sleep(0)

    with pytest.raises(ToolError):
        await _tool_named(server, "echo")({})
    first_task = server._client_task
    assert first_task is not None
    await asyncio.wait_for(first_task, timeout=5)

    _DyingSession.release.set()
    with pytest.raises(ToolError):
        await in_flight

    await server.initialize()
    await asyncio.sleep(0.1)
    assert server.initialized
    await server.aclose()
