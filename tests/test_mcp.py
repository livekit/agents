from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mcp")
import mcp.types as mcp_types
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import GetSessionIdCallback
from mcp.shared.message import SessionMessage

from livekit.agents.llm import AgentConfigUpdate, ChatContext, ToolContext, function_tool
from livekit.agents.llm.mcp import MCPServer, MCPTool, MCPToolset
from livekit.agents.voice.agent_activity import AgentActivity


def _make_mcp_tool(name: str) -> MCPTool:
    @function_tool(
        raw_schema={
            "name": name,
            "description": f"{name} test tool",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
    )
    async def _tool(raw_arguments: dict[str, Any]) -> str:
        return name

    return _tool


def _tool_names(toolset: MCPToolset) -> set[str]:
    return set(ToolContext([toolset]).function_tools)


async def _wait_for(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)


def _patch_reload_backoff_to_instant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the 1s/2s/4s reload backoff with zero waits for fast tests."""
    from livekit.agents.llm import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "_RELOAD_RETRY_DELAYS", (0.0, 0.0, 0.0))


def _make_test_activity(
    *,
    toolset: MCPToolset,
    rt_session: _FakeRealtimeSession | None,
    chat_ctx: ChatContext | None = None,
    session: SimpleNamespace | None = None,
    closed: bool = False,
) -> AgentActivity:
    """Construct an AgentActivity bypassing __init__ for unit tests.

    Initializes only the attributes touched by the toolset-refresh / publish-tools
    paths. Tests that exercise close paths (which touch stt/tts/vad/etc.) should
    extend the returned object themselves.
    """
    activity = object.__new__(AgentActivity)
    activity._session = session or SimpleNamespace(tools=[], llm=None)
    activity._agent = SimpleNamespace(
        id="agent",
        tools=[toolset],
        _tools=[toolset],
        _chat_ctx=chat_ctx if chat_ctx is not None else ChatContext.empty(),
        llm=object(),
    )
    activity._mcp_tools = []
    activity._rt_session = rt_session
    activity._closed = closed
    activity._toolset_changed_callbacks = {}
    activity._mcp_reload_failed_callbacks = {}
    activity._toolset_refresh_task = None
    activity._toolset_refresh_pending = False
    activity._tool_names_snapshot = set(ToolContext([toolset]).function_tools)
    activity._refresh_lock = asyncio.Lock()
    return activity


class _FakeMCPServer(MCPServer):
    def __init__(self, tools: list[str]) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self.tool_names = tools
        self.list_tools_calls = 0
        self.invalidate_cache_calls = 0
        self.fail_list_tools = False
        # When > 0, list_tools fails this many calls and then succeeds. Decremented
        # on each failing call. Useful for scripting transient-failure scenarios.
        self.fail_list_tools_remaining = 0
        self._fake_initialized = False
        self.block_list_tools: asyncio.Event | None = None
        self.list_tools_started: asyncio.Event | None = None
        self.list_tools_response_names: list[str] | None = None

    @property
    def initialized(self) -> bool:
        return self._fake_initialized

    async def initialize(self) -> None:
        self._fake_initialized = True

    def invalidate_cache(self) -> None:
        self.invalidate_cache_calls += 1
        super().invalidate_cache()

    async def list_tools(self) -> list[MCPTool]:
        self.list_tools_calls += 1
        if self.list_tools_started is not None:
            self.list_tools_started.set()
            self.list_tools_started = None
        if self.block_list_tools is not None:
            await self.block_list_tools.wait()
            self.block_list_tools = None
        if self.fail_list_tools:
            raise RuntimeError("list_tools failed")
        if self.fail_list_tools_remaining > 0:
            self.fail_list_tools_remaining -= 1
            raise RuntimeError("list_tools transient failure")
        tool_names = self.list_tools_response_names or self.tool_names
        self.list_tools_response_names = None
        return [_make_mcp_tool(name) for name in tool_names]

    async def aclose(self) -> None:
        self._fake_initialized = False

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
        raise NotImplementedError


class _FakeRealtimeSession:
    def __init__(self, on_close: Callable[[], Awaitable[None]] | None = None) -> None:
        self.tool_updates: list[set[str]] = []
        self.on_close = on_close
        self.closing = False
        self.updated_while_closing = False
        self.fail_update = False

    async def update_tools(self, tools: list[Any]) -> None:
        if self.fail_update:
            raise RuntimeError("update_tools failed")
        if self.closing:
            self.updated_while_closing = True
        self.tool_updates.append({tool.info.name for tool in tools if isinstance(tool, MCPTool)})

    async def aclose(self) -> None:
        self.closing = True
        if self.on_close is not None:
            await self.on_close()


@pytest.mark.asyncio
async def test_mcp_server_invalidates_cache_on_tool_list_changed_notification() -> None:
    server = _FakeMCPServer(["alpha"])
    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    server._cache_dirty = False
    server._add_tool_list_changed_callback(_callback)

    await server._handle_message(
        mcp_types.ServerNotification(root=mcp_types.ToolListChangedNotification())
    )

    assert server._cache_dirty is True
    assert server.invalidate_cache_calls == 1
    assert callback_calls == 1


@pytest.mark.asyncio
async def test_mcp_toolset_reloads_tools_after_tool_list_changed_notification() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)

    await toolset.setup()
    assert _tool_names(toolset) == {"alpha"}

    server.tool_names = ["beta", "gamma"]
    server._handle_tool_list_changed()

    await _wait_for(lambda: _tool_names(toolset) == {"beta", "gamma"})
    assert server.list_tools_calls >= 2

    await toolset.aclose()


@pytest.mark.asyncio
async def test_mcp_toolset_does_not_miss_notification_during_initial_list_tools() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    server.block_list_tools = asyncio.Event()
    server.list_tools_started = asyncio.Event()
    server.list_tools_response_names = ["alpha"]

    setup_task = asyncio.create_task(toolset.setup())
    await server.list_tools_started.wait()

    server.tool_names = ["beta"]
    server._handle_tool_list_changed()
    server.block_list_tools.set()

    await setup_task

    await _wait_for(lambda: _tool_names(toolset) == {"beta"})
    assert server.list_tools_calls >= 2

    await toolset.aclose()


@pytest.mark.asyncio
async def test_agent_activity_refreshes_realtime_tools_after_toolset_reload() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()

    rt_session = _FakeRealtimeSession()
    chat_ctx = ChatContext.empty()
    activity = _make_test_activity(toolset=toolset, rt_session=rt_session, chat_ctx=chat_ctx)
    activity._sync_toolset_change_subscriptions()

    server.tool_names = ["beta"]
    server._handle_tool_list_changed()

    await _wait_for(lambda: rt_session.tool_updates[-1:] == [{"beta"}])
    config_updates = [item for item in chat_ctx.items if isinstance(item, AgentConfigUpdate)]
    assert config_updates[-1].tools_added == ["beta"]
    assert config_updates[-1].tools_removed == ["alpha"]

    activity._clear_toolset_change_subscriptions()
    await toolset.aclose()


@pytest.mark.asyncio
async def test_agent_activity_unsubscribes_toolset_before_realtime_close() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()

    async def _notify_during_close() -> None:
        server.tool_names = ["beta"]
        server._handle_tool_list_changed()
        await asyncio.sleep(0.05)

    rt_session = _FakeRealtimeSession(on_close=_notify_during_close)
    activity = _make_test_activity(
        toolset=toolset,
        rt_session=rt_session,
        session=SimpleNamespace(tools=[], llm=None, stt=None, tts=None, vad=None),
    )
    activity._agent.stt = object()
    activity._agent.tts = object()
    activity._agent.vad = object()
    activity._realtime_spans = None
    activity._audio_recognition = None
    activity._interruption_detector = None
    activity._cancel_speech_pause_task = None

    async def _cancel_speech_pause(**kwargs: Any) -> None:
        return None

    activity._cancel_speech_pause = _cancel_speech_pause
    activity._lock = asyncio.Lock()
    activity._sync_toolset_change_subscriptions()

    async with activity._lock:
        await activity._close_session()

    assert rt_session.updated_while_closing is False
    assert rt_session.tool_updates == []
    assert activity._toolset_changed_callbacks == {}


@pytest.mark.asyncio
async def test_agent_activity_keeps_snapshot_when_realtime_tool_update_fails() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()

    rt_session = _FakeRealtimeSession()
    rt_session.fail_update = True
    activity = _make_test_activity(toolset=toolset, rt_session=rt_session)
    activity._sync_toolset_change_subscriptions()

    server.tool_names = ["beta"]
    server._handle_tool_list_changed()

    await _wait_for(
        lambda: activity._toolset_refresh_task is not None and activity._toolset_refresh_task.done()
    )
    assert activity._tool_names_snapshot == {"alpha"}
    assert [
        item for item in activity._agent._chat_ctx.items if isinstance(item, AgentConfigUpdate)
    ] == []

    activity._clear_toolset_change_subscriptions()
    await toolset.aclose()


@pytest.mark.asyncio
async def test_mcp_toolset_unregisters_tool_list_changed_callback_on_close() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)

    await toolset.setup()
    assert len(server._tool_list_changed_callbacks) == 1

    await toolset.aclose()

    assert server._tool_list_changed_callbacks == set()


@pytest.mark.asyncio
async def test_mcp_toolset_keeps_filter_after_tool_list_reload() -> None:
    server = _FakeMCPServer(["alpha", "beta"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)

    await toolset.setup()
    toolset.filter_tools(lambda tool: tool.info.name.startswith("a"))
    assert _tool_names(toolset) == {"alpha"}

    server.tool_names = ["alpha2", "beta2"]
    server._handle_tool_list_changed()

    await _wait_for(lambda: _tool_names(toolset) == {"alpha2"})

    await toolset.aclose()


@pytest.mark.asyncio
async def test_mcp_toolset_keeps_existing_tools_when_reload_fails(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reload_backoff_to_instant(monkeypatch)

    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)

    await toolset.setup()
    initial_calls = server.list_tools_calls
    assert _tool_names(toolset) == {"alpha"}

    server.fail_list_tools = True
    server._handle_tool_list_changed()

    # Reload retries 4 times total (initial attempt + 3 backoff retries) before
    # giving up. Each calls list_tools once.
    await _wait_for(lambda: server.list_tools_calls >= initial_calls + 4)
    assert _tool_names(toolset) == {"alpha"}
    assert "giving up reloading MCP tools" in caplog.text

    await toolset.aclose()


@pytest.mark.asyncio
async def test_mcp_toolset_retries_reload_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reload_backoff_to_instant(monkeypatch)

    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()
    initial_calls = server.list_tools_calls

    failure_callback_calls: list[BaseException] = []
    server._add_reload_failed_callback(failure_callback_calls.append)

    # Fail twice, then succeed on the third retry (total: 1 fail + 1 fail + success).
    server.fail_list_tools_remaining = 2
    server.tool_names = ["beta"]
    server._handle_tool_list_changed()

    await _wait_for(lambda: _tool_names(toolset) == {"beta"})
    # 2 transient failures + 1 success.
    assert server.list_tools_calls == initial_calls + 3
    assert failure_callback_calls == []

    await toolset.aclose()


@pytest.mark.asyncio
async def test_mcp_toolset_emits_terminal_error_after_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reload_backoff_to_instant(monkeypatch)

    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()

    failures: list[BaseException] = []
    server._add_reload_failed_callback(failures.append)

    server.fail_list_tools = True
    server._handle_tool_list_changed()

    await _wait_for(lambda: len(failures) == 1)
    assert isinstance(failures[0], RuntimeError)
    assert _tool_names(toolset) == {"alpha"}

    await toolset.aclose()


@pytest.mark.asyncio
async def test_agent_activity_emits_session_error_on_terminal_reload_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reload_backoff_to_instant(monkeypatch)

    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()

    rt_session = _FakeRealtimeSession()
    activity = _make_test_activity(toolset=toolset, rt_session=rt_session)

    emitted: list[tuple[str, Any]] = []

    def _emit(event_name: str, payload: Any) -> None:
        emitted.append((event_name, payload))

    activity._session.emit = _emit  # type: ignore[attr-defined]
    activity._sync_toolset_change_subscriptions()

    server.fail_list_tools = True
    server._handle_tool_list_changed()

    from livekit.agents.voice.events import ErrorEvent

    await _wait_for(
        lambda: any(name == "error" and isinstance(p, ErrorEvent) for name, p in emitted)
    )
    error_events = [p for name, p in emitted if name == "error"]
    assert len(error_events) == 1
    assert isinstance(error_events[0].error, RuntimeError)

    activity._clear_toolset_change_subscriptions()
    await toolset.aclose()


@pytest.mark.asyncio
async def test_publish_tools_change_serializes_under_concurrent_callers() -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)
    await toolset.setup()

    rt_session = _FakeRealtimeSession()

    block = asyncio.Event()
    started = asyncio.Event()
    enter_count = 0
    concurrent_observed = False
    original_update = rt_session.update_tools

    async def _slow_update(tools: list[Any]) -> None:
        nonlocal enter_count, concurrent_observed
        enter_count += 1
        if enter_count > 1:
            concurrent_observed = True
        started.set()
        await block.wait()
        await original_update(tools)
        enter_count -= 1

    rt_session.update_tools = _slow_update  # type: ignore[assignment]

    activity = _make_test_activity(toolset=toolset, rt_session=rt_session)
    activity._sync_toolset_change_subscriptions()

    # First publish blocks inside rt_session.update_tools.
    first = asyncio.create_task(activity._publish_tools_change())
    await started.wait()

    # Second publish should be queued on the lock, not enter rt_session.update_tools yet.
    second = asyncio.create_task(activity._publish_tools_change())
    await asyncio.sleep(0.05)
    assert enter_count == 1, "second publish must wait for the first to release the lock"

    block.set()
    await asyncio.gather(first, second)
    assert concurrent_observed is False

    activity._clear_toolset_change_subscriptions()
    await toolset.aclose()


@pytest.mark.asyncio
async def test_handle_message_ignores_unrelated_messages() -> None:
    """Exception-typed messages and non-tool-list notifications must not touch
    the cache or fire tool-list-changed callbacks. Matches the MCP SDK default
    handler behavior, which absorbs exceptions silently."""
    server = _FakeMCPServer(["alpha"])
    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    server._cache_dirty = False
    server._add_tool_list_changed_callback(_callback)

    # Exception-typed message: no raise, no cache invalidation, no callback.
    await server._handle_message(RuntimeError("transport hiccup"))
    assert server._cache_dirty is False
    assert callback_calls == 0
    assert server.invalidate_cache_calls == 0


@pytest.mark.asyncio
async def test_handle_message_lets_cancellation_propagate() -> None:
    """The checkpoint() in the Exception branch is a cooperative cancel point;
    a cancellation scheduled before the task runs must surface at that await
    instead of being swallowed by a no-await early return."""
    server = _FakeMCPServer(["alpha"])

    async def _runner() -> None:
        await server._handle_message(RuntimeError("transport hiccup"))

    task = asyncio.create_task(_runner())
    task.cancel()  # marked before the task gets a chance to run
    with pytest.raises(asyncio.CancelledError):
        await task
