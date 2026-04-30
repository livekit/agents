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


class _FakeMCPServer(MCPServer):
    def __init__(self, tools: list[str]) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self.tool_names = tools
        self.list_tools_calls = 0
        self.invalidate_cache_calls = 0
        self.fail_list_tools = False
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
    activity = object.__new__(AgentActivity)
    activity._session = SimpleNamespace(tools=[], llm=None)
    activity._agent = SimpleNamespace(
        id="agent",
        tools=[toolset],
        _tools=[toolset],
        _chat_ctx=chat_ctx,
        llm=object(),
    )
    activity._mcp_tools = []
    activity._rt_session = rt_session
    activity._closed = False
    activity._toolset_changed_callbacks = {}
    activity._toolset_refresh_task = None
    activity._toolset_refresh_pending = False
    activity._tool_names_snapshot = {"alpha"}
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
    activity = object.__new__(AgentActivity)
    activity._session = SimpleNamespace(tools=[], llm=None, stt=None, tts=None, vad=None)
    activity._agent = SimpleNamespace(
        id="agent",
        tools=[toolset],
        _tools=[toolset],
        _chat_ctx=ChatContext.empty(),
        llm=object(),
        stt=object(),
        tts=object(),
        vad=object(),
    )
    activity._mcp_tools = []
    activity._rt_session = rt_session
    activity._realtime_spans = None
    activity._audio_recognition = None
    activity._interruption_detector = None
    activity._closed = False
    activity._toolset_changed_callbacks = {}
    activity._toolset_refresh_task = None
    activity._toolset_refresh_pending = False
    activity._tool_names_snapshot = {"alpha"}
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
    activity = object.__new__(AgentActivity)
    activity._session = SimpleNamespace(tools=[], llm=None)
    activity._agent = SimpleNamespace(
        id="agent",
        tools=[toolset],
        _tools=[toolset],
        _chat_ctx=ChatContext.empty(),
        llm=object(),
    )
    activity._mcp_tools = []
    activity._rt_session = rt_session
    activity._closed = False
    activity._toolset_changed_callbacks = {}
    activity._toolset_refresh_task = None
    activity._toolset_refresh_pending = False
    activity._tool_names_snapshot = {"alpha"}
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
) -> None:
    server = _FakeMCPServer(["alpha"])
    toolset = MCPToolset(id="test_mcp", mcp_server=server)

    await toolset.setup()
    assert _tool_names(toolset) == {"alpha"}

    server.fail_list_tools = True
    server._handle_tool_list_changed()

    await _wait_for(lambda: server.list_tools_calls >= 2)
    assert _tool_names(toolset) == {"alpha"}
    assert "failed to reload MCP tools" in caplog.text

    await toolset.aclose()
