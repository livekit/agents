from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

import livekit.agents.llm.mcp as lk_mcp
from livekit.agents import Agent, AgentSession

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD


class _FakeClientSession:
    instances: list[_FakeClientSession] = []
    initialize_errors: list[BaseException | None] = []
    initialize_started: asyncio.Event | None = None
    initialize_blocker: asyncio.Event | None = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.enter_task: asyncio.Task[object] | None = None
        self.exit_task: asyncio.Task[object] | None = None
        self.initialize_task: asyncio.Task[object] | None = None
        self._initialize_error = (
            type(self).initialize_errors.pop(0) if type(self).initialize_errors else None
        )
        type(self).instances.append(self)

    async def __aenter__(self) -> _FakeClientSession:
        self.enter_task = asyncio.current_task()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object,
    ) -> None:
        self.exit_task = asyncio.current_task()

    async def initialize(self) -> None:
        self.initialize_task = asyncio.current_task()
        if type(self).initialize_started is not None:
            type(self).initialize_started.set()
        if type(self).initialize_blocker is not None:
            await type(self).initialize_blocker.wait()
        if self._initialize_error is not None:
            raise self._initialize_error

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[])


class _FakeMCPServer(lk_mcp.MCPServer):
    def __init__(self) -> None:
        super().__init__(client_session_timeout_seconds=5)
        self.stream_enter_tasks: list[asyncio.Task[object] | None] = []
        self.stream_exit_tasks: list[asyncio.Task[object] | None] = []
        self.tools_listed = asyncio.Event()

    @asynccontextmanager
    async def client_streams(self):  # type: ignore[override]
        self.stream_enter_tasks.append(asyncio.current_task())
        try:
            yield object(), object()
        finally:
            self.stream_exit_tasks.append(asyncio.current_task())

    async def list_tools(self) -> list[lk_mcp.MCPTool]:
        tools = await super().list_tools()
        self.tools_listed.set()
        return tools


class _SimpleAgent(Agent):
    def __init__(self, toolset: lk_mcp.MCPToolset) -> None:
        super().__init__(instructions="You are a test agent.", tools=[toolset])


@pytest.fixture
def fake_client_session(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClientSession]:
    _FakeClientSession.instances = []
    _FakeClientSession.initialize_errors = []
    _FakeClientSession.initialize_started = None
    _FakeClientSession.initialize_blocker = None
    monkeypatch.setattr(lk_mcp, "ClientSession", _FakeClientSession)
    return _FakeClientSession


def _create_session() -> AgentSession:
    session = AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[], min_silence_duration=0.5, min_speech_duration=0.05),
        stt=FakeSTT(fake_user_speeches=[]),
        llm=FakeLLM(fake_responses=[]),
        tts=FakeTTS(fake_responses=[]),
        aec_warmup_duration=None,
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    return session


async def _cleanup(session: AgentSession) -> None:
    with contextlib.suppress(RuntimeError):
        await session.drain()
    await session.aclose()


def _assert_same_task_lifecycle(
    server: _FakeMCPServer, client_session: _FakeClientSession, *, index: int
) -> None:
    assert len(server.stream_enter_tasks) > index
    assert len(server.stream_exit_tasks) > index
    stream_enter_task = server.stream_enter_tasks[index]
    stream_exit_task = server.stream_exit_tasks[index]

    assert stream_enter_task is not None
    assert stream_exit_task is not None
    assert client_session.enter_task is not None
    assert client_session.exit_task is not None
    assert stream_enter_task is stream_exit_task
    assert client_session.enter_task is client_session.exit_task
    assert stream_enter_task is client_session.enter_task


async def test_mcp_lifecycle_uses_same_task_on_session_close(
    fake_client_session: type[_FakeClientSession],
) -> None:
    server = _FakeMCPServer()
    toolset = lk_mcp.MCPToolset(id="mcp-toolset", mcp_server=server)
    session = _create_session()

    await session.start(_SimpleAgent(toolset), record=False)
    await asyncio.wait_for(server.tools_listed.wait(), timeout=1)
    await _cleanup(session)

    assert len(fake_client_session.instances) == 1
    _assert_same_task_lifecycle(server, fake_client_session.instances[0], index=0)


async def test_mcp_lifecycle_uses_same_task_across_agent_handoff(
    fake_client_session: type[_FakeClientSession],
) -> None:
    first_server = _FakeMCPServer()
    second_server = _FakeMCPServer()
    first_agent = _SimpleAgent(
        lk_mcp.MCPToolset(id="mcp-toolset-a", mcp_server=first_server)
    )
    second_agent = _SimpleAgent(
        lk_mcp.MCPToolset(id="mcp-toolset-b", mcp_server=second_server)
    )
    session = _create_session()

    await session.start(first_agent, record=False)
    await asyncio.wait_for(first_server.tools_listed.wait(), timeout=1)

    session.update_agent(second_agent)
    await asyncio.wait_for(second_server.tools_listed.wait(), timeout=1)
    await _cleanup(session)

    assert len(fake_client_session.instances) == 2
    _assert_same_task_lifecycle(first_server, fake_client_session.instances[0], index=0)
    _assert_same_task_lifecycle(second_server, fake_client_session.instances[1], index=0)


async def test_mcp_toolset_can_retry_after_initialize_failure(
    fake_client_session: type[_FakeClientSession],
) -> None:
    fake_client_session.initialize_errors = [RuntimeError("mcp init failed"), None]

    server = _FakeMCPServer()
    toolset = lk_mcp.MCPToolset(id="mcp-toolset", mcp_server=server)

    with pytest.raises(RuntimeError, match="mcp init failed"):
        await toolset.setup()

    assert not server.initialized

    await toolset.setup()
    await toolset.aclose()

    assert len(fake_client_session.instances) == 2
    _assert_same_task_lifecycle(server, fake_client_session.instances[0], index=0)
    _assert_same_task_lifecycle(server, fake_client_session.instances[1], index=1)


async def test_mcp_toolset_cleans_up_on_setup_cancellation(
    fake_client_session: type[_FakeClientSession],
) -> None:
    fake_client_session.initialize_started = asyncio.Event()
    fake_client_session.initialize_blocker = asyncio.Event()

    server = _FakeMCPServer()
    toolset = lk_mcp.MCPToolset(id="mcp-toolset", mcp_server=server)

    setup_task = asyncio.create_task(toolset.setup())
    await asyncio.wait_for(fake_client_session.initialize_started.wait(), timeout=1)

    setup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await setup_task

    assert len(fake_client_session.instances) == 1
    assert server._lifecycle_task is None
    assert not server.initialized
    _assert_same_task_lifecycle(server, fake_client_session.instances[0], index=0)


async def test_mcp_toolset_does_not_hang_on_cancelled_initialize_failure(
    fake_client_session: type[_FakeClientSession],
) -> None:
    fake_client_session.initialize_errors = [asyncio.CancelledError()]

    server = _FakeMCPServer()
    toolset = lk_mcp.MCPToolset(id="mcp-toolset", mcp_server=server)

    setup_task = asyncio.create_task(toolset.setup())
    done, pending = await asyncio.wait({setup_task}, timeout=1)

    assert not pending
    assert setup_task in done
    assert setup_task.cancelled()
    assert len(fake_client_session.instances) == 1
    assert server._lifecycle_task is None
    assert not server.initialized
    _assert_same_task_lifecycle(server, fake_client_session.instances[0], index=0)
