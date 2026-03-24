from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.metrics import AgentSessionUsage
from livekit.agents.voice.remote_session import (
    RemoteSession,
    SessionHost,
    SessionTransport,
)
from livekit.protocol.agent_pb import agent_session as agent_pb


class PairedTransport(SessionTransport):
    """Two linked transports: what one sends, the other receives."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[agent_pb.AgentSessionMessage] = asyncio.Queue()
        self._peer: PairedTransport | None = None
        self._closed = False

    @classmethod
    def create_pair(cls) -> tuple[PairedTransport, PairedTransport]:
        a, b = cls(), cls()
        a._peer = b
        b._peer = a
        return a, b

    async def start(self) -> None:
        pass

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._peer and not self._peer._closed:
            self._peer._inbox.put_nowait(msg)

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise StopAsyncIteration from None


def _make_mock_session() -> MagicMock:
    session = MagicMock()
    session.on = MagicMock()
    session.off = MagicMock()

    history = MagicMock()
    history.items = [
        ChatMessage(role="user", content=["hello"], id="msg-1"),
        ChatMessage(role="assistant", content=["hi there"], id="msg-2"),
    ]
    session.history = history

    agent = MagicMock()
    agent.id = "agent-1"
    agent.instructions = "Be helpful"
    agent.tools = []
    agent.chat_ctx = ChatContext()
    session.current_agent = agent

    session.agent_state = "idle"
    session.user_state = "listening"
    session._started_at = 1000.0

    options = MagicMock()
    options.endpointing = MagicMock(__iter__=lambda s: iter([]))
    options.interruption = MagicMock(__iter__=lambda s: iter([]))
    options.max_tool_steps = 5
    options.user_away_timeout = 30
    options.preemptive_generation = False
    options.min_consecutive_speech_delay = 0.5
    options.use_tts_aligned_transcript = True
    options.ivr_detection = False
    session.options = options

    usage = AgentSessionUsage(model_usage=[])
    session.usage = usage

    return session


@pytest.mark.asyncio
async def test_ping():
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    await client.wait_for_ready(timeout=2.0)

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_get_chat_history():
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.get_chat_history()
    assert len(resp.items) == 2
    assert resp.items[0].message.id == "msg-1"
    assert resp.items[1].message.id == "msg-2"

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_get_agent_info():
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.get_agent_info()
    assert resp.id == "agent-1"
    assert resp.instructions == "Be helpful"

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_get_session_state():
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.get_session_state()
    assert resp.agent_id == "agent-1"
    assert resp.agent_state == agent_pb.AS_IDLE
    assert resp.user_state == agent_pb.US_LISTENING

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_run_input():
    host_transport, client_transport = PairedTransport.create_pair()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()

    class FakeRunResult:
        events: list = []
        def done(self): return True
        def __await__(self):
            return asyncio.sleep(0).__await__()

    mock_session.run = MagicMock(return_value=FakeRunResult())

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.run_input("order a big mac", timeout=5.0)
    assert resp is not None

    await client.aclose()
    await host.aclose()


