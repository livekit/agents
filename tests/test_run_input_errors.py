"""Test that LLM errors propagate through session.run() → RunResult,
including the full e2e path through SessionHost → RemoteSession."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from livekit.agents import APIStatusError
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.remote_session import (
    RemoteSession,
    SessionHost,
    SessionTransport,
)
from livekit.protocol.agent_pb import agent_session as agent_pb

from .fake_llm import FakeLLM


class FailingLLM(FakeLLM):
    """A FakeLLM that raises a retryable API error, going through the retry loop."""

    def chat(self, **kwargs):
        raise APIStatusError(
            "object cannot be found",
            status_code=401,
            retryable=True,
        )


class PairedTransport(SessionTransport):
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


@pytest.mark.asyncio
async def test_run_propagates_llm_error_no_retry():
    session = AgentSession(conn_options=APIConnectOptions(max_retry=0))
    agent = Agent(instructions="test agent", llm=FailingLLM())

    await session.start(agent=agent)

    result = session.run(user_input="hello")
    with pytest.raises(Exception):
        await asyncio.wait_for(result, timeout=10.0)

    await session.aclose()


@pytest.mark.asyncio
async def test_run_propagates_llm_error_with_retry():
    session = AgentSession(conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01))
    agent = Agent(instructions="test agent", llm=FailingLLM())

    await session.start(agent=agent)

    result = session.run(user_input="hello")
    with pytest.raises(Exception):
        await asyncio.wait_for(result, timeout=10.0)

    await session.aclose()


@pytest.mark.asyncio
async def test_run_input_error_e2e_through_remote_session():
    """Full e2e: RemoteSession → SessionHost → AgentSession with failing LLM.

    Verifies that an LLM 401 error propagates all the way back to the
    RemoteSession.run_input() caller as a RuntimeError.
    """
    host_transport, client_transport = PairedTransport.create_pair()

    session = AgentSession(conn_options=APIConnectOptions(max_retry=0))
    agent = Agent(instructions="test agent", llm=FailingLLM())

    host = SessionHost(host_transport)
    host.register_session(session)

    await session.start(agent=agent)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    with pytest.raises(RuntimeError, match="failed"):
        await client.run_input("order a big mac", timeout=10.0)

    await client.aclose()
    await host.aclose()
    await session.aclose()
