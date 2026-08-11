from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.metrics import AgentSessionUsage
from livekit.agents.voice.remote_session import (
    RemoteSession,
    RoomSessionTransport,
    SessionHost,
    SessionTransport,
)
from livekit.protocol.agent_pb import agent_session as agent_pb

pytestmark = pytest.mark.unit


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
    options.preemptive_generation = MagicMock(__iter__=lambda s: iter([]))
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
        events = [MagicMock(item=ChatMessage(role="assistant", content=["hi there"], id="m-1"))]

        def done(self):
            return True

        def __await__(self):
            return asyncio.sleep(0).__await__()

    mock_session.run = MagicMock(return_value=FakeRunResult())

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.run("order a big mac", timeout=5.0)
    assert resp is not None

    await client.aclose()
    await host.aclose()


class _SlowRunResult:
    """A run that completes only once the test releases it."""

    def __init__(self, release: asyncio.Event) -> None:
        self.events = [MagicMock(item=ChatMessage(role="assistant", content=["done"], id="m-slow"))]
        self._release = release

    def done(self) -> bool:
        return True

    def __await__(self):
        return self._release.wait().__await__()


@pytest.mark.asyncio
async def test_aclose_flushes_the_response_of_an_inflight_request():
    """Shutdown must not drop the answer to a request that already did its work.

    `aclose` used to cancel in-flight handlers outright, so a run that had just
    finished lost its response and the caller waited out its whole timeout for
    an answer the session already had.
    """
    host_transport, client_transport = PairedTransport.create_pair()

    release = asyncio.Event()
    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    mock_session.run = MagicMock(return_value=_SlowRunResult(release))

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    run_task = asyncio.ensure_future(client.run("order a big mac", timeout=1.0))
    await asyncio.sleep(0.05)  # let the handler reach the pending run

    async def _finish_during_shutdown() -> None:
        await asyncio.sleep(0.05)
        release.set()

    # the run completes while aclose is already draining: cancelling the handler
    # at that moment is what loses the response
    finisher = asyncio.ensure_future(_finish_during_shutdown())
    await host.aclose()
    await finisher

    resp = await asyncio.wait_for(run_task, timeout=2.0)
    assert [i.message.content[0].text for i in resp.items] == ["done"]

    await client.aclose()


def _fake_room(*, connected: bool = True, stream_error: Exception | None = None) -> MagicMock:
    room = MagicMock()
    room.isconnected = MagicMock(return_value=connected)

    async def _stream_bytes(**kwargs):
        if stream_error is not None:
            raise stream_error
        writer = MagicMock()
        writer.write = AsyncMock()
        writer.aclose = AsyncMock()
        return writer

    room.local_participant.stream_bytes = _stream_bytes
    return room


@pytest.mark.asyncio
async def test_room_transport_raises_instead_of_dropping_the_message():
    """A transport that cannot deliver must say so, not drop the message.

    A silently dropped response is indistinguishable from a hung agent: the
    caller blocks until its timeout with nothing explaining why.
    """
    # the send itself fails
    transport = RoomSessionTransport(_fake_room(stream_error=RuntimeError("stream refused")))
    with pytest.raises(RuntimeError, match="failed to send binary stream message"):
        await transport.send_message(agent_pb.AgentSessionMessage())

    # the room is gone
    transport = RoomSessionTransport(_fake_room(connected=False))
    with pytest.raises(RuntimeError, match="closed"):
        await transport.send_message(agent_pb.AgentSessionMessage())

    # and a healthy room still sends
    transport = RoomSessionTransport(_fake_room())
    await transport.send_message(agent_pb.AgentSessionMessage())


@pytest.mark.asyncio
async def test_room_transport_serializes_concurrent_sends():
    """Concurrent writers must not interleave on the shared topic."""
    overlapping = False
    in_flight = 0

    async def _stream_bytes(**kwargs):
        nonlocal overlapping, in_flight
        in_flight += 1
        if in_flight > 1:
            overlapping = True
        await asyncio.sleep(0.01)  # a real send yields several times
        in_flight -= 1
        writer = MagicMock()
        writer.write = AsyncMock()
        writer.aclose = AsyncMock()
        return writer

    room = _fake_room()
    room.local_participant.stream_bytes = _stream_bytes
    transport = RoomSessionTransport(room)

    await asyncio.gather(
        *(transport.send_message(agent_pb.AgentSessionMessage()) for _ in range(8))
    )
    assert not overlapping


class _FailingTransport(PairedTransport):
    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        raise RuntimeError("transport is down")


@pytest.mark.asyncio
async def test_event_send_failure_does_not_escape_as_task_exception(
    caplog: pytest.LogCaptureFixture,
):
    """Events stay fire-and-forget even though the transport now raises."""
    host = SessionHost(_FailingTransport())
    host.register_session(_make_mock_session())
    await host.start()

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        host._send_event(agent_pb.AgentSessionEvent())
        await asyncio.sleep(0.05)

    assert [r for r in caplog.records if "failed to send session event" in r.message]

    await host.aclose()


@pytest.mark.asyncio
async def test_events_are_written_by_one_task_in_emission_order():
    """Events queue to a single writer rather than a task each."""
    sent: list[str] = []

    class _RecordingTransport(PairedTransport):
        async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
            await asyncio.sleep(0.005)  # a real send yields
            sent.append(msg.event.user_input_transcribed.transcript)

    host = SessionHost(_RecordingTransport())
    host.register_session(_make_mock_session())
    await host.start()

    before = len(host._tasks.tasks)
    for i in range(10):
        host._send_event(
            agent_pb.AgentSessionEvent(
                user_input_transcribed=agent_pb.AgentSessionEvent.UserInputTranscribed(
                    transcript=str(i)
                )
            )
        )

    # emitting events must not spawn tasks
    assert len(host._tasks.tasks) == before

    await host.aclose()  # closing drains what is still queued
    assert sent == [str(i) for i in range(10)]
