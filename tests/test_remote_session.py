from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.metrics import AgentSessionUsage
from livekit.agents.voice import remote_session as remote_session_module
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


@pytest.mark.asyncio
async def test_wait_for_ready_polls_through_a_transport_that_is_not_up():
    """`wait_for_ready` exists to poll while the transport is still connecting.

    That transport now reports its state instead of dropping the ping, so the
    error has to be retried rather than surfaced on the first attempt.
    """
    host_transport, client_transport = PairedTransport.create_pair()
    attempts = 0

    class _LateTransport(PairedTransport):
        async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("room session transport is closed")
            await client_transport.send_message(msg)

    late = _LateTransport()
    late._peer = host_transport._peer
    host_transport._peer = late

    mock_session = _make_mock_session()
    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(late)
    late._peer = host_transport
    host_transport._peer = late
    await client.start()

    await client.wait_for_ready(timeout=5.0, retry_interval=0.05)
    assert attempts >= 3

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_wait_for_ready_surfaces_the_transport_error_past_the_deadline():
    """Once the deadline passes, the transport error is the useful one."""

    class _DeadTransport(PairedTransport):
        async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
            raise RuntimeError("room session transport is closed")

    client = RemoteSession(_DeadTransport())
    await client.start()

    with pytest.raises(RuntimeError, match="transport is closed"):
        await client.wait_for_ready(timeout=0.2, retry_interval=0.05)

    await client.aclose()


class _FakeReader:
    """Stands in for an rtc.ByteStreamReader over one message."""

    def __init__(self, payload: bytes) -> None:
        # split so the reader yields mid-message, the window a per-stream task
        # used to interleave in
        self._parts = [payload[:1], payload[1:]] if len(payload) > 1 else [payload]

    async def __aiter__(self):
        for part in self._parts:
            await asyncio.sleep(0)  # a real reader yields between chunks
            yield part


@pytest.mark.asyncio
async def test_inbound_messages_are_read_without_a_task_each_and_stay_ordered():
    """Reading spawns one loop, not one task per message, and preserves order."""
    transport = RoomSessionTransport(_fake_room())
    await transport.start()

    before = len(asyncio.all_tasks())

    for i in range(10):
        msg = agent_pb.AgentSessionMessage(
            event=agent_pb.AgentSessionEvent(
                user_input_transcribed=agent_pb.AgentSessionEvent.UserInputTranscribed(
                    transcript=str(i)
                )
            )
        )
        transport._on_byte_stream(_FakeReader(msg.SerializeToString()), "remote")

    # queueing 10 messages must not create 10 tasks
    assert len(asyncio.all_tasks()) <= before

    received = []
    for _ in range(10):
        received.append(await asyncio.wait_for(transport.__anext__(), timeout=2.0))

    assert [m.event.user_input_transcribed.transcript for m in received] == [
        str(i) for i in range(10)
    ]

    await transport.close()


@pytest.mark.asyncio
async def test_cancelled_requests_are_logged_by_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A request that outlives the grace period is reported by what it was.

    The type is what can be acted on; the request id would only mean something
    alongside a client log that a co-shutting-down client never emits.
    """
    monkeypatch.setattr(remote_session_module, "_SHUTDOWN_DRAIN_TIMEOUT", 0.05)

    host_transport, client_transport = PairedTransport.create_pair()

    never_finishes = asyncio.Event()
    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    mock_session.run = MagicMock(return_value=_SlowRunResult(never_finishes))

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    run_task = asyncio.ensure_future(client.run("order a big mac", timeout=2.0))
    await asyncio.sleep(0.05)  # let the handler reach the pending run

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await host.aclose()

    records = [r for r in caplog.records if "cancelled in-flight" in r.message]
    assert records, "abandoning a request at shutdown must be reported"
    assert records[0].request_types == ["run_input"]

    never_finishes.set()
    run_task.cancel()
    await client.aclose()
