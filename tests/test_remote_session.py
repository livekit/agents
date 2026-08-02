from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.metrics import AgentSessionUsage
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.remote_session import (
    RemoteSession,
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
        self.active_sends = 0
        self.max_concurrent_sends = 0
        self.send_count = 0

    @classmethod
    def create_pair(cls) -> tuple[PairedTransport, PairedTransport]:
        a, b = cls(), cls()
        a._peer = b
        b._peer = a
        return a, b

    async def start(self) -> None:
        pass

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        self.active_sends += 1
        self.max_concurrent_sends = max(self.max_concurrent_sends, self.active_sends)
        self.send_count += 1
        try:
            await asyncio.sleep(0)
            if self._peer and not self._peer._closed:
                self._peer._inbox.put_nowait(msg)
        finally:
            self.active_sends -= 1

    async def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=1.0)
        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
            raise StopAsyncIteration from None


class AdversarialTransport(SessionTransport):
    """Controllable transport for deterministic race reproduction.

    * Events are delivered immediately to the peer.
    * Responses can be paused until explicitly released.
    * Tracks concurrent ``send_message`` calls.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[agent_pb.AgentSessionMessage] = asyncio.Queue()
        self._peer: AdversarialTransport | None = None
        self._closed = False
        self.active_sends = 0
        self.max_concurrent_sends = 0
        self.sent_messages: list[agent_pb.AgentSessionMessage] = []
        self.response_ids: list[str] = []
        self.event_count = 0
        self._pause_responses = False
        self._response_gate = asyncio.Event()
        self._response_gate.set()
        self._on_response_send: asyncio.Event | None = None
        self._block_until_release: asyncio.Event | None = None

    @classmethod
    def create_pair(cls) -> tuple[AdversarialTransport, AdversarialTransport]:
        a, b = cls(), cls()
        a._peer = b
        b._peer = a
        return a, b

    def pause_responses(self) -> None:
        self._pause_responses = True
        self._response_gate.clear()

    def release_responses(self) -> None:
        self._pause_responses = False
        self._response_gate.set()

    async def start(self) -> None:
        pass

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        self.active_sends += 1
        self.max_concurrent_sends = max(self.max_concurrent_sends, self.active_sends)
        try:
            if self._closed:
                raise RuntimeError("adversarial transport is closed")

            is_response = msg.HasField("response")
            if is_response:
                self.response_ids.append(msg.response.request_id)
                if self._on_response_send is not None and not self._on_response_send.is_set():
                    self._on_response_send.set()
                if self._block_until_release is not None:
                    await self._block_until_release.wait()
                if self._pause_responses:
                    await self._response_gate.wait()
            elif msg.HasField("event"):
                self.event_count += 1

            # Yield so concurrent send attempts would overlap without serialization.
            await asyncio.sleep(0)
            if self._closed:
                raise RuntimeError("adversarial transport is closed")

            self.sent_messages.append(msg)
            if self._peer and not self._peer._closed:
                self._peer._inbox.put_nowait(msg)
        finally:
            self.active_sends -= 1

    async def close(self) -> None:
        self._closed = True
        self.release_responses()
        if self._block_until_release is not None:
            self._block_until_release.set()

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=2.0)
        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
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


def _fake_run_result(
    *,
    text: str = "hi there",
    msg_id: str = "m-1",
    delay: float = 0.0,
    emit_event: ConversationItemAddedEvent | None = None,
    host: SessionHost | None = None,
) -> MagicMock:
    class FakeRunResult:
        events = [MagicMock(item=ChatMessage(role="assistant", content=[text], id=msg_id))]

        def done(self) -> bool:
            return True

        def __await__(self):
            return self._await().__await__()

        async def _await(self) -> FakeRunResult:
            if emit_event is not None and host is not None:
                host._on_conversation_item_added(emit_event)
                # Let the event queue through the writer before the run completes.
                await asyncio.sleep(0)
            if delay:
                await asyncio.sleep(delay)
            return self

    return FakeRunResult()


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
async def test_wait_for_ready_retries_transport_send_failures():
    """Transient send failures must be retried until the peer becomes reachable."""
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    attempts = {"n": 0}
    original_send = client_transport.send_message

    async def flaky_send(msg: agent_pb.AgentSessionMessage) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("room session transport is closed")
        await original_send(msg)

    client_transport.send_message = flaky_send  # type: ignore[method-assign]
    await client.wait_for_ready(timeout=2.0, retry_interval=0.05)
    assert attempts["n"] >= 3

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_wait_for_ready_surfaces_transport_error_after_deadline():
    host_transport, client_transport = PairedTransport.create_pair()

    client = RemoteSession(client_transport)
    await client.start()

    async def always_fail(msg: agent_pb.AgentSessionMessage) -> None:
        raise RuntimeError("failed to send binary stream message: peer missing")

    client_transport.send_message = always_fail  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="wait_for_ready timed out") as ei:
        await client.wait_for_ready(timeout=0.15, retry_interval=0.05)

    assert ei.value.__cause__ is not None
    assert "peer missing" in str(ei.value.__cause__)

    await client.aclose()


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
    mock_session.run = MagicMock(return_value=_fake_run_result())

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.run("order a big mac", timeout=5.0)
    assert resp is not None
    assert len(resp.items) == 1

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_events_arrive_before_run_input_response():
    """Conversation events may be delivered before the matching SessionResponse."""
    host_transport, client_transport = AdversarialTransport.create_pair()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()

    host = SessionHost(host_transport)
    assistant_item = ChatMessage(role="assistant", content=["booked"], id="asst-1")
    emit_event = ConversationItemAddedEvent(item=assistant_item)
    mock_session.run = MagicMock(
        return_value=_fake_run_result(
            text="booked",
            msg_id="asst-1",
            emit_event=emit_event,
            host=host,
        )
    )
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    seen_events: list[agent_pb.AgentSessionEvent] = []
    response_seen = asyncio.Event()

    def _on_conversation_item_added(event: agent_pb.AgentSessionEvent) -> None:
        seen_events.append(event)
        # Event should be observable before run() returns.
        assert not response_seen.is_set()

    client.on("conversation_item_added", _on_conversation_item_added)

    resp = await client.run("book me", timeout=5.0)
    response_seen.set()

    assert len(resp.items) == 1
    assert resp.items[0].message.id == "asst-1"
    assert len(seen_events) == 1
    assert host_transport.max_concurrent_sends == 1

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_serialized_writes_never_overlap():
    host_transport, client_transport = AdversarialTransport.create_pair()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    # Flood events while also answering run_input.
    for i in range(20):
        host._on_conversation_item_added(
            ConversationItemAddedEvent(
                item=ChatMessage(role="assistant", content=[f"e{i}"], id=f"e-{i}")
            )
        )

    mock_session.run = MagicMock(return_value=_fake_run_result(text="done", msg_id="final"))

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.run("hello", timeout=5.0)
    assert len(resp.items) == 1
    assert host_transport.max_concurrent_sends == 1
    assert host_transport.event_count >= 20

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_concurrent_run_input_request_ids_are_unique_and_matched():
    host_transport, client_transport = PairedTransport.create_pair()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    call_count = 0

    def _run(user_input: str = "") -> MagicMock:
        nonlocal call_count
        call_count += 1
        n = call_count
        return _fake_run_result(text=f"reply-{n}", msg_id=f"m-{n}", delay=0.01)

    mock_session.run = MagicMock(side_effect=_run)

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    results = await asyncio.gather(
        client.run("one", timeout=5.0),
        client.run("two", timeout=5.0),
        client.run("three", timeout=5.0),
    )

    ids = [r.items[0].message.id for r in results]
    assert len(set(ids)) == 3
    assert client._pending_requests == {}
    assert host_transport.max_concurrent_sends == 1

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_shutdown_flushes_queued_run_input_response():
    """Response delayed behind a gate must still be delivered across shutdown.

    Reproduces the #6661 failure mode where aclose cancelled in-flight sends
    after conversation events had already been delivered. The old cancel-all
    path drops the ack (proven separately); this asserts the new drain path
    keeps it.
    """
    host_transport, client_transport = AdversarialTransport.create_pair()
    host_transport.pause_responses()
    host_transport._on_response_send = asyncio.Event()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()

    host = SessionHost(host_transport)
    assistant_item = ChatMessage(role="assistant", content=["confirmed"], id="asst-shutdown")
    mock_session.run = MagicMock(
        return_value=_fake_run_result(
            text="confirmed",
            msg_id="asst-shutdown",
            emit_event=ConversationItemAddedEvent(item=assistant_item),
            host=host,
        )
    )
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    seen_events: list[agent_pb.AgentSessionEvent] = []
    client.on("conversation_item_added", lambda ev: seen_events.append(ev))

    run_task = asyncio.create_task(client.run("confirm booking", timeout=5.0))

    # Wait until the host has begun sending the run_input response (paused).
    await asyncio.wait_for(host_transport._on_response_send.wait(), timeout=2.0)
    assert len(seen_events) == 1
    assert not run_task.done()

    # Begin shutdown while the response write is paused; then release it so the
    # drained writer can finish delivering the acknowledgement.
    close_task = asyncio.create_task(host.aclose())
    await asyncio.sleep(0)
    host_transport.release_responses()

    resp = await asyncio.wait_for(run_task, timeout=5.0)
    await asyncio.wait_for(close_task, timeout=5.0)

    assert len(resp.items) == 1
    assert resp.items[0].message.id == "asst-shutdown"
    assert len(host_transport.response_ids) >= 1

    await client.aclose()


@pytest.mark.asyncio
async def test_response_arriving_at_timeout_boundary_still_resolves():
    """If the ack lands in the same tick as the wait timeout, prefer success."""
    host_transport, client_transport = AdversarialTransport.create_pair()
    host_transport.pause_responses()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    mock_session.run = MagicMock(return_value=_fake_run_result(text="barely", msg_id="m-race"))

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    async def _release_near_timeout() -> None:
        await asyncio.sleep(0.08)
        host_transport.release_responses()

    release_task = asyncio.create_task(_release_near_timeout())
    resp = await client.run("hello", timeout=0.1)
    await release_task

    assert len(resp.items) == 1
    assert resp.items[0].message.id == "m-race"

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_session_host_restart_after_aclose():
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()
    await host.aclose()

    # Reuse the same SessionHost with a fresh transport; start() must recreate
    # the outbound channel that aclose() closed.
    host_transport, client_transport = PairedTransport.create_pair()
    host._transport = host_transport
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()
    await client.wait_for_ready(timeout=2.0)

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_pending_future_registered_before_send():
    """A response that arrives during send_message must still resolve."""
    host_transport, client_transport = PairedTransport.create_pair()

    registered = asyncio.Event()
    original_send = client_transport.send_message

    async def send_and_signal(msg: agent_pb.AgentSessionMessage) -> None:
        # By the time send is entered, RemoteSession must already have registered.
        # We observe this via the client's pending map after a scheduling point.
        await asyncio.sleep(0)
        assert any(client._pending_requests), "future must be registered before send"
        registered.set()
        await original_send(msg)

    client = RemoteSession(client_transport)
    await client.start()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client_transport.send_message = send_and_signal  # type: ignore[method-assign]
    await client.wait_for_ready(timeout=2.0)
    assert registered.is_set()

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_late_response_after_timeout_is_logged(caplog: pytest.LogCaptureFixture):
    host_transport, client_transport = AdversarialTransport.create_pair()
    host_transport.pause_responses()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    mock_session.run = MagicMock(return_value=_fake_run_result())

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    with caplog.at_level(logging.WARNING):
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await client.run("hello", timeout=0.05)

        assert client._pending_requests == {}
        host_transport.release_responses()
        # Allow the late response to arrive and be logged.
        await asyncio.sleep(0.1)

    assert any(
        "unknown or timed-out request" in r.message or "timed out" in r.message
        for r in caplog.records
    )

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_unknown_request_id_does_not_crash(caplog: pytest.LogCaptureFixture):
    host_transport, client_transport = PairedTransport.create_pair()

    host = SessionHost(host_transport)
    host.register_session(_make_mock_session())
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    orphan = agent_pb.AgentSessionMessage(
        response=agent_pb.SessionResponse(
            request_id="req_does_not_exist",
            pong=agent_pb.SessionResponse.Pong(),
        )
    )
    with caplog.at_level(logging.WARNING):
        await host_transport.send_message(orphan)
        await asyncio.sleep(0.1)

    assert any("unknown or timed-out request" in r.message for r in caplog.records)

    # Client remains usable.
    await client.wait_for_ready(timeout=2.0)

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_transport_closure_fails_pending_futures():
    host_transport, client_transport = AdversarialTransport.create_pair()
    host_transport.pause_responses()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    mock_session.run = MagicMock(return_value=_fake_run_result(delay=0.01))

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    run_task = asyncio.create_task(client.run("hello", timeout=5.0))
    await asyncio.sleep(0.05)

    await client.aclose()

    with pytest.raises(RuntimeError, match="remote session closed"):
        await run_task

    host_transport.release_responses()
    await host.aclose()


@pytest.mark.asyncio
async def test_empty_run_result_is_not_a_timeout():
    host_transport, client_transport = PairedTransport.create_pair()

    class EmptyRunResult:
        events: list[MagicMock] = []

        def done(self) -> bool:
            return True

        def __await__(self):
            return asyncio.sleep(0).__await__()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    mock_session.run = MagicMock(return_value=EmptyRunResult())

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    with pytest.raises(RuntimeError, match="no response items"):
        await client.run("hello", timeout=5.0)

    await client.aclose()
    await host.aclose()


@pytest.mark.asyncio
async def test_run_input_stress_with_scheduling_jitter():
    host_transport, client_transport = PairedTransport.create_pair()

    mock_session = _make_mock_session()
    mock_session.interrupt = AsyncMock()
    n = 0

    def _run(user_input: str = "") -> MagicMock:
        nonlocal n
        n += 1
        return _fake_run_result(text=f"r{n}", msg_id=f"id-{n}", delay=0.0)

    mock_session.run = MagicMock(side_effect=_run)

    host = SessionHost(host_transport)
    host.register_session(mock_session)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    async def _one_turn(i: int) -> str:
        # Interleave events with requests to stress the outbound writer.
        host._on_conversation_item_added(
            ConversationItemAddedEvent(
                item=ChatMessage(role="user", content=[f"u{i}"], id=f"u-{i}")
            )
        )
        await asyncio.sleep(0)
        resp = await client.run(f"turn-{i}", timeout=5.0)
        return resp.items[0].message.id

    ids = await asyncio.gather(*[_one_turn(i) for i in range(50)])
    assert len(ids) == 50
    assert len(set(ids)) == 50
    assert client._pending_requests == {}
    assert host_transport.max_concurrent_sends == 1

    await client.aclose()
    await host.aclose()
