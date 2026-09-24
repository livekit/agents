"""Test that LLM errors propagate through session.run() → RunResult,
including the full e2e path through SessionHost → RemoteSession.

A silent turn still leaves RunResult successful. The session can separately
emit a recoverable error event so applications can handle an empty LLM completion."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from livekit.agents import APIStatusError
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.remote_session import (
    RemoteSession,
    SessionHost,
    SessionTransport,
)
from livekit.protocol.agent_pb import agent_session as agent_pb

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


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
    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0))
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())

    await session.start(agent=agent)

    result = session.run(user_input="hello")
    with pytest.raises(APIStatusError):
        await asyncio.wait_for(result, timeout=10.0)

    await session.aclose()


@pytest.mark.asyncio
async def test_run_propagates_llm_error_with_retry():
    session = AgentSession(
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01)
        )
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())

    await session.start(agent=agent)

    result = session.run(user_input="hello")
    with pytest.raises(APIStatusError):
        await asyncio.wait_for(result, timeout=10.0)

    await session.aclose()


@pytest.mark.asyncio
async def test_run_input_error_e2e_through_remote_session():
    """Full e2e: RemoteSession → SessionHost → AgentSession with failing LLM.

    Verifies that an LLM 401 error propagates all the way back to the
    RemoteSession.run_input() caller as a RuntimeError.
    """
    host_transport, client_transport = PairedTransport.create_pair()

    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0))
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())

    host = SessionHost(host_transport)
    host.register_session(session)

    await session.start(agent=agent)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    with pytest.raises(RuntimeError, match="failed"):
        await client.run("order a big mac", timeout=10.0)

    await client.aclose()
    await host.aclose()
    await session.aclose()


@pytest.mark.asyncio
async def test_run_silent_turn_is_not_an_error():
    """A turn that produces no items is a silent agent, not a failed one.

    An LLM can return a completion with no text and no tool calls — a
    close-the-call tool whose output tells the model it has already said
    goodbye is the usual way to get there. Every item-add site in
    ``agent_activity`` is guarded on non-empty text, so such a turn reaches
    ``RunResult`` with zero events. That must come back as an empty item list
    rather than an error, so the caller can decide what the silence means.
    """
    host_transport, client_transport = PairedTransport.create_pair()

    session = AgentSession()
    # FakeLLM yields an empty completion for any input outside its response map
    agent = Agent(instructions="test agent", llm=FakeLLM())

    host = SessionHost(host_transport)
    host.register_session(session)

    await session.start(agent=agent)
    await host.start()

    client = RemoteSession(client_transport)
    await client.start()

    resp = await client.run("okay, thank you", timeout=10.0)
    assert list(resp.items) == []

    await client.aclose()
    await host.aclose()
    await session.aclose()


@pytest.mark.asyncio
async def test_usage_only_user_reply_emits_recoverable_error_and_warning(caplog):
    """An exhausted user reply with only usage must be visible to the application."""
    from livekit.agents.llm import ChatChunk, CompletionUsage
    from livekit.agents.voice.events import ErrorEvent

    class UsageOnlyAgent(Agent):
        async def llm_node(self, chat_ctx, tools, model_settings):
            yield ChatChunk(
                id="usage-only",
                usage=CompletionUsage(
                    completion_tokens=3,
                    prompt_tokens=5,
                    total_tokens=8,
                ),
            )

    session = AgentSession()
    llm_model = FakeLLM()
    agent = UsageOnlyAgent(instructions="test agent", llm=llm_model)
    errors: list[ErrorEvent] = []
    session.on("error", errors.append)

    try:
        await session.start(agent=agent)
        with caplog.at_level("WARNING", logger="livekit.agents"):
            result = await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)

        result.expect.no_more_events()
        assert len(errors) == 1
        assert errors[0].error.type == "llm_error"
        assert errors[0].error.recoverable is True
        assert errors[0].source is llm_model
        assert "empty" in str(errors[0].error.error).lower()
        assert any("empty" in record.message.lower() for record in caplog.records)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_tool_only_user_reply_does_not_emit_empty_completion_error():
    """A tool call is a valid LLM response even if it has no text."""
    from livekit.agents import function_tool
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLMResponse

    class ToolOnlyAgent(Agent):
        @function_tool
        async def complete_task(self) -> None:
            """Complete the user's task."""

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hello",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="complete_task", arguments="{}", call_id="call_1")
                ],
            )
        ]
    )
    session = AgentSession()
    agent = ToolOnlyAgent(instructions="test agent", llm=llm)
    errors = []
    session.on("error", errors.append)

    try:
        await session.start(agent=agent)
        result = await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)
        assert any(type(event).__name__ == "FunctionCallEvent" for event in result.events)
        assert errors == []
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["", "   "])
async def test_blank_custom_llm_reply_emits_recoverable_error(reply):
    """Blank custom node output cannot silently complete a scheduled user turn."""

    class BlankReplyAgent(Agent):
        async def llm_node(self, chat_ctx, tools, model_settings):
            return reply

    session = AgentSession()
    agent = BlankReplyAgent(instructions="test agent", llm=FakeLLM())
    errors = []
    session.on("error", errors.append)

    try:
        await session.start(agent=agent)
        await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)
        assert len(errors) == 1
        assert errors[0].error.recoverable is True
        assert "empty" in str(errors[0].error.error).lower()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_failed_llm_does_not_emit_empty_completion_error():
    """A provider failure is already surfaced and is not a completed blank response."""
    session = AgentSession(
        conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=0))
    )
    agent = Agent(instructions="test agent", llm=FailingLLM())
    errors = []
    session.on("error", errors.append)

    try:
        await session.start(agent=agent)
        with pytest.raises(APIStatusError):
            await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)
        assert all("empty" not in str(event.error.error).lower() for event in errors)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_discarded_unscheduled_usage_only_generation_stays_quiet():
    """Discarded preemptive work does not owe the user a reply."""
    from livekit.agents.llm import ChatChunk, ChatMessage, CompletionUsage

    class UsageOnlyAgent(Agent):
        def __init__(self):
            super().__init__(instructions="test agent", llm=FakeLLM())
            self.exhausted = asyncio.Event()

        async def llm_node(self, chat_ctx, tools, model_settings):
            yield ChatChunk(
                id="usage-only",
                usage=CompletionUsage(
                    completion_tokens=3,
                    prompt_tokens=5,
                    total_tokens=8,
                ),
            )
            self.exhausted.set()

    session = AgentSession()
    agent = UsageOnlyAgent()
    errors = []
    session.on("error", errors.append)

    try:
        await session.start(agent=agent)
        assert session._activity is not None
        handle = session._activity._generate_reply(
            user_message=ChatMessage(role="user", content=["hello"]),
            schedule_speech=False,
        )
        await asyncio.wait_for(agent.exhausted.wait(), timeout=10.0)
        handle._cancel()
        await asyncio.wait_for(handle, timeout=10.0)
        assert errors == []
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_empty_tool_followup_emits_recoverable_error():
    """A blank LLM generation after a tool reply still leaves the user waiting."""
    from livekit.agents import function_tool
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLMResponse

    class ToolAgent(Agent):
        @function_tool
        async def look_up(self) -> str:
            """Look up a value for the user."""
            return "lookup complete"

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hello",
                content="",
                ttft=0,
                duration=0.01,
                tool_calls=[FunctionToolCall(name="look_up", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(input="lookup complete", content="", ttft=0, duration=0.01),
        ]
    )
    session = AgentSession()
    errors = []
    session.on("error", errors.append)

    try:
        await session.start(agent=ToolAgent(instructions="test agent", llm=llm))
        result = await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)
        assert any(type(event).__name__ == "FunctionCallEvent" for event in result.events)
        assert len(errors) == 1
        assert errors[0].error.recoverable is True
        assert "empty" in str(errors[0].error.error).lower()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_scheduled_generate_reply_empty_completion_emits_recoverable_error():
    """A public generate_reply request should surface its empty completion."""
    from .fake_llm import FakeLLMResponse

    llm = FakeLLM(
        fake_responses=[FakeLLMResponse(input="greet", content="", ttft=0, duration=0.01)]
    )
    session = AgentSession(llm=llm)
    errors = []
    session.on("error", errors.append)

    try:
        await session.start(agent=Agent(instructions="test agent"))
        handle = session.generate_reply(instructions="greet")
        await asyncio.wait_for(handle, timeout=10.0)
        assert len(errors) == 1
        assert errors[0].error.recoverable is True
        assert "empty" in str(errors[0].error.error).lower()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_silent_turn_without_llm_generation_emits_no_empty_completion_error():
    """A turn with no LLM request is not an empty LLM completion."""
    session = AgentSession()
    errors = []
    session.on("error", errors.append)

    try:
        result = await asyncio.wait_for(
            session.start(agent=Agent(instructions="silent agent", llm=None), capture_run=True),
            timeout=10.0,
        )
        result.expect.next_event().is_agent_handoff()
        result.expect.no_more_events()
        assert errors == []
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_error_handler_can_schedule_recovery_reply():
    """An app can recover from the event before the empty turn finishes cleanup."""
    from .fake_llm import FakeLLMResponse

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="", ttft=0, duration=0.01),
            FakeLLMResponse(input="recover", content="Recovered", ttft=0, duration=0.01),
        ]
    )
    session = AgentSession(llm=llm)
    agent = Agent(instructions="test agent")
    errors = []
    recovery_handles = []

    def on_error(event):
        errors.append(event)
        if len(errors) == 1:
            recovery_handles.append(session.generate_reply(instructions="recover"))

    session.on("error", on_error)

    try:
        await session.start(agent=agent)
        await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)
        assert len(errors) == 1
        assert len(recovery_handles) == 1
        await asyncio.wait_for(recovery_handles[0], timeout=10.0)
        assert any(
            item.type == "message" and item.role == "assistant" and item.text_content == "Recovered"
            for item in agent.chat_ctx.items
        )
    finally:
        await session.aclose()
