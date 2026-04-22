from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from livekit.agents.llm import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import (
    AgentSessionUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    STTModelUsage,
    TTSModelUsage,
)
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    SessionUsageUpdatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.voice.remote_session import (
    RoomSessionTransport,
    SessionHost,
    SessionTransport,
    _chat_item_to_proto,
    _metrics_to_proto,
    _session_usage_to_proto,
)
from livekit.protocol.agent_pb import agent_session as agent_pb

# ---------------------------------------------------------------------------
# In-memory transport for testing
# ---------------------------------------------------------------------------


class InMemoryTransport(SessionTransport):
    """A simple in-memory transport that captures sent messages."""

    def __init__(self) -> None:
        self.sent: list[agent_pb.AgentSessionMessage] = []
        self._inbound: asyncio.Queue[agent_pb.AgentSessionMessage] = asyncio.Queue()
        self._closed = False

    async def start(self) -> None:
        pass

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._closed:
            return
        self.sent.append(msg)

    async def close(self) -> None:
        self._closed = True

    def inject(self, msg: agent_pb.AgentSessionMessage) -> None:
        self._inbound.put_nowait(msg)

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._inbound.get(), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise StopAsyncIteration from None


# ---------------------------------------------------------------------------
# Proto conversion helpers
# ---------------------------------------------------------------------------


class TestChatItemToProto:
    def test_chat_message(self) -> None:
        msg = ChatMessage(role="user", content=["hello"], id="msg-1")
        pb_item = _chat_item_to_proto(msg)
        assert pb_item.HasField("message")
        assert pb_item.message.id == "msg-1"
        assert pb_item.message.role == agent_pb.USER
        assert len(pb_item.message.content) == 1
        assert pb_item.message.content[0].text == "hello"

    def test_chat_message_assistant(self) -> None:
        msg = ChatMessage(role="assistant", content=["hi there"], id="msg-2")
        pb_item = _chat_item_to_proto(msg)
        assert pb_item.message.role == agent_pb.ASSISTANT

    def test_chat_message_developer(self) -> None:
        msg = ChatMessage(role="developer", content=["system prompt"], id="msg-3")
        pb_item = _chat_item_to_proto(msg)
        assert pb_item.message.role == agent_pb.DEVELOPER

    def test_function_call(self) -> None:
        fc = FunctionCall(
            id="fc-1",
            call_id="call-1",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        pb_item = _chat_item_to_proto(fc)
        assert pb_item.HasField("function_call")
        assert pb_item.function_call.name == "get_weather"
        assert pb_item.function_call.call_id == "call-1"
        assert pb_item.function_call.arguments == '{"location": "NYC"}'

    def test_function_call_output(self) -> None:
        fco = FunctionCallOutput(
            call_id="call-1",
            output="sunny, 72F",
            is_error=False,
        )
        pb_item = _chat_item_to_proto(fco)
        assert pb_item.HasField("function_call_output")
        assert pb_item.function_call_output.call_id == "call-1"
        assert pb_item.function_call_output.output == "sunny, 72F"
        assert pb_item.function_call_output.is_error is False

    def test_function_call_output_error(self) -> None:
        fco = FunctionCallOutput(
            call_id="call-2",
            output="not found",
            is_error=True,
        )
        pb_item = _chat_item_to_proto(fco)
        assert pb_item.function_call_output.is_error is True


class TestMetricsToProto:
    def test_empty(self) -> None:
        pb = _metrics_to_proto(None)
        assert isinstance(pb, agent_pb.MetricsReport)

    def test_with_fields(self) -> None:
        metrics = {
            "transcription_delay": 0.1,
            "end_of_turn_delay": 0.2,
            "llm_node_ttft": 0.3,
            "tts_node_ttfb": 0.4,
            "e2e_latency": 0.5,
        }
        pb = _metrics_to_proto(metrics)
        assert pb.transcription_delay == pytest.approx(0.1)
        assert pb.end_of_turn_delay == pytest.approx(0.2)
        assert pb.llm_node_ttft == pytest.approx(0.3)
        assert pb.tts_node_ttfb == pytest.approx(0.4)
        assert pb.e2e_latency == pytest.approx(0.5)

    def test_partial_fields(self) -> None:
        metrics = {"transcription_delay": 0.42}
        pb = _metrics_to_proto(metrics)
        assert pb.transcription_delay == pytest.approx(0.42)


class TestSessionUsageToProto:
    def test_llm_usage(self) -> None:
        usage = AgentSessionUsage(
            model_usage=[
                LLMModelUsage(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                )
            ]
        )
        pb = _session_usage_to_proto(usage)
        assert len(pb.model_usage) == 1
        assert pb.model_usage[0].HasField("llm")
        assert pb.model_usage[0].llm.provider == "openai"
        assert pb.model_usage[0].llm.model == "gpt-4"
        assert pb.model_usage[0].llm.input_tokens == 100
        assert pb.model_usage[0].llm.output_tokens == 50

    def test_tts_usage(self) -> None:
        usage = AgentSessionUsage(
            model_usage=[
                TTSModelUsage(
                    provider="elevenlabs",
                    model="eleven_turbo_v2",
                    characters_count=500,
                    audio_duration=3.5,
                )
            ]
        )
        pb = _session_usage_to_proto(usage)
        assert len(pb.model_usage) == 1
        assert pb.model_usage[0].HasField("tts")
        assert pb.model_usage[0].tts.provider == "elevenlabs"
        assert pb.model_usage[0].tts.characters_count == 500

    def test_stt_usage(self) -> None:
        usage = AgentSessionUsage(
            model_usage=[
                STTModelUsage(
                    provider="deepgram",
                    model="nova-2",
                    audio_duration=10.0,
                )
            ]
        )
        pb = _session_usage_to_proto(usage)
        assert len(pb.model_usage) == 1
        assert pb.model_usage[0].HasField("stt")
        assert pb.model_usage[0].stt.audio_duration == pytest.approx(10.0)

    def test_interruption_usage(self) -> None:
        usage = AgentSessionUsage(
            model_usage=[
                InterruptionModelUsage(
                    provider="livekit",
                    model="turn-detector",
                    total_requests=42,
                )
            ]
        )
        pb = _session_usage_to_proto(usage)
        assert len(pb.model_usage) == 1
        assert pb.model_usage[0].HasField("interruption")
        assert pb.model_usage[0].interruption.total_requests == 42

    def test_mixed_usage(self) -> None:
        usage = AgentSessionUsage(
            model_usage=[
                LLMModelUsage(provider="openai", model="gpt-4"),
                TTSModelUsage(provider="elevenlabs", model="v2"),
                STTModelUsage(provider="deepgram", model="nova"),
            ]
        )
        pb = _session_usage_to_proto(usage)
        assert len(pb.model_usage) == 3


# ---------------------------------------------------------------------------
# RoomSessionTransport
# ---------------------------------------------------------------------------


class TestRoomSessionTransport:
    def test_remote_identity_property(self) -> None:
        room = MagicMock()
        transport = RoomSessionTransport(room, remote_identity="user-1")
        assert transport.remote_identity == "user-1"

        transport.remote_identity = "user-2"
        assert transport.remote_identity == "user-2"

    def test_remote_identity_none(self) -> None:
        room = MagicMock()
        transport = RoomSessionTransport(room)
        assert transport.remote_identity is None


# ---------------------------------------------------------------------------
# SessionHost event forwarding
# ---------------------------------------------------------------------------


class TestSessionHostEvents:
    @pytest.fixture
    def transport(self) -> InMemoryTransport:
        return InMemoryTransport()

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        session = MagicMock()
        session.on = MagicMock()
        session.off = MagicMock()
        return session

    def test_register_session(self, transport: InMemoryTransport, mock_session: MagicMock) -> None:
        host = SessionHost(transport)
        host.register_session(mock_session)
        assert mock_session.on.call_count == 8

    @pytest.mark.asyncio
    async def test_agent_state_changed(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        await host.start()

        event = AgentStateChangedEvent(
            type="agent_state_changed",
            old_state="idle",
            new_state="listening",
            created_at=1000.0,
        )
        host._on_agent_state_changed(event)
        await asyncio.sleep(0.1)

        assert len(transport.sent) == 1
        msg = transport.sent[0]
        assert msg.HasField("event")
        assert msg.event.HasField("agent_state_changed")
        assert msg.event.agent_state_changed.old_state == agent_pb.AS_IDLE
        assert msg.event.agent_state_changed.new_state == agent_pb.AS_LISTENING
        assert msg.event.HasField("created_at")

        await host.aclose()

    @pytest.mark.asyncio
    async def test_user_state_changed(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        await host.start()

        event = UserStateChangedEvent(
            type="user_state_changed",
            old_state="listening",
            new_state="speaking",
            created_at=1000.0,
        )
        host._on_user_state_changed(event)
        await asyncio.sleep(0.1)

        assert len(transport.sent) == 1
        msg = transport.sent[0]
        assert msg.event.user_state_changed.old_state == agent_pb.US_LISTENING
        assert msg.event.user_state_changed.new_state == agent_pb.US_SPEAKING

        await host.aclose()

    @pytest.mark.asyncio
    async def test_user_input_transcribed(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        await host.start()

        event = UserInputTranscribedEvent(
            type="user_input_transcribed",
            transcript="hello world",
            is_final=True,
            created_at=1000.0,
        )
        host._on_user_input_transcribed(event)
        await asyncio.sleep(0.1)

        assert len(transport.sent) == 1
        msg = transport.sent[0]
        assert msg.event.user_input_transcribed.transcript == "hello world"
        assert msg.event.user_input_transcribed.is_final is True

        await host.aclose()

    @pytest.mark.asyncio
    async def test_conversation_item_added(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        await host.start()

        chat_msg = ChatMessage(role="user", content=["hello"], id="msg-1")
        event = ConversationItemAddedEvent(
            type="conversation_item_added",
            item=chat_msg,
            created_at=1000.0,
        )
        host._on_conversation_item_added(event)
        await asyncio.sleep(0.1)

        assert len(transport.sent) == 1
        msg = transport.sent[0]
        assert msg.event.conversation_item_added.item.message.id == "msg-1"

        await host.aclose()

    @pytest.mark.asyncio
    async def test_error_event(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        await host.start()

        event = ErrorEvent(
            type="error",
            error=RuntimeError("something went wrong"),
            source="llm",
            created_at=1000.0,
        )
        host._on_error(event)
        await asyncio.sleep(0.1)

        assert len(transport.sent) == 1
        msg = transport.sent[0]
        assert msg.event.error.message == "something went wrong"

        await host.aclose()

    @pytest.mark.asyncio
    async def test_session_usage_updated(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        await host.start()

        usage = AgentSessionUsage(
            model_usage=[
                LLMModelUsage(provider="openai", model="gpt-4", input_tokens=100, output_tokens=50)
            ]
        )
        event = SessionUsageUpdatedEvent(
            type="session_usage_updated",
            usage=usage,
            created_at=1000.0,
        )
        host._on_session_usage_updated(event)
        await asyncio.sleep(0.1)

        assert len(transport.sent) == 1
        msg = transport.sent[0]
        assert msg.event.session_usage_updated.usage.model_usage[0].llm.provider == "openai"

        await host.aclose()


# ---------------------------------------------------------------------------
# SessionHost request handling
# ---------------------------------------------------------------------------


class TestSessionHostRequests:
    @pytest.fixture
    def transport(self) -> InMemoryTransport:
        return InMemoryTransport()

    def _make_mock_session(self) -> MagicMock:
        session = MagicMock()
        session.on = MagicMock()
        session.off = MagicMock()

        history = MagicMock()
        history.items = [ChatMessage(role="user", content=["hi"], id="h-1")]
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
        options.preemptive_generation = {"enabled": False}
        options.min_consecutive_speech_delay = 0.5
        options.use_tts_aligned_transcript = True
        options.ivr_detection = False
        session.options = options

        usage = AgentSessionUsage(model_usage=[])
        session.usage = usage

        return session

    @pytest.mark.asyncio
    async def test_ping_pong(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        host.register_session(session)
        await host.start()

        req = agent_pb.SessionRequest(
            request_id="req-1",
            ping=agent_pb.SessionRequest.Ping(),
        )
        await host._handle_request(req)

        assert len(transport.sent) == 1
        resp = transport.sent[0].response
        assert resp.request_id == "req-1"
        assert resp.HasField("pong")

        await host.aclose()

    @pytest.mark.asyncio
    async def test_get_chat_history(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        host.register_session(session)
        await host.start()

        req = agent_pb.SessionRequest(
            request_id="req-2",
            get_chat_history=agent_pb.SessionRequest.GetChatHistory(),
        )
        await host._handle_request(req)

        assert len(transport.sent) == 1
        resp = transport.sent[0].response
        assert resp.request_id == "req-2"
        assert resp.HasField("get_chat_history")
        assert len(resp.get_chat_history.items) == 1

        await host.aclose()

    @pytest.mark.asyncio
    async def test_get_agent_info(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        host.register_session(session)
        await host.start()

        req = agent_pb.SessionRequest(
            request_id="req-3",
            get_agent_info=agent_pb.SessionRequest.GetAgentInfo(),
        )
        await host._handle_request(req)

        assert len(transport.sent) == 1
        resp = transport.sent[0].response
        assert resp.request_id == "req-3"
        assert resp.HasField("get_agent_info")
        assert resp.get_agent_info.id == "agent-1"
        assert resp.get_agent_info.instructions == "Be helpful"

        await host.aclose()

    @pytest.mark.asyncio
    async def test_get_session_state(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        host.register_session(session)
        await host.start()

        req = agent_pb.SessionRequest(
            request_id="req-4",
            get_session_state=agent_pb.SessionRequest.GetSessionState(),
        )
        await host._handle_request(req)

        assert len(transport.sent) == 1
        resp = transport.sent[0].response
        assert resp.request_id == "req-4"
        assert resp.HasField("get_session_state")
        assert resp.get_session_state.agent_id == "agent-1"
        assert resp.get_session_state.agent_state == agent_pb.AS_IDLE
        assert resp.get_session_state.user_state == agent_pb.US_LISTENING

        await host.aclose()

    @pytest.mark.asyncio
    async def test_get_session_usage(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        host.register_session(session)
        await host.start()

        req = agent_pb.SessionRequest(
            request_id="req-5",
            get_session_usage=agent_pb.SessionRequest.GetSessionUsage(),
        )
        await host._handle_request(req)

        assert len(transport.sent) == 1
        resp = transport.sent[0].response
        assert resp.request_id == "req-5"
        assert resp.HasField("get_session_usage")

        await host.aclose()

    @pytest.mark.asyncio
    async def test_handle_request_error(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        session.history = property(lambda s: (_ for _ in ()).throw(RuntimeError("broken")))
        host.register_session(session)
        await host.start()

        req = agent_pb.SessionRequest(
            request_id="req-err",
            get_chat_history=agent_pb.SessionRequest.GetChatHistory(),
        )
        await host._handle_request_safe(req)

        assert len(transport.sent) == 1
        resp = transport.sent[0].response
        assert resp.request_id == "req-err"
        assert resp.error == "internal error"

        await host.aclose()

    @pytest.mark.asyncio
    async def test_aclose_unregisters_events(self, transport: InMemoryTransport) -> None:
        host = SessionHost(transport)
        session = self._make_mock_session()
        host.register_session(session)
        await host.start()
        await host.aclose()
        assert session.off.call_count == 8
