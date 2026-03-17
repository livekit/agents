from __future__ import annotations

import asyncio
import struct
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from .. import llm, utils
from ..language import LanguageCode
from ..llm import (
    AgentConfigUpdate,
    AgentHandoff,
    ChatItem,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RawFunctionTool,
    Toolset,
)
from ..log import logger
from ..metrics import (
    AgentMetrics,
    AgentSessionUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    STTModelUsage,
    TTSModelUsage,
)
from ..types import (
    RPC_GET_AGENT_INFO,
    RPC_GET_CHAT_HISTORY,
    RPC_GET_SESSION_STATE,
    RPC_SEND_MESSAGE,
    TOPIC_AGENT_REQUEST,
    TOPIC_AGENT_RESPONSE,
    TOPIC_CHAT,
    TOPIC_CLIENT_EVENTS,
)
from .events import (
    AgentState,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SessionUsageUpdatedEvent,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..cli.tcp_console import TcpAudioInput, TcpAudioOutput
    from ..inference.interruption import OverlappingSpeechEvent
    from .agent_session import AgentSession
    from .room_io import RoomIO
    from .room_io.types import TextInputCallback


TOPIC_SESSION_MESSAGES = "lk.agent.session"


class SessionTransport(ABC):
    async def start(self) -> None: ...
    @abstractmethod
    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None: ...
    @abstractmethod
    async def close(self) -> None: ...
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]: ...
    @abstractmethod
    async def __anext__(self) -> agent_pb.AgentSessionMessage: ...


class RoomSessionTransport(SessionTransport):
    def __init__(self, room: rtc.Room, remote_identity: str | None = None) -> None:
        self._room = room
        self._remote_identity = remote_identity
        self._recv_ch: utils.aio.Chan[agent_pb.AgentSessionMessage] = utils.aio.Chan()
        self._handler_registered = False

    async def start(self) -> None:
        if self._handler_registered:
            return
        self._room.register_byte_stream_handler(
            TOPIC_SESSION_MESSAGES, self._on_byte_stream
        )
        self._handler_registered = True

    def _on_byte_stream(
        self, reader: rtc.ByteStreamReader, participant_identity: str
    ) -> None:
        if self._remote_identity and participant_identity != self._remote_identity:
            return
        asyncio.create_task(self._read_stream(reader))

    async def _read_stream(self, reader: rtc.ByteStreamReader) -> None:
        try:
            chunks: list[bytes] = []
            async for chunk in reader:
                chunks.append(chunk)
            data = b"".join(chunks)
            msg = agent_pb.AgentSessionMessage()
            msg.ParseFromString(data)
            self._recv_ch.send_nowait(msg)
        except utils.aio.ChanClosed:
            pass
        except Exception as e:
            logger.warning("failed to read binary stream message", exc_info=e)

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._recv_ch.closed or not self._room.isconnected():
            return
        try:
            data = msg.SerializeToString()
            kwargs: dict[str, str | list[str]] = {
                "topic": TOPIC_SESSION_MESSAGES,
                "name": TOPIC_SESSION_MESSAGES,
            }
            if self._remote_identity:
                kwargs["destination_identities"] = [self._remote_identity]
            writer = await self._room.local_participant.stream_bytes(**kwargs)
            await writer.write(data)
            await writer.aclose()
        except Exception:
            logger.warning("failed to send binary stream message", exc_info=True)

    async def close(self) -> None:
        if self._recv_ch.closed:
            return
        self._recv_ch.close()
        if self._handler_registered:
            try:
                self._room.unregister_byte_stream_handler(TOPIC_SESSION_MESSAGES)
            except (ValueError, AttributeError):
                pass
            self._handler_registered = False

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self._recv_ch.__aiter__()

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        return await self._recv_ch.__anext__()


_TCP_HEADER_SIZE = 4
_TCP_MAX_MESSAGE_SIZE = 1 << 20


class TcpSessionTransport(SessionTransport):

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        reader, writer = await asyncio.open_connection(self._host, self._port)
        sock = writer.transport.get_extra_info("socket")
        if sock is not None:
            import socket

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = reader
        self._writer = writer
        self._loop = asyncio.get_running_loop()

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._closed or self._writer is None:
            return
        data = msg.SerializeToString()
        header = struct.pack(">I", len(data))
        self._writer.write(header + data)
        if self._writer.transport.get_write_buffer_size() > 64 * 1024:
            await self._writer.drain()

    def send_message_threadsafe(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._closed or self._writer is None or self._loop is None:
            return
        data = msg.SerializeToString()
        payload = struct.pack(">I", len(data)) + data
        self._loop.call_soon_threadsafe(self._writer.write, payload)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed or self._reader is None:
            raise StopAsyncIteration

        try:
            header = await self._reader.readexactly(_TCP_HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            raise StopAsyncIteration

        length = struct.unpack(">I", header)[0]
        if length > _TCP_MAX_MESSAGE_SIZE:
            logger.error("TCP message too large: %d bytes", length)
            raise StopAsyncIteration

        try:
            data = await self._reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            raise StopAsyncIteration

        msg = agent_pb.AgentSessionMessage()
        msg.ParseFromString(data)
        return msg


_AGENT_STATE_MAP: dict[AgentState, int] = {
    "initializing": agent_pb.SESSION_AGENT_STATE_INITIALIZING,
    "idle": agent_pb.SESSION_AGENT_STATE_IDLE,
    "listening": agent_pb.SESSION_AGENT_STATE_LISTENING,
    "thinking": agent_pb.SESSION_AGENT_STATE_THINKING,
    "speaking": agent_pb.SESSION_AGENT_STATE_SPEAKING,
}

_USER_STATE_MAP: dict[UserState, int] = {
    "speaking": agent_pb.SESSION_USER_STATE_SPEAKING,
    "listening": agent_pb.SESSION_USER_STATE_LISTENING,
    "away": agent_pb.SESSION_USER_STATE_AWAY,
}

_METRICS_FIELDS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "e2e_latency",
)


def _tool_names(tools: list[Any]) -> list[str]:
    result: list[str] = []
    for tool in tools:
        if isinstance(tool, (FunctionTool, RawFunctionTool)):
            result.append(tool.info.name)
        elif isinstance(tool, Toolset):
            result.extend(_tool_names(tool.tools))
    return result


def _metrics_to_proto(metrics: dict | None) -> agent_pb.MetricsReport:
    if not metrics:
        return agent_pb.MetricsReport()
    kwargs = {k: metrics[k] for k in _METRICS_FIELDS if k in metrics}
    return agent_pb.MetricsReport(**kwargs)


def _chat_item_to_proto(item: llm.ChatItem) -> agent_pb.ChatContext.ChatItem:
    if isinstance(item, ChatMessage):
        role_map = {
            "developer": agent_pb.DEVELOPER,
            "system": agent_pb.SYSTEM,
            "user": agent_pb.USER,
            "assistant": agent_pb.ASSISTANT,
        }
        pb_role = role_map.get(item.role, agent_pb.ASSISTANT)
        content = []
        if item.text_content:
            content.append(agent_pb.ChatMessage.ChatContent(text=item.text_content))
        pb_msg = agent_pb.ChatMessage(
            id=item.id,
            role=pb_role,
            content=content,
            metrics=_metrics_to_proto(item.metrics),
        )
        return agent_pb.ChatContext.ChatItem(message=pb_msg)
    elif isinstance(item, FunctionCall):
        return agent_pb.ChatContext.ChatItem(
            function_call=agent_pb.FunctionCall(
                id=item.id,
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
            )
        )
    elif isinstance(item, FunctionCallOutput):
        return agent_pb.ChatContext.ChatItem(
            function_call_output=agent_pb.FunctionCallOutput(
                call_id=item.call_id,
                output=item.output,
                is_error=item.is_error,
            )
        )
    elif isinstance(item, AgentHandoff):
        return agent_pb.ChatContext.ChatItem(
            agent_handoff=agent_pb.AgentHandoff(
                id=item.id,
                old_agent_id=item.old_agent_id,
                new_agent_id=item.new_agent_id,
            )
        )
    elif isinstance(item, AgentConfigUpdate):
        return agent_pb.ChatContext.ChatItem(
            agent_config_update=agent_pb.AgentConfigUpdate(
                id=item.id,
                instructions=item.instructions,
                tools_added=item.tools_added or [],
                tools_removed=item.tools_removed or [],
            )
        )
    return agent_pb.ChatContext.ChatItem()


def _serialize_options(opts: Any) -> dict[str, str]:
    return {
        "endpointing": str(dict(opts.endpointing)),
        "interruption": str(dict(opts.interruption)),
        "max_tool_steps": str(opts.max_tool_steps),
        "user_away_timeout": str(opts.user_away_timeout),
        "preemptive_generation": str(opts.preemptive_generation),
        "min_consecutive_speech_delay": str(opts.min_consecutive_speech_delay),
        "use_tts_aligned_transcript": str(opts.use_tts_aligned_transcript),
        "ivr_detection": str(opts.ivr_detection),
    }


class _SessionHost:

    def __init__(
        self,
        transport: SessionTransport,
        audio_input: TcpAudioInput | None = None,
        audio_output: TcpAudioOutput | None = None,
    ) -> None:
        self._transport = transport
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._started = False
        self._recv_task: asyncio.Task[None] | None = None
        self._tasks = utils.aio.TaskSet()
        self._session: AgentSession | None = None
        self._events_registered = False

    def register_session(self, session: AgentSession) -> None:
        self._session = session
        if not self._events_registered:
            self._events_registered = True
            session.on("agent_state_changed", self._on_agent_state_changed)
            session.on("user_state_changed", self._on_user_state_changed)
            session.on("conversation_item_added", self._on_conversation_item_added)
            session.on("user_input_transcribed", self._on_user_input_transcribed)
            session.on("function_tools_executed", self._on_function_tools_executed)
            session.on("session_usage_updated", self._on_session_usage_updated)
            session.on("user_overlapping_speech", self._on_overlapping_speech)
            session.on("error", self._on_error)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self._transport.start()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def aclose(self) -> None:
        if not self._started:
            return
        self._started = False

        if self._session and self._events_registered:
            self._events_registered = False
            self._session.off("agent_state_changed", self._on_agent_state_changed)
            self._session.off("user_state_changed", self._on_user_state_changed)
            self._session.off("conversation_item_added", self._on_conversation_item_added)
            self._session.off("user_input_transcribed", self._on_user_input_transcribed)
            self._session.off("function_tools_executed", self._on_function_tools_executed)
            self._session.off("session_usage_updated", self._on_session_usage_updated)
            self._session.off("user_overlapping_speech", self._on_overlapping_speech)
            self._session.off("error", self._on_error)

        if self._recv_task:
            await utils.aio.cancel_and_wait(self._recv_task)

        await utils.aio.cancel_and_wait(*self._tasks.tasks)
        await self._transport.close()

    async def _recv_loop(self) -> None:
        try:
            async for msg in self._transport:
                if msg.HasField("request"):
                    if self._session is not None:
                        self._tasks.create_task(self._handle_request_safe(msg.request))
                else:
                    msg_type = msg.WhichOneof("message")
                    if msg_type:
                        self._dispatch_transport_message(msg_type, msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("error processing session message", exc_info=True)

    def _dispatch_transport_message(
        self, msg_type: str, msg: agent_pb.AgentSessionMessage
    ) -> None:
        if msg_type == "audio_input" and self._audio_input is not None:
            self._audio_input.push_frame(msg.audio_input)
        elif msg_type == "audio_playback_finished" and self._audio_output is not None:
            self._audio_output.notify_playout_finished()

    def _send_event(self, event: agent_pb.SessionEvent) -> None:
        msg = agent_pb.AgentSessionMessage(event=event)
        self._tasks.create_task(self._transport.send_message(msg))

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        old_pb = _AGENT_STATE_MAP.get(event.old_state, agent_pb.SESSION_AGENT_STATE_IDLE)
        new_pb = _AGENT_STATE_MAP.get(event.new_state, agent_pb.SESSION_AGENT_STATE_IDLE)
        self._send_event(
            agent_pb.SessionEvent(
                agent_state_changed=agent_pb.SessionEvent.AgentStateChanged(
                    old_state=old_pb,
                    new_state=new_pb,
                )
            )
        )

    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        old_pb = _USER_STATE_MAP.get(event.old_state, agent_pb.SESSION_USER_STATE_LISTENING)
        new_pb = _USER_STATE_MAP.get(event.new_state, agent_pb.SESSION_USER_STATE_LISTENING)
        self._send_event(
            agent_pb.SessionEvent(
                user_state_changed=agent_pb.SessionEvent.UserStateChanged(
                    old_state=old_pb,
                    new_state=new_pb,
                )
            )
        )

    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        self._send_event(
            agent_pb.SessionEvent(
                user_input_transcribed=agent_pb.SessionEvent.UserInputTranscribed(
                    transcript=event.transcript,
                    is_final=event.is_final,
                )
            )
        )

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        chat_item = _chat_item_to_proto(event.item)
        self._send_event(
            agent_pb.SessionEvent(
                conversation_item_added=agent_pb.SessionEvent.ConversationItemAdded(
                    item=chat_item,
                )
            )
        )

    def _on_function_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        pb_calls = [
            agent_pb.FunctionCall(
                name=fc.name,
                arguments=fc.arguments,
                call_id=fc.call_id,
            )
            for fc in event.function_calls
        ]
        pb_outputs = [
            agent_pb.FunctionCallOutput(
                call_id=fco.call_id,
                output=fco.output,
                is_error=fco.is_error,
            )
            for fco in event.function_call_outputs
        ]
        self._send_event(
            agent_pb.SessionEvent(
                function_tools_executed=agent_pb.SessionEvent.FunctionToolsExecuted(
                    function_calls=pb_calls,
                    function_call_outputs=pb_outputs,
                )
            )
        )

    def _on_overlapping_speech(self, event: OverlappingSpeechEvent) -> None:
        from google.protobuf.timestamp_pb2 import Timestamp

        detected_at = Timestamp()
        detected_at.FromSeconds(int(event.detected_at))

        kwargs: dict[str, Any] = {
            "is_interruption": event.is_interruption,
            "detection_delay": event.detection_delay,
            "detected_at": detected_at,
        }
        if event.overlap_started_at is not None:
            overlap_ts = Timestamp()
            overlap_ts.FromSeconds(int(event.overlap_started_at))
            kwargs["overlap_started_at"] = overlap_ts

        self._send_event(
            agent_pb.SessionEvent(
                overlapping_speech=agent_pb.SessionEvent.OverlappingSpeech(**kwargs)
            )
        )

    def _on_session_usage_updated(self, event: SessionUsageUpdatedEvent) -> None:
        self._send_event(
            agent_pb.SessionEvent(
                session_usage_updated=agent_pb.SessionEvent.SessionUsageUpdated(
                    usage=_session_usage_to_proto(event.usage),
                )
            )
        )

    def _on_error(self, event: ErrorEvent) -> None:
        self._send_event(
            agent_pb.SessionEvent(
                error=agent_pb.SessionEvent.Error(
                    message=str(event.error) if event.error else "Unknown error",
                )
            )
        )

    async def _handle_request_safe(self, req: agent_pb.SessionRequest) -> None:
        try:
            await self._handle_request(req)
        except Exception:
            logger.warning(
                "error handling session request",
                exc_info=True,
                extra={"request_id": req.request_id},
            )
            try:
                resp = agent_pb.AgentSessionMessage(
                    response=agent_pb.SessionResponse(
                        request_id=req.request_id,
                        error="internal error",
                    )
                )
                await self._transport.send_message(resp)
            except Exception:
                pass

    async def _handle_request(self, req: agent_pb.SessionRequest) -> None:
        assert self._session is not None

        if req.HasField("ping"):
            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    pong=agent_pb.SessionResponse.Pong(),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("get_chat_history"):
            items = [_chat_item_to_proto(item) for item in self._session.history.items]
            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    get_chat_history=agent_pb.SessionResponse.GetChatHistoryResponse(
                        items=items,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("get_agent_info"):
            agent = self._session.current_agent
            items = [_chat_item_to_proto(item) for item in agent.chat_ctx.items]
            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    get_agent_info=agent_pb.SessionResponse.GetAgentInfoResponse(
                        id=agent.id,
                        instructions=agent.instructions,
                        tools=_tool_names(agent.tools),
                        chat_ctx=items,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("run_input"):
            items_list: list[agent_pb.ChatContext.ChatItem] = []
            error: str | None = None
            text = req.run_input.text
            if text:
                self._session.output.audio = None
                self._session.output.transcription = None

                try:
                    await self._session.interrupt(force=True)
                except RuntimeError:
                    pass

                result = self._session.run(user_input=text)
                try:
                    await result
                except Exception as e:
                    error = str(e)
                items_list = [_chat_item_to_proto(ev.item) for ev in result.events]

            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    error=error,
                    run_input=agent_pb.SessionResponse.RunInputResponse(
                        items=items_list,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("get_session_state"):
            from google.protobuf.timestamp_pb2 import Timestamp

            agent = self._session.current_agent
            created_at = Timestamp()
            started_at = getattr(self._session, "_started_at", None) or time.time()
            created_at.FromSeconds(int(started_at))

            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    get_session_state=agent_pb.SessionResponse.GetSessionStateResponse(
                        agent_state=_AGENT_STATE_MAP.get(
                            self._session.agent_state,
                            agent_pb.SESSION_AGENT_STATE_IDLE,
                        ),
                        user_state=_USER_STATE_MAP.get(
                            self._session.user_state,
                            agent_pb.SESSION_USER_STATE_LISTENING,
                        ),
                        agent_id=agent.id,
                        options=_serialize_options(self._session.options),
                        created_at=created_at,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("get_rtc_stats"):
            from google.protobuf.json_format import ParseDict
            from google.protobuf.struct_pb2 import Struct

            rtc_stats = await self._session._room.get_rtc_stats() if hasattr(self._session, "_room") else None
            publisher_stats: list[Struct] = []
            subscriber_stats: list[Struct] = []
            if rtc_stats:
                from google.protobuf.json_format import MessageToDict

                for s in rtc_stats.publisher_stats:
                    d = MessageToDict(s)
                    st = Struct()
                    st.update(d)
                    publisher_stats.append(st)
                for s in rtc_stats.subscriber_stats:
                    d = MessageToDict(s)
                    st = Struct()
                    st.update(d)
                    subscriber_stats.append(st)

            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    get_rtc_stats=agent_pb.SessionResponse.GetRTCStatsResponse(
                        publisher_stats=publisher_stats,
                        subscriber_stats=subscriber_stats,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("get_session_usage"):
            from google.protobuf.timestamp_pb2 import Timestamp

            created_at = Timestamp()
            created_at.FromSeconds(int(time.time()))

            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    get_session_usage=agent_pb.SessionResponse.GetSessionUsageResponse(
                        usage=_session_usage_to_proto(self._session.usage),
                        created_at=created_at,
                    ),
                )
            )
            await self._transport.send_message(resp)


def _session_usage_to_proto(usage: AgentSessionUsage) -> agent_pb.AgentSessionUsage:
    model_usages: list[agent_pb.ModelUsage] = []
    for mu in usage.model_usage:
        if isinstance(mu, LLMModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    llm=agent_pb.LLMModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        input_tokens=mu.input_tokens,
                        input_cached_tokens=mu.input_cached_tokens,
                        input_audio_tokens=mu.input_audio_tokens,
                        input_cached_audio_tokens=mu.input_cached_audio_tokens,
                        input_text_tokens=mu.input_text_tokens,
                        input_cached_text_tokens=mu.input_cached_text_tokens,
                        input_image_tokens=mu.input_image_tokens,
                        input_cached_image_tokens=mu.input_cached_image_tokens,
                        output_tokens=mu.output_tokens,
                        output_audio_tokens=mu.output_audio_tokens,
                        output_text_tokens=mu.output_text_tokens,
                        session_duration=mu.session_duration,
                    )
                )
            )
        elif isinstance(mu, TTSModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    tts=agent_pb.TTSModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        input_tokens=mu.input_tokens,
                        output_tokens=mu.output_tokens,
                        characters_count=mu.characters_count,
                        audio_duration=mu.audio_duration,
                    )
                )
            )
        elif isinstance(mu, STTModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    stt=agent_pb.STTModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        input_tokens=mu.input_tokens,
                        output_tokens=mu.output_tokens,
                        audio_duration=mu.audio_duration,
                    )
                )
            )
        elif isinstance(mu, InterruptionModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    interruption=agent_pb.InterruptionModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        total_requests=mu.total_requests,
                    )
                )
            )
    return agent_pb.AgentSessionUsage(model_usage=model_usages)


# --- JSON-over-text-streams handler for room-based communication ---

# Pydantic models for JSON protocol (used by ClientEventsHandler and RemoteSession)
from pydantic import BaseModel, Field
from typing import Annotated


class ClientAgentStateChangedEvent(BaseModel):
    type: Literal["agent_state_changed"] = "agent_state_changed"
    old_state: AgentState
    new_state: AgentState
    created_at: float


class ClientUserStateChangedEvent(BaseModel):
    type: Literal["user_state_changed"] = "user_state_changed"
    old_state: UserState
    new_state: UserState
    created_at: float


class ClientConversationItemAddedEvent(BaseModel):
    type: Literal["conversation_item_added"] = "conversation_item_added"
    item: ChatMessage
    created_at: float


class ClientUserInputTranscribedEvent(BaseModel):
    type: Literal["user_input_transcribed"] = "user_input_transcribed"
    transcript: str
    is_final: bool
    language: LanguageCode | None
    created_at: float


class ClientFunctionToolsExecutedEvent(BaseModel):
    type: Literal["function_tools_executed"] = "function_tools_executed"
    function_calls: list[FunctionCall]
    function_call_outputs: list[FunctionCallOutput | None]
    created_at: float


class ClientMetricsCollectedEvent(BaseModel):
    type: Literal["metrics_collected"] = "metrics_collected"
    metrics: AgentMetrics
    created_at: float


class ClientErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    created_at: float


class ClientOverlappingSpeechEvent(BaseModel):
    type: Literal["overlapping_speech"] = "overlapping_speech"
    is_interruption: bool
    created_at: float
    detected_at: float
    detection_delay: float
    overlap_started_at: float | None


class ClientSessionUsageUpdatedEvent(BaseModel):
    type: Literal["session_usage_updated"] = "session_usage_updated"
    usage: AgentSessionUsage
    created_at: float


ClientEvent = Annotated[
    ClientAgentStateChangedEvent
    | ClientUserStateChangedEvent
    | ClientConversationItemAddedEvent
    | ClientUserInputTranscribedEvent
    | ClientFunctionToolsExecutedEvent
    | ClientMetricsCollectedEvent
    | ClientErrorEvent
    | ClientOverlappingSpeechEvent
    | ClientSessionUsageUpdatedEvent,
    Field(discriminator="type"),
]


class GetSessionStateRequest(BaseModel):
    pass


class GetSessionStateResponse(BaseModel):
    agent_state: AgentState
    user_state: UserState
    agent_id: str
    options: dict[str, Any]
    created_at: float


class GetChatHistoryRequest(BaseModel):
    pass


class GetChatHistoryResponse(BaseModel):
    items: list[ChatItem]


class GetAgentInfoRequest(BaseModel):
    pass


class GetAgentInfoResponse(BaseModel):
    id: str
    instructions: str | None
    tools: list[str]
    chat_ctx: list[ChatItem]


class RunInputRequest(BaseModel):
    text: str


class RunInputResponse(BaseModel):
    items: list[ChatItem]


class GetRTCStatsRequest(BaseModel):
    pass


class GetRTCStatsResponse(BaseModel):
    publisher_stats: list[dict[str, Any]]
    subscriber_stats: list[dict[str, Any]]


class GetSessionUsageRequest(BaseModel):
    pass


class GetSessionUsageResponse(BaseModel):
    usage: AgentSessionUsage
    created_at: float


class StreamRequest(BaseModel):
    request_id: str
    method: str
    payload: str


class StreamResponse(BaseModel):
    request_id: str
    payload: str
    error: str | None = None


class ClientEventsHandler:
    """
    Handles exposing AgentSession state to room participants and allows interaction.

    This class provides:
    - Event streaming: Automatically streams AgentSession events to clients via a text stream
    - RPC handlers: Allows clients to request state, chat history, and agent info on demand
    - Text input handling: Receives text messages from clients and generates agent replies
    """

    def __init__(
        self,
        session: AgentSession,
        room_io: RoomIO,
    ) -> None:
        self._session = session
        self._room_io = room_io

        self._text_input_cb: TextInputCallback | None = None
        self._text_stream_handler_registered = False
        self._rpc_handlers_registered = False
        self._request_handler_registered = False
        self._event_handlers_registered = False

        self._tasks: set[asyncio.Task[Any]] = set()
        self._started = False

    @property
    def _room(self) -> rtc.Room:
        return self._room_io.room

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._register_rpc_handlers()
        self._register_request_handler()
        self._register_event_handlers()

    async def aclose(self) -> None:
        if not self._started:
            return

        self._started = False

        if self._text_stream_handler_registered:
            try:
                self._room.unregister_text_stream_handler(TOPIC_CHAT)
            except ValueError:
                pass
            self._text_stream_handler_registered = False

        if self._rpc_handlers_registered:
            try:
                self._room.local_participant.unregister_rpc_method(RPC_GET_SESSION_STATE)
                self._room.local_participant.unregister_rpc_method(RPC_GET_CHAT_HISTORY)
                self._room.local_participant.unregister_rpc_method(RPC_GET_AGENT_INFO)
                self._room.local_participant.unregister_rpc_method(RPC_SEND_MESSAGE)
            except Exception:
                pass
            self._rpc_handlers_registered = False

        if self._request_handler_registered:
            try:
                self._room.unregister_text_stream_handler(TOPIC_AGENT_REQUEST)
            except ValueError:
                pass
            self._request_handler_registered = False

        if self._event_handlers_registered:
            self._session.off("agent_state_changed", self._on_agent_state_changed)
            self._session.off("user_state_changed", self._on_user_state_changed)
            self._session.off("conversation_item_added", self._on_conversation_item_added)
            self._session.off("function_tools_executed", self._on_function_tools_executed)
            self._session.off("metrics_collected", self._on_metrics_collected)
            self._session.off("session_usage_updated", self._on_session_usage_updated)
            self._session.off("user_input_transcribed", self._on_user_input_transcribed)
            self._session.off("user_overlapping_speech", self._on_overlapping_speech)
            self._session.off("error", self._on_error)
            self._event_handlers_registered = False

        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    def register_text_input(self, text_input_cb: TextInputCallback) -> None:
        self._text_input_cb = text_input_cb

        if not self._text_stream_handler_registered:
            try:
                self._room.register_text_stream_handler(TOPIC_CHAT, self._on_user_text_input)
                self._text_stream_handler_registered = True
            except ValueError:
                logger.warning(
                    f"text stream handler for topic '{TOPIC_CHAT}' already set, ignoring"
                )

    def _register_rpc_handlers(self) -> None:
        if self._rpc_handlers_registered:
            return

        self._room.local_participant.register_rpc_method(
            RPC_GET_SESSION_STATE, self._rpc_get_session_state
        )
        self._room.local_participant.register_rpc_method(
            RPC_GET_CHAT_HISTORY, self._rpc_get_chat_history
        )
        self._room.local_participant.register_rpc_method(
            RPC_GET_AGENT_INFO, self._rpc_get_agent_info
        )
        self._room.local_participant.register_rpc_method(RPC_SEND_MESSAGE, self._rpc_send_message)

        self._rpc_handlers_registered = True

    def _register_request_handler(self) -> None:
        if self._request_handler_registered:
            return

        try:
            self._room.register_text_stream_handler(TOPIC_AGENT_REQUEST, self._on_stream_request)
            self._request_handler_registered = True
        except ValueError:
            logger.warning(f"text stream handler for topic '{TOPIC_AGENT_REQUEST}' already set")

    def _on_stream_request(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        task = asyncio.create_task(self._handle_stream_request(reader, participant_identity))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @utils.log_exceptions(logger=logger)
    async def _handle_stream_request(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ) -> None:
        try:
            data = await reader.read_all()
            request = StreamRequest.model_validate_json(data)

            response_payload: str
            error: str | None = None

            try:
                if request.method == "get_session_state":
                    response_payload = await self._stream_get_session_state(request.payload)
                elif request.method == "get_chat_history":
                    response_payload = await self._stream_get_chat_history(request.payload)
                elif request.method == "get_agent_info":
                    response_payload = await self._stream_get_agent_info(request.payload)
                elif request.method == "run_input":
                    response_payload = await self._stream_run_input(request.payload)
                elif request.method == "get_rtc_stats":
                    response_payload = await self._stream_get_rtc_stats(request.payload)
                elif request.method == "get_session_usage":
                    response_payload = await self._stream_get_session_usage(request.payload)
                else:
                    response_payload = ""
                    error = f"Unknown method: {request.method}"
            except Exception as e:
                response_payload = ""
                error = str(e)

            response = StreamResponse(
                request_id=request.request_id,
                payload=response_payload,
                error=error,
            )

            await self._room.local_participant.send_text(
                response.model_dump_json(),
                topic=TOPIC_AGENT_RESPONSE,
                destination_identities=[participant_identity],
            )

        except Exception as e:
            logger.warning("failed to handle stream request", exc_info=e)

    async def _stream_get_session_state(self, payload: str) -> str:
        agent = self._session.current_agent
        response = GetSessionStateResponse(
            agent_state=self._session.agent_state,
            user_state=self._session.user_state,
            agent_id=agent.id,
            options={k: str(v) for k, v in _serialize_options(self._session.options).items()},
            created_at=self._session._started_at or time.time(),
        )
        return response.model_dump_json()

    async def _stream_get_chat_history(self, payload: str) -> str:
        response = GetChatHistoryResponse(items=list(self._session.history.items))
        return response.model_dump_json()

    async def _stream_get_agent_info(self, payload: str) -> str:
        agent = self._session.current_agent
        response = GetAgentInfoResponse(
            id=agent.id,
            instructions=agent.instructions,
            tools=_tool_names(agent.tools),
            chat_ctx=list(agent.chat_ctx.items),
        )
        return response.model_dump_json()

    async def _stream_run_input(self, payload: str) -> str:
        from .run_result import RunResult

        request = RunInputRequest.model_validate_json(payload)
        run_result: RunResult[None] = await self._session.run(user_input=request.text)

        items: list[ChatItem] = []
        for event in run_result.events:
            items.append(event.item)

        response = RunInputResponse(items=items)
        return response.model_dump_json()

    async def _stream_get_rtc_stats(self, payload: str) -> str:
        from google.protobuf.json_format import MessageToDict

        rtc_stats = await self._room.get_rtc_stats()
        response = GetRTCStatsResponse(
            publisher_stats=[MessageToDict(s) for s in rtc_stats.publisher_stats],
            subscriber_stats=[MessageToDict(s) for s in rtc_stats.subscriber_stats],
        )
        return response.model_dump_json()

    async def _stream_get_session_usage(self, payload: str) -> str:
        response = GetSessionUsageResponse(
            usage=self._session.usage,
            created_at=time.time(),
        )
        return response.model_dump_json()

    def _register_event_handlers(self) -> None:
        if self._event_handlers_registered:
            return

        self._session.on("agent_state_changed", self._on_agent_state_changed)
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("conversation_item_added", self._on_conversation_item_added)
        self._session.on("function_tools_executed", self._on_function_tools_executed)
        self._session.on("metrics_collected", self._on_metrics_collected)
        self._session.on("session_usage_updated", self._on_session_usage_updated)
        self._session.on("user_input_transcribed", self._on_user_input_transcribed)
        self._session.on("user_overlapping_speech", self._on_overlapping_speech)
        self._session.on("error", self._on_error)

        self._event_handlers_registered = True

    def _on_overlapping_speech(self, event: OverlappingSpeechEvent) -> None:
        client_event = ClientOverlappingSpeechEvent(
            is_interruption=event.is_interruption,
            created_at=time.time(),
            detected_at=event.detected_at,
            overlap_started_at=event.overlap_started_at,
            detection_delay=event.detection_delay,
        )
        self._stream_client_event(client_event)

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        client_event = ClientAgentStateChangedEvent(
            old_state=event.old_state,
            new_state=event.new_state,
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        client_event = ClientUserStateChangedEvent(
            old_state=event.old_state,
            new_state=event.new_state,
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        if not isinstance(event.item, ChatMessage):
            return

        client_event = ClientConversationItemAddedEvent(
            item=event.item,
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        client_event = ClientUserInputTranscribedEvent(
            transcript=event.transcript,
            is_final=event.is_final,
            language=event.language,
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _on_function_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        client_event = ClientFunctionToolsExecutedEvent(
            function_calls=event.function_calls,
            function_call_outputs=event.function_call_outputs,
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _on_metrics_collected(self, event: MetricsCollectedEvent) -> None:
        if event.metrics is None:
            return

        metrics_event = ClientMetricsCollectedEvent(
            metrics=event.metrics,
            created_at=event.created_at,
        )
        self._stream_client_event(metrics_event)

    def _on_session_usage_updated(self, event: SessionUsageUpdatedEvent) -> None:
        usage_event = ClientSessionUsageUpdatedEvent(
            usage=event.usage,
            created_at=event.created_at,
        )
        self._stream_client_event(usage_event)

    def _on_error(self, event: ErrorEvent) -> None:
        client_event = ClientErrorEvent(
            message=str(event.error) if event.error else "Unknown error",
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _get_target_identities(self) -> list[str] | None:
        linked = self._room_io.linked_participant
        if linked is None:
            return None

        has_permission = True
        if not has_permission:
            return None

        return [linked.identity]

    def _stream_client_event(self, event: ClientEvent) -> None:
        task = asyncio.create_task(self._send_client_event(event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @utils.log_exceptions(logger=logger)
    async def _send_client_event(self, event: ClientEvent) -> None:
        if not self._room.isconnected():
            return

        destination_identities = self._get_target_identities()
        if destination_identities is None:
            return

        try:
            event_data = event.model_dump_json()

            writer = await self._room.local_participant.stream_text(
                topic=TOPIC_CLIENT_EVENTS,
                destination_identities=destination_identities,
            )
            await writer.write(event_data)
            await writer.aclose()
        except Exception as e:
            logger.warning("failed to stream event to clients", exc_info=e)

    async def _rpc_get_session_state(self, data: rtc.RpcInvocationData) -> str:
        agent = self._session.current_agent

        response = GetSessionStateResponse(
            agent_state=self._session.agent_state,
            user_state=self._session.user_state,
            agent_id=agent.id,
            options={k: str(v) for k, v in _serialize_options(self._session.options).items()},
            created_at=self._session._started_at or time.time(),
        )

        return response.model_dump_json()

    async def _rpc_get_chat_history(self, data: rtc.RpcInvocationData) -> str:
        response = GetChatHistoryResponse(items=list(self._session.history.items))
        return response.model_dump_json()

    async def _rpc_get_agent_info(self, data: rtc.RpcInvocationData) -> str:
        agent = self._session.current_agent
        response = GetAgentInfoResponse(
            id=agent.id,
            instructions=agent.instructions,
            tools=_tool_names(agent.tools),
            chat_ctx=list(agent.chat_ctx.items),
        )
        return response.model_dump_json()

    async def _rpc_send_message(self, data: rtc.RpcInvocationData) -> str:
        from .run_result import RunResult

        request = RunInputRequest.model_validate_json(data.payload)
        run_result: RunResult[None] = await self._session.run(user_input=request.text)

        items: list[ChatItem] = []
        for event in run_result.events:
            items.append(event.item)

        response = RunInputResponse(items=items)
        return response.model_dump_json()

    def _on_user_text_input(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        linked_participant = self._room_io.linked_participant
        if linked_participant and participant_identity != linked_participant.identity:
            return

        participant = self._room.remote_participants.get(participant_identity)
        if not participant:
            logger.warning("participant not found, ignoring text input")
            return

        async def _read_text(text_input_cb: TextInputCallback) -> None:
            from .room_io.types import TextInputEvent

            text = await reader.read_all()

            text_input_result = text_input_cb(
                self._session,
                TextInputEvent(text=text, info=reader.info, participant=participant),
            )
            if asyncio.iscoroutine(text_input_result):
                await text_input_result

        if self._text_input_cb is None:
            logger.error("text input callback is not set, ignoring text input")
            return

        task = asyncio.create_task(_read_text(self._text_input_cb))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


RemoteSessionEventTypes = Literal[
    "agent_state_changed",
    "user_state_changed",
    "conversation_item_added",
    "user_input_transcribed",
    "function_tools_executed",
    "metrics_collected",
    "overlapping_speech",
    "session_usage_updated",
    "error",
]


class RemoteSession(rtc.EventEmitter[RemoteSessionEventTypes]):
    """
    Client-side interface to interact with a remote AgentSession.

    This class allows frontends/clients to:
    - Subscribe to real-time events from the agent session
    - Query session state, chat history, and agent info via RPC
    - Send messages to the agent
    """

    def __init__(
        self,
        room: rtc.Room,
        agent_identity: str,
    ) -> None:
        super().__init__()
        self._room = room
        self._agent_identity = agent_identity
        self._started = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._pending_requests: dict[str, asyncio.Future[StreamResponse]] = {}

    @classmethod
    def from_room(cls, room: rtc.Room, agent_identity: str) -> RemoteSession:
        transport = RoomSessionTransport(room, agent_identity)
        return cls(room, agent_identity)

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._room.register_text_stream_handler(TOPIC_CLIENT_EVENTS, self._on_event_stream)
        self._room.register_text_stream_handler(TOPIC_AGENT_RESPONSE, self._on_response_stream)

    async def aclose(self) -> None:
        if not self._started:
            return

        self._started = False
        try:
            self._room.unregister_text_stream_handler(TOPIC_CLIENT_EVENTS)
        except ValueError:
            pass
        try:
            self._room.unregister_text_stream_handler(TOPIC_AGENT_RESPONSE)
        except ValueError:
            pass

        for future in self._pending_requests.values():
            future.cancel()
        self._pending_requests.clear()

        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    def _on_event_stream(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity != self._agent_identity:
            return

        task = asyncio.create_task(self._read_event(reader))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_response_stream(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity != self._agent_identity:
            return

        task = asyncio.create_task(self._read_response(reader))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _read_response(self, reader: rtc.TextStreamReader) -> None:
        try:
            data = await reader.read_all()
            response = StreamResponse.model_validate_json(data)

            future = self._pending_requests.pop(response.request_id, None)
            if future and not future.done():
                future.set_result(response)
        except Exception as e:
            logger.warning("failed to read stream response", exc_info=e)

    async def _read_event(self, reader: rtc.TextStreamReader) -> None:
        try:
            data = await reader.read_all()
            event = self._parse_event(data)
            if event:
                self.emit(event.type, event)
        except Exception as e:
            logger.warning("failed to parse client event", exc_info=e)

    def _parse_event(self, data: str) -> ClientEvent | None:
        from pydantic import TypeAdapter

        try:
            return TypeAdapter(ClientEvent).validate_json(data)
        except Exception as e:
            logger.warning(f"failed to parse event: {e}")
            return None

    async def _send_request(self, method: str, payload: str, timeout: float = 60.0) -> str:
        request_id = utils.shortuuid("req_")
        request = StreamRequest(
            request_id=request_id,
            method=method,
            payload=payload,
        )

        future: asyncio.Future[StreamResponse] = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            await self._room.local_participant.send_text(
                request.model_dump_json(),
                topic=TOPIC_AGENT_REQUEST,
                destination_identities=[self._agent_identity],
            )

            response = await asyncio.wait_for(future, timeout=timeout)

            if response.error:
                raise Exception(response.error)

            return response.payload

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise

    async def wait_for_ready(
        self, timeout: float = 5.0, retry_interval: float = 0.5
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError("wait_for_ready timed out")
            try:
                await self.fetch_session_state()
                return
            except (TimeoutError, asyncio.TimeoutError):
                if asyncio.get_event_loop().time() >= deadline:
                    raise TimeoutError("wait_for_ready timed out")

    async def fetch_session_state(self) -> GetSessionStateResponse:
        request = GetSessionStateRequest()
        response = await self._send_request(
            method="get_session_state",
            payload=request.model_dump_json(),
        )
        return GetSessionStateResponse.model_validate_json(response)

    async def fetch_chat_history(self) -> GetChatHistoryResponse:
        request = GetChatHistoryRequest()
        response = await self._send_request(
            method="get_chat_history",
            payload=request.model_dump_json(),
        )
        return GetChatHistoryResponse.model_validate_json(response)

    async def fetch_agent_info(self) -> GetAgentInfoResponse:
        request = GetAgentInfoRequest()
        response = await self._send_request(
            method="get_agent_info",
            payload=request.model_dump_json(),
        )
        return GetAgentInfoResponse.model_validate_json(response)

    async def run_input(self, text: str, timeout: float = 60.0) -> RunInputResponse:
        request = RunInputRequest(text=text)
        response = await self._send_request(
            method="run_input",
            payload=request.model_dump_json(),
            timeout=timeout,
        )
        return RunInputResponse.model_validate_json(response)

    async def fetch_rtc_stats(self) -> GetRTCStatsResponse:
        request = GetRTCStatsRequest()
        response = await self._send_request(
            method="get_rtc_stats",
            payload=request.model_dump_json(),
        )
        return GetRTCStatsResponse.model_validate_json(response)

    async def fetch_session_usage(self) -> GetSessionUsageResponse:
        request = GetSessionUsageRequest()
        response = await self._send_request(
            method="get_session_usage",
            payload=request.model_dump_json(),
        )
        return GetSessionUsageResponse.model_validate_json(response)
