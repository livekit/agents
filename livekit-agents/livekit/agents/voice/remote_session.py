from __future__ import annotations

import asyncio
import struct
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from google.protobuf.timestamp_pb2 import Timestamp

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from .. import llm, utils
from ..llm import (
    AgentConfigUpdate,
    AgentHandoff,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RawFunctionTool,
    Toolset,
)
from ..log import logger
from ..metrics import (
    AgentSessionUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    STTModelUsage,
    TTSModelUsage,
)
from .events import (
    AgentState,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    SessionUsageUpdatedEvent,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)
from .run_result import RunResult

if TYPE_CHECKING:
    from ..cli.tcp_console import TcpAudioInput, TcpAudioOutput  # type: ignore[import-untyped]
    from ..inference.interruption import OverlappingSpeechEvent
    from .agent_session import AgentSession, AgentSessionOptions
    from .room_io.types import TextInputCallback


TOPIC_SESSION_MESSAGES = "lk.agent.session"


class SessionTransport(ABC):
    @abstractmethod
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
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def remote_identity(self) -> str | None:
        return self._remote_identity

    @remote_identity.setter
    def remote_identity(self, value: str | None) -> None:
        self._remote_identity = value

    async def start(self) -> None:
        if self._handler_registered:
            return
        self._room.register_byte_stream_handler(TOPIC_SESSION_MESSAGES, self._on_byte_stream)
        self._handler_registered = True

    def _on_byte_stream(self, reader: rtc.ByteStreamReader, participant_identity: str) -> None:
        if self._remote_identity and participant_identity != self._remote_identity:
            return
        task = asyncio.create_task(self._read_stream(reader))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

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
            dest = [self._remote_identity] if self._remote_identity else None
            writer = await self._room.local_participant.stream_bytes(
                name=utils.shortuuid("AS_"),
                topic=TOPIC_SESSION_MESSAGES,
                destination_identities=dest,
            )
            await writer.write(data)
            await writer.aclose()
        except Exception:
            logger.warning("failed to send binary stream message", exc_info=True)

    async def close(self) -> None:
        if self._recv_ch.closed:
            return
        self._recv_ch.close()
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()
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
            raise StopAsyncIteration from None

        length = struct.unpack(">I", header)[0]
        if length > _TCP_MAX_MESSAGE_SIZE:
            logger.error("TCP message too large: %d bytes", length)
            raise StopAsyncIteration

        try:
            data = await self._reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            raise StopAsyncIteration from None

        msg = agent_pb.AgentSessionMessage()
        msg.ParseFromString(data)
        return msg


_AGENT_STATE_MAP: dict[AgentState, agent_pb.AgentState] = {
    "initializing": agent_pb.AS_INITIALIZING,
    "idle": agent_pb.AS_IDLE,
    "listening": agent_pb.AS_LISTENING,
    "thinking": agent_pb.AS_THINKING,
    "speaking": agent_pb.AS_SPEAKING,
}

_USER_STATE_MAP: dict[UserState, agent_pb.UserState] = {
    "speaking": agent_pb.US_SPEAKING,
    "listening": agent_pb.US_LISTENING,
    "away": agent_pb.US_AWAY,
}

_METRICS_FIELDS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "e2e_latency",
)


def _tool_names(tools: Sequence[llm.Tool | Toolset]) -> list[str]:
    result: list[str] = []
    for tool in tools:
        if isinstance(tool, FunctionTool | RawFunctionTool):
            result.append(tool.info.name)
        elif isinstance(tool, Toolset):
            result.extend(_tool_names(tool.tools))
    return result


def _metrics_to_proto(metrics: Mapping[str, Any] | None) -> agent_pb.MetricsReport:
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


def _serialize_options(opts: AgentSessionOptions) -> dict[str, str]:
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


class SessionHost:
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
        self._text_input_cb: TextInputCallback | None = None

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
            session.on("overlapping_speech", self._on_overlapping_speech)
            session.on("error", self._on_error)

    def register_text_input(self, text_input_cb: TextInputCallback) -> None:
        self._text_input_cb = text_input_cb

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
            self._session.off("overlapping_speech", self._on_overlapping_speech)
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

    def _dispatch_transport_message(self, msg_type: str, msg: agent_pb.AgentSessionMessage) -> None:
        if msg_type == "audio_input" and self._audio_input is not None:
            self._audio_input.push_frame(msg.audio_input)
        elif msg_type == "audio_playback_finished" and self._audio_output is not None:
            self._audio_output.notify_playout_finished()

    def _send_event(
        self, event: agent_pb.AgentSessionEvent, created_at: float | None = None
    ) -> None:
        ts = Timestamp()
        ts.FromNanoseconds(int((created_at if created_at is not None else time.time()) * 1e9))
        event.created_at.CopyFrom(ts)
        msg = agent_pb.AgentSessionMessage(event=event)
        self._tasks.create_task(self._transport.send_message(msg))

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        old_pb = _AGENT_STATE_MAP.get(event.old_state, agent_pb.AS_IDLE)
        new_pb = _AGENT_STATE_MAP.get(event.new_state, agent_pb.AS_IDLE)
        self._send_event(
            agent_pb.AgentSessionEvent(
                agent_state_changed=agent_pb.AgentSessionEvent.AgentStateChanged(
                    old_state=old_pb,
                    new_state=new_pb,
                )
            )
        )

    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        old_pb = _USER_STATE_MAP.get(event.old_state, agent_pb.US_LISTENING)
        new_pb = _USER_STATE_MAP.get(event.new_state, agent_pb.US_LISTENING)
        # use the original timestamp which is adjusted for VAD latency
        self._send_event(
            agent_pb.AgentSessionEvent(
                user_state_changed=agent_pb.AgentSessionEvent.UserStateChanged(
                    old_state=old_pb,
                    new_state=new_pb,
                )
            ),
            created_at=event.created_at,
        )

    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        self._send_event(
            agent_pb.AgentSessionEvent(
                user_input_transcribed=agent_pb.AgentSessionEvent.UserInputTranscribed(
                    transcript=event.transcript,
                    is_final=event.is_final,
                )
            )
        )

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        if not isinstance(
            event.item,
            ChatMessage | FunctionCall | FunctionCallOutput | AgentHandoff | AgentConfigUpdate,
        ):
            return
        chat_item = _chat_item_to_proto(event.item)
        self._send_event(
            agent_pb.AgentSessionEvent(
                conversation_item_added=agent_pb.AgentSessionEvent.ConversationItemAdded(
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
            if fco is not None
        ]
        self._send_event(
            agent_pb.AgentSessionEvent(
                function_tools_executed=agent_pb.AgentSessionEvent.FunctionToolsExecuted(
                    function_calls=pb_calls,
                    function_call_outputs=pb_outputs,
                )
            )
        )

    def _on_overlapping_speech(self, event: OverlappingSpeechEvent) -> None:
        detected_at = Timestamp()
        detected_at.FromNanoseconds(int(event.detected_at * 1e9))

        overlap_started_at: Timestamp | None = None
        if event.overlap_started_at is not None:
            overlap_started_at = Timestamp()
            overlap_started_at.FromNanoseconds(int(event.overlap_started_at * 1e9))

        pb = agent_pb.AgentSessionEvent.OverlappingSpeech(
            is_interruption=event.is_interruption,
            detection_delay=event.detection_delay,
            detected_at=detected_at,
        )
        if overlap_started_at is not None:
            pb.overlap_started_at.CopyFrom(overlap_started_at)

        self._send_event(agent_pb.AgentSessionEvent(overlapping_speech=pb))

    def _on_session_usage_updated(self, event: SessionUsageUpdatedEvent) -> None:
        self._send_event(
            agent_pb.AgentSessionEvent(
                session_usage_updated=agent_pb.AgentSessionEvent.SessionUsageUpdated(
                    usage=_session_usage_to_proto(event.usage),
                )
            )
        )

    def _on_error(self, event: ErrorEvent) -> None:
        self._send_event(
            agent_pb.AgentSessionEvent(
                error=agent_pb.AgentSessionEvent.Error(
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
                if self._text_input_cb is not None:
                    from .room_io.types import TextInputEvent

                    cb_result = self._text_input_cb(
                        self._session,
                        TextInputEvent(text=text, info=None, participant=None),
                    )
                    if asyncio.iscoroutine(cb_result):
                        await cb_result
                else:
                    try:
                        await self._session.interrupt(force=True)
                    except RuntimeError:
                        pass

                    try:
                        result: RunResult[None] = self._session.run(user_input=text)
                        await result
                        items_list = [_chat_item_to_proto(ev.item) for ev in result.events]
                    except Exception as e:
                        error = str(e)

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
            agent = self._session.current_agent
            created_at = Timestamp()
            started_at = self._session._started_at or time.time()
            created_at.FromNanoseconds(int(started_at * 1e9))

            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    get_session_state=agent_pb.SessionResponse.GetSessionStateResponse(
                        agent_state=_AGENT_STATE_MAP.get(
                            self._session.agent_state,
                            agent_pb.AS_IDLE,
                        ),
                        user_state=_USER_STATE_MAP.get(
                            self._session.user_state,
                            agent_pb.US_LISTENING,
                        ),
                        agent_id=agent.id,
                        options=_serialize_options(self._session.options),
                        created_at=created_at,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("get_rtc_stats"):
            from google.protobuf.struct_pb2 import Struct

            rtc_stats = (
                await self._session._room_io.room.get_rtc_stats()
                if self._session._room_io is not None
                else None
            )
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
            created_at = Timestamp()
            created_at.FromNanoseconds(int(time.time() * 1e9))

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
