from __future__ import annotations

import asyncio
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

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
from .events import (
    AgentState,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)

if TYPE_CHECKING:
    from ..cli.tcp_console import TcpAudioInput, TcpAudioOutput
    from .agent_session import AgentSession


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
            data = await reader.read_all()
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
            kwargs: dict[str, str | list[str]] = {"topic": TOPIC_SESSION_MESSAGES}
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

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._closed = False
        self._loop = asyncio.get_running_loop()

    @classmethod
    async def connect(cls, host: str, port: int) -> TcpSessionTransport:
        reader, writer = await asyncio.open_connection(host, port)
        sock = writer.transport.get_extra_info("socket")
        if sock is not None:
            import socket

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(reader, writer)

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._closed:
            return
        data = msg.SerializeToString()
        header = struct.pack(">I", len(data))
        self._writer.write(header + data)
        if self._writer.transport.get_write_buffer_size() > 64 * 1024:
            await self._writer.drain()

    def send_message_threadsafe(self, msg: agent_pb.AgentSessionMessage) -> None:
        if self._closed:
            return
        data = msg.SerializeToString()
        payload = struct.pack(">I", len(data)) + data
        self._loop.call_soon_threadsafe(self._writer.write, payload)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass

    def __aiter__(self) -> AsyncIterator[agent_pb.AgentSessionMessage]:
        return self

    async def __anext__(self) -> agent_pb.AgentSessionMessage:
        if self._closed:
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


def _tool_names(tools: list) -> list[str]:
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
                arguments=item.raw_arguments,
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
            session.on("metrics_collected", self._on_metrics_collected)
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
            self._session.off("metrics_collected", self._on_metrics_collected)
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

    # -- event forwarding --

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
                arguments=fc.raw_arguments,
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

    def _on_metrics_collected(self, event: MetricsCollectedEvent) -> None:
        pass

    def _on_error(self, event: ErrorEvent) -> None:
        self._send_event(
            agent_pb.SessionEvent(
                error=agent_pb.SessionEvent.Error(
                    message=str(event.error) if event.error else "Unknown error",
                )
            )
        )

    # -- request handling --

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
                    get_chat_history=agent_pb.SessionRequest.GetChatHistoryResponse(
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
                    get_agent_info=agent_pb.SessionRequest.GetAgentInfoResponse(
                        id=agent.id,
                        instructions=agent.instructions,
                        tools=_tool_names(agent.tools),
                        chat_ctx=items,
                    ),
                )
            )
            await self._transport.send_message(resp)

        elif req.HasField("send_message"):
            items: list[agent_pb.ChatContext.ChatItem] = []
            error: str | None = None
            text = req.send_message.text
            if text:
                result = self._session.run(user_input=text)
                try:
                    await result
                except Exception as e:
                    error = str(e)
                items = [_chat_item_to_proto(ev.item) for ev in result.events]

            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    error=error,
                    send_message=agent_pb.SessionRequest.SendMessageResponse(
                        items=items,
                    ),
                )
            )
            await self._transport.send_message(resp)


RemoteSessionEventTypes = Literal[
    "agent_state_changed",
    "user_state_changed",
    "conversation_item_added",
    "user_input_transcribed",
    "function_tools_executed",
    "metrics_collected",
    "error",
]


class RemoteSession(rtc.EventEmitter[RemoteSessionEventTypes]):

    def __init__(self, transport: SessionTransport) -> None:
        super().__init__()
        self._transport = transport
        self._started = False
        self._pending_requests: dict[str, asyncio.Future[agent_pb.SessionResponse]] = {}
        self._recv_task: asyncio.Task[None] | None = None

    @classmethod
    def from_room(cls, room: rtc.Room, agent_identity: str) -> RemoteSession:
        transport = RoomSessionTransport(room, agent_identity)
        return cls(transport)

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

        for future in self._pending_requests.values():
            future.cancel()
        self._pending_requests.clear()

        if self._recv_task:
            await utils.aio.cancel_and_wait(self._recv_task)

        await self._transport.close()

    async def _recv_loop(self) -> None:
        try:
            async for msg in self._transport:
                if msg.HasField("response"):
                    self._dispatch_response(msg.response)
                elif msg.HasField("event"):
                    event_field = msg.event.WhichOneof("event")
                    if event_field:
                        self.emit(event_field, msg.event)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("error processing session message", exc_info=True)

    def _dispatch_response(self, response: agent_pb.SessionResponse) -> None:
        future = self._pending_requests.pop(response.request_id, None)
        if future and not future.done():
            future.set_result(response)

    async def _send_request(
        self,
        request: agent_pb.SessionRequest,
        timeout: float = 60.0,
    ) -> agent_pb.SessionResponse:
        future: asyncio.Future[agent_pb.SessionResponse] = asyncio.Future()
        self._pending_requests[request.request_id] = future

        try:
            msg = agent_pb.AgentSessionMessage(request=request)
            await self._transport.send_message(msg)
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.TimeoutError, Exception):
            self._pending_requests.pop(request.request_id, None)
            raise

    async def wait_for_ready(self, timeout: float = 5.0) -> None:
        req = agent_pb.SessionRequest(
            request_id=utils.shortuuid("req_"),
            ping=agent_pb.SessionRequest.Ping(),
        )
        await self._send_request(req, timeout=timeout)

    async def fetch_chat_history(self) -> agent_pb.SessionRequest.GetChatHistoryResponse:
        req = agent_pb.SessionRequest(
            request_id=utils.shortuuid("req_"),
            get_chat_history=agent_pb.SessionRequest.GetChatHistory(),
        )
        resp = await self._send_request(req)
        return resp.get_chat_history

    async def fetch_agent_info(self) -> agent_pb.SessionRequest.GetAgentInfoResponse:
        req = agent_pb.SessionRequest(
            request_id=utils.shortuuid("req_"),
            get_agent_info=agent_pb.SessionRequest.GetAgentInfo(),
        )
        resp = await self._send_request(req)
        return resp.get_agent_info

    async def send_message(
        self, text: str, timeout: float = 60.0
    ) -> agent_pb.SessionRequest.SendMessageResponse:
        req = agent_pb.SessionRequest(
            request_id=utils.shortuuid("req_"),
            send_message=agent_pb.SessionRequest.SendMessage(text=text),
        )
        resp = await self._send_request(req, timeout=timeout)
        if resp.error:
            raise RuntimeError(resp.error)
        return resp.send_message
