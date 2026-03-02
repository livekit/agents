from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field

from livekit import rtc

from .. import utils
from ..language import Language
from ..llm import (
    ChatItem,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RawFunctionTool,
    Toolset,
)
from ..log import logger
from ..metrics import AgentMetrics
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
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .room_io import RoomIO
    from .room_io.types import TextInputCallback


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
    language: Language | None
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


ClientEvent = Annotated[
    ClientAgentStateChangedEvent
    | ClientUserStateChangedEvent
    | ClientConversationItemAddedEvent
    | ClientUserInputTranscribedEvent
    | ClientFunctionToolsExecutedEvent
    | ClientMetricsCollectedEvent
    | ClientErrorEvent,
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


class SendMessageRequest(BaseModel):
    text: str


class SendMessageResponse(BaseModel):
    items: list[ChatItem]


# Text stream request/response protocol (no size limit unlike RPC)
class StreamRequest(BaseModel):
    """Request sent via text stream."""

    request_id: str
    method: str
    payload: str  # JSON-encoded method-specific request


class StreamResponse(BaseModel):
    """Response sent via text stream."""

    request_id: str
    payload: str  # JSON-encoded method-specific response
    error: str | None = None


def _tool_names(tools: list[Any]) -> list[str]:
    result: list[str] = []
    for tool in tools:
        if isinstance(tool, (FunctionTool, RawFunctionTool)):
            result.append(tool.info.name)
        elif isinstance(tool, Toolset):
            result.extend(_tool_names(tool.tools))
    return result


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
        """Cleanup and unregister handlers."""
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
            self._session.off("user_input_transcribed", self._on_user_input_transcribed)
            self._session.off("error", self._on_error)
            self._event_handlers_registered = False

        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    def register_text_input(self, text_input_cb: TextInputCallback) -> None:
        """
        Register a text input handler to receive messages from clients.

        Args:
            text_input_cb: Callback function to handle incoming text messages.
        """
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
        """Register text stream handler for requests (no size limit unlike RPC)."""
        if self._request_handler_registered:
            return

        try:
            self._room.register_text_stream_handler(TOPIC_AGENT_REQUEST, self._on_stream_request)
            self._request_handler_registered = True
        except ValueError:
            logger.warning(f"text stream handler for topic '{TOPIC_AGENT_REQUEST}' already set")

    def _on_stream_request(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        """Handle incoming text stream requests."""
        task = asyncio.create_task(self._handle_stream_request(reader, participant_identity))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @utils.log_exceptions(logger=logger)
    async def _handle_stream_request(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ) -> None:
        """Process a text stream request and send response."""
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
                elif request.method == "send_message":
                    response_payload = await self._stream_send_message(request.payload)
                else:
                    response_payload = ""
                    error = f"Unknown method: {request.method}"
            except Exception as e:
                response_payload = ""
                error = str(e)

            # Send response
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
        """Handle get_session_state via text stream."""
        agent = self._session.current_agent
        response = GetSessionStateResponse(
            agent_state=self._session.agent_state,
            user_state=self._session.user_state,
            agent_id=agent.id,
            options=asdict(self._session.options),
            created_at=self._session._started_at or time.time(),
        )
        return response.model_dump_json()

    async def _stream_get_chat_history(self, payload: str) -> str:
        """Handle get_chat_history via text stream."""
        response = GetChatHistoryResponse(items=list(self._session.history.items))
        return response.model_dump_json()

    async def _stream_get_agent_info(self, payload: str) -> str:
        """Handle get_agent_info via text stream (no size limit)."""
        agent = self._session.current_agent
        response = GetAgentInfoResponse(
            id=agent.id,
            instructions=agent.instructions,
            tools=_tool_names(agent.tools),
            chat_ctx=list(agent.chat_ctx.items),
        )
        return response.model_dump_json()

    async def _stream_send_message(self, payload: str) -> str:
        """Handle send_message via text stream."""
        from .run_result import RunResult

        request = SendMessageRequest.model_validate_json(payload)
        run_result: RunResult[None] = await self._session.run(user_input=request.text)

        items: list[ChatItem] = []
        for event in run_result.events:
            items.append(event.item)

        response = SendMessageResponse(items=items)
        return response.model_dump_json()

    def _register_event_handlers(self) -> None:
        if self._event_handlers_registered:
            return

        self._session.on("agent_state_changed", self._on_agent_state_changed)
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("conversation_item_added", self._on_conversation_item_added)
        self._session.on("function_tools_executed", self._on_function_tools_executed)
        self._session.on("metrics_collected", self._on_metrics_collected)
        self._session.on("user_input_transcribed", self._on_user_input_transcribed)
        self._session.on("error", self._on_error)

        self._event_handlers_registered = True

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

        client_event = ClientMetricsCollectedEvent(
            metrics=event.metrics,
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _on_error(self, event: ErrorEvent) -> None:
        client_event = ClientErrorEvent(
            message=str(event.error) if event.error else "Unknown error",
            created_at=event.created_at,
        )
        self._stream_client_event(client_event)

    def _get_target_identities(self) -> list[str] | None:
        """Get the identities of participants that should receive client events.

        Returns the linked RoomIO participant if it has the required permissions,
        or None if no valid target is available.
        """
        linked = self._room_io.linked_participant
        if linked is None:
            return None

        # TODO(permissions): check linked.permissions.can_subscribe_metrics
        # once the rtc SDK exposes participant permissions
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
            options=asdict(self._session.options),
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

        request = SendMessageRequest.model_validate_json(data.payload)
        run_result: RunResult[None] = await self._session.run(user_input=request.text)

        items: list[ChatItem] = []
        for event in run_result.events:
            items.append(event.item)

        response = SendMessageResponse(items=items)
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
    "error",
]


class RemoteSession(rtc.EventEmitter[RemoteSessionEventTypes]):
    """
    Client-side interface to interact with a remote AgentSession.

    This class allows frontends/clients to:
    - Subscribe to real-time events from the agent session
    - Query session state, chat history, and agent info via RPC
    - Send messages to the agent

    Example:
        ```python
        session = RemoteSession(room, agent_identity="agent")
        session.on("agent_state_changed", lambda ev: print(f"Agent state: {ev.new_state}"))
        session.on("user_state_changed", lambda ev: print(f"User state: {ev.new_state}"))
        session.on("conversation_item_added", lambda ev: print(f"Message: {ev.item}"))
        await session.start()

        # Query current state
        state = await session.fetch_session_state()
        history = await session.fetch_chat_history()

        # Send a message and get all generated items
        response = await session.send_message("Hello!")
        for item in response.items:
            print(f"Item: {item}")
        ```
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

        # Cancel pending requests
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
        """Read and dispatch a response to the waiting request."""
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
        """Send a request via text stream and wait for response."""
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

    async def send_message(self, text: str, response_timeout: float = 60.0) -> SendMessageResponse:
        request = SendMessageRequest(text=text)
        response = await self._send_request(
            method="send_message",
            payload=request.model_dump_json(),
            timeout=response_timeout,
        )
        return SendMessageResponse.model_validate_json(response)
