from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field

from livekit import rtc

from .. import utils
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
    language: str | None
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


class ClientSessionState(BaseModel):
    """Current state of the agent session."""

    agent_state: AgentState
    user_state: UserState
    agent_id: str
    options: dict[str, Any]
    created_at: float


class ChatHistoryResponse(BaseModel):
    """Response containing the agent<>user conversation turns."""

    items: list[ChatMessage]


class AgentInfoResponse(BaseModel):
    """Information about the current agent."""

    id: str
    instructions: str | None
    tools: list[str]
    chat_ctx: list[ChatItem]


class SendMessageRequest(BaseModel):
    """Request to send a message to the agent."""

    text: str


class SendMessageResponse(BaseModel):
    """Response from sending a message to the agent."""

    items: list[ChatItem]


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
        room: rtc.Room,
        room_io: RoomIO,
        *,
        expose_instructions: bool = False,
        stream_events: bool = True,
    ) -> None:
        """
        Initialize the ClientEventsHandler.

        Args:
            session: The AgentSession to expose events from.
            room: The LiveKit room to publish events to.
            room_io: The RoomIO instance for participant context.
            expose_instructions: Whether to include agent instructions in responses.
            stream_events: Whether to automatically stream events to clients.
        """
        self._session = session
        self._room = room
        self._room_io = room_io
        self._expose_instructions = expose_instructions
        self._stream_events = stream_events

        self._text_input_cb: TextInputCallback | None = None
        self._text_stream_handler_registered = False
        self._rpc_handlers_registered = False
        self._event_handlers_registered = False

        self._tasks: set[asyncio.Task[Any]] = set()
        self._started = False

    async def start(self) -> None:
        """Register RPC handlers and subscribe to session events."""
        if self._started:
            return

        self._started = True
        self._register_rpc_handlers()

        if self._stream_events:
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

        self._room.local_participant.register_rpc_method(RPC_GET_SESSION_STATE, self._rpc_get_state)
        self._room.local_participant.register_rpc_method(
            RPC_GET_CHAT_HISTORY, self._rpc_get_history
        )
        self._room.local_participant.register_rpc_method(
            RPC_GET_AGENT_INFO, self._rpc_get_agent_info
        )
        self._room.local_participant.register_rpc_method(RPC_SEND_MESSAGE, self._rpc_send_message)

        self._rpc_handlers_registered = True

    async def _rpc_get_state(self, data: rtc.RpcInvocationData) -> str:
        return await self._handle_get_state(data)

    async def _rpc_get_history(self, data: rtc.RpcInvocationData) -> str:
        return await self._handle_get_history(data)

    async def _rpc_get_agent_info(self, data: rtc.RpcInvocationData) -> str:
        return await self._handle_get_agent_info(data)

    async def _rpc_send_message(self, data: rtc.RpcInvocationData) -> str:
        return await self._handle_send_message(data)

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

    async def _handle_get_state(self, data: rtc.RpcInvocationData) -> str:
        agent = self._session._agent

        state = ClientSessionState(
            agent_state=self._session.agent_state,
            user_state=self._session.user_state,
            agent_id=agent.id if agent else "unknown",
            options=asdict(self._session.options),
            created_at=self._session._started_at or time.time(),
        )

        return state.model_dump_json()

    async def _handle_get_history(self, data: rtc.RpcInvocationData) -> str:
        response = ChatHistoryResponse(items=list(self._session.history.items))
        return response.model_dump_json()

    async def _handle_get_agent_info(self, data: rtc.RpcInvocationData) -> str:
        agent = self._session._agent

        tools: list[str] = []
        chat_ctx_items: list[ChatItem] = []
        if agent:
            tools = _tool_names(agent.tools)
            chat_ctx_items = list(agent.chat_ctx.items)

        response = AgentInfoResponse(
            id=agent.id if agent else "unknown",
            instructions=agent.instructions if agent and self._expose_instructions else None,
            tools=tools,
            chat_ctx=chat_ctx_items,
        )

        return response.model_dump_json()

    async def _handle_send_message(self, data: rtc.RpcInvocationData) -> str:
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
        """
        Initialize the receiver.

        Args:
            room: The LiveKit room to receive events from.
            agent_identity: The identity of the agent participant.
        """
        super().__init__()
        self._room = room
        self._agent_identity = agent_identity
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._room.register_text_stream_handler(TOPIC_CLIENT_EVENTS, self._on_event_stream)

    async def aclose(self) -> None:
        if not self._started:
            return

        self._started = False
        try:
            self._room.unregister_text_stream_handler(TOPIC_CLIENT_EVENTS)
        except ValueError:
            pass

    def _on_event_stream(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity != self._agent_identity:
            return

        asyncio.create_task(self._read_event(reader))

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

    async def fetch_session_state(self) -> ClientSessionState:
        response = await self._room.local_participant.perform_rpc(
            destination_identity=self._agent_identity,
            method=RPC_GET_SESSION_STATE,
            payload="",
        )
        return ClientSessionState.model_validate_json(response)

    async def fetch_chat_history(self) -> ChatHistoryResponse:
        response = await self._room.local_participant.perform_rpc(
            destination_identity=self._agent_identity,
            method=RPC_GET_CHAT_HISTORY,
            payload="",
        )
        return ChatHistoryResponse.model_validate_json(response)

    async def fetch_agent_info(self) -> AgentInfoResponse:
        response = await self._room.local_participant.perform_rpc(
            destination_identity=self._agent_identity,
            method=RPC_GET_AGENT_INFO,
            payload="",
        )
        return AgentInfoResponse.model_validate_json(response)

    async def send_message(self, text: str) -> SendMessageResponse:
        """Send a message to the agent and wait for the response.

        Args:
            text: The message to send.

        Returns:
            SendMessageResponse containing all items generated during the run.
        """
        request = SendMessageRequest(text=text)
        response = await self._room.local_participant.perform_rpc(
            destination_identity=self._agent_identity,
            method=RPC_SEND_MESSAGE,
            payload=request.model_dump_json(),
        )
        return SendMessageResponse.model_validate_json(response)
