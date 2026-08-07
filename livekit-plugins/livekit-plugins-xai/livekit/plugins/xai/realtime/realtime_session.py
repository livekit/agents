from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    RealtimeConversationItemFunctionCall,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit.agents import llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

from ..log import logger
from ..tools import XAITool
from ._session_update import lift_xai_session_fields

if TYPE_CHECKING:
    from .realtime_model import RealtimeModel


class RealtimeSession(openai.realtime.RealtimeSession):
    """xAI Realtime Session: built-in tools, transcript hold, and force_message say()."""

    _pending_transcription: ConversationItemInputAudioTranscriptionCompletedEvent | None = None
    _response_spoke: bool = False
    # instance attribute; annotated here so __new__ test doubles can assign it
    _pending_say_event_ids: deque[str]

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._xai_model: RealtimeModel = realtime_model
        self._session_connected_at: float = 0.0
        self._pending_say_event_ids = deque()
        self.on("openai_server_event_received", self._on_xai_server_event)

    async def _run_ws(self, ws_conn: Any) -> None:
        self._session_connected_at = time.time()
        await super()._run_ws(ws_conn)

    def _reset_input_turn_state(self) -> None:
        self._flush_input_transcription()
        super()._reset_input_turn_state()
        self._response_spoke = False

    async def aclose(self) -> None:
        self._flush_input_transcription()
        if self._session_connected_at > 0:
            self.emit(
                "metrics_collected",
                RealtimeModelMetrics(
                    timestamp=time.time(),
                    request_id="session_close",
                    session_duration=time.time() - self._session_connected_at,
                    input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                    output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                    metadata=Metadata(
                        model_name=self._xai_model.model,
                        model_provider=self._xai_model.provider,
                    ),
                ),
            )
        await super().aclose()

    def _on_xai_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "conversation.item.input_audio_transcription.updated":
            item_id = event.get("item_id") or ""
            transcript = event.get("transcript") or ""
            if item_id and transcript:
                self.emit(
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=item_id, transcript=transcript, is_final=False
                    ),
                )
        elif event_type == "input_audio_buffer.timeout_triggered":
            logger.debug("xAI idle timeout triggered; server will start a proactive turn")
        elif event_type == "session.created":
            if model := (event.get("session") or {}).get("model"):
                logger.debug("xAI session created", extra={"model": model})

    def _wrap_session_update(
        self, event_id: str, session: RealtimeSessionCreateRequest
    ) -> SessionUpdateEvent | dict[str, Any]:
        lift_xai_session_fields(session)
        return super()._wrap_session_update(event_id=event_id, session=session)

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        event = super()._create_tools_update_event(tools)
        xai_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, XAITool):
                xai_tools.append(tool.to_dict())
        event["session"]["tools"] += xai_tools
        return event

    def _handle_function_call(self, item: RealtimeConversationItemFunctionCall) -> None:
        if not self._tools.get_function_tool(item.name):
            logger.warning(f"unknown function tool: {item.name}, ignoring")
            return
        super()._handle_function_call(item)

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        pending = self._pending_transcription
        node = self._remote_chat_ctx.get(pending.item_id) if pending else None
        if node is not None and chat_ctx.get_by_id(node.item.id) is None:
            chat_ctx = chat_ctx.copy()
            index, previous = 0, node._prev
            while previous is not None:
                if (at := chat_ctx.index_by_id(previous.item.id)) is not None:
                    index = at + 1
                    break
                previous = previous._prev
            chat_ctx.items.insert(index, node.item.model_copy())
        return super()._create_update_chat_ctx_events(chat_ctx)

    def _discard_abandoned_response(self) -> None:
        generation = self._current_generation
        if (
            generation is None
            or self._response_spoke
            or isinstance(generation, _DiscardedGeneration)
        ):
            return
        logger.debug("discarding the response xAI left in flight")
        self._close_current_generation()
        self._current_generation = _DiscardedGeneration()

    def interrupt(self) -> None:
        super().interrupt()
        self._discard_abandoned_response()

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Speak scripted text via xAI ``force_message`` (no ``response.create``)."""
        event_id = utils.shortuuid("say_")
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        self._response_created_futures[event_id] = fut
        # armed only after force_message is sent (covers server RTT, not text collection)
        timeout_handle: list[asyncio.TimerHandle | None] = [None]

        def _clear_pending() -> None:
            self._response_created_futures.pop(event_id, None)
            try:
                self._pending_say_event_ids.remove(event_id)
            except ValueError:
                pass

        def _on_timeout() -> None:
            _clear_pending()
            if not fut.done():
                self._discarded_event_ids.add(event_id)
                fut.set_exception(llm.RealtimeError("say timed out."))

        def _on_fut_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            if timeout_handle[0] is not None:
                timeout_handle[0].cancel()
            _clear_pending()
            if f.cancelled():
                # force_message may already be in flight; cancel server-side and discard by id
                self.send_event(ResponseCancelEvent(type="response.cancel"))
                self._discarded_event_ids.add(event_id)

        fut.add_done_callback(_on_fut_done)

        async def _send() -> None:
            full_text = text if isinstance(text, str) else "".join([c async for c in text])
            if fut.done():
                return
            self.send_event(
                {
                    "type": "conversation.item.create",
                    "event_id": event_id,
                    "item": {
                        "type": "force_message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": full_text}],
                    },
                }
            )
            if fut.done():
                self.send_event(ResponseCancelEvent(type="response.cancel"))
                self._discarded_event_ids.add(event_id)
                return
            # only tag response.created after the force_message is on the wire (FIFO)
            self._pending_say_event_ids.append(event_id)
            timeout_handle[0] = asyncio.get_event_loop().call_later(10.0, _on_timeout)

        task = asyncio.create_task(_send(), name="xai-say")

        def _on_send_done(t: asyncio.Task[None]) -> None:
            if not t.cancelled() and (exc := t.exception()) is not None and not fut.done():
                _clear_pending()
                fut.set_exception(exc)

        task.add_done_callback(_on_send_done)
        return fut

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        # force_message omits client_event_id; attach the oldest post-send say id
        if self._pending_say_event_ids and not (
            isinstance(event.response.metadata, dict)
            and event.response.metadata.get("client_event_id")
        ):
            if not isinstance(event.response.metadata, dict):
                event.response.metadata = {}
            event.response.metadata["client_event_id"] = self._pending_say_event_ids.popleft()

        self._discard_abandoned_response()
        self._close_current_generation()
        self._response_spoke = False
        super()._handle_response_created(event)

    def _handle_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        if self._pending_transcription and self._pending_transcription.item_id != event.item_id:
            self._flush_input_transcription()

        started_at = self._input_speech_started_at.get(event.item_id)
        super()._handle_input_audio_buffer_speech_started(event)
        if started_at is not None:
            self._input_speech_started_at[event.item_id] = started_at

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        if event.previous_item_id and not self._remote_chat_ctx.get(event.previous_item_id):
            logger.warning(
                "xAI anchored an item to one it never announced, appending it instead",
                extra={"item_id": event.item.id, "previous_item_id": event.previous_item_id},
            )
            event.previous_item_id = None

        if event.previous_item_id is None:
            event.previous_item_id = (
                self._remote_chat_ctx._tail.item.id if self._remote_chat_ctx._tail else None
            )

        super()._handle_conversion_item_added(event)

    def _handle_conversion_item_deleted(self, event: ConversationItemDeletedEvent) -> None:
        if event.item_id == "" and self._item_delete_future:
            event.item_id = list(self._item_delete_future.keys())[0]
        super()._handle_conversion_item_deleted(event)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        if getattr(event, "status", None) != "in_progress":
            if self._pending_transcription and self._pending_transcription.item_id != event.item_id:
                self._flush_input_transcription()
            self._pending_transcription = event
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id, transcript=event.transcript, is_final=False
            ),
        )

    def _flush_input_transcription(self) -> None:
        if (event := self._pending_transcription) is None:
            return
        self._pending_transcription = None
        if (remote_item := self._remote_chat_ctx.get(event.item_id)) and (
            remote_item.item.type == "message"
        ):
            remote_item.item.content = [
                c for c in remote_item.item.content if not isinstance(c, str)
            ]
        super()._handle_conversion_item_input_audio_transcription_completed(event)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        self._response_spoke = True
        self._flush_input_transcription()
        super()._handle_response_audio_delta(event)

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        self._response_spoke = True
        self._flush_input_transcription()
        super()._handle_response_text_delta(event)
