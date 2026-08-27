from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import AsyncIterable
from typing import Any

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeAudioConfigOutput,
    RealtimeConversationItemFunctionCall,
    RealtimeReasoning,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit.agents import llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

from ..log import logger
from ..tools import XAITool
from ..types import GrokRealtimeModels, GrokVoices

XAI_BASE_URL = "wss://api.x.ai/v1/realtime"

XAI_DEFAULT_MODEL: GrokRealtimeModels = "grok-voice-latest"

XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription(model="grok-transcribe")

XAI_DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
    interrupt_response=True,
)


class RealtimeModel(openai.realtime.RealtimeModel):
    def __init__(
        self,
        *,
        model: NotGivenOr[GrokRealtimeModels | str] = NOT_GIVEN,
        voice: NotGivenOr[GrokVoices | str | None] = "Ara",
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        reasoning: NotGivenOr[RealtimeReasoning | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the XAI_API_KEY environment variable"
            )

        # resolve NotGivenOr values before super().__init__ so mypy does not explode
        # on the OpenAI overload union combinations
        resolved_base_url = base_url if is_given(base_url) else XAI_BASE_URL
        resolved_model = model if is_given(model) else XAI_DEFAULT_MODEL
        resolved_voice = voice if is_given(voice) else "Ara"
        resolved_transcription = (
            input_audio_transcription
            if is_given(input_audio_transcription)
            else XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION
        )
        resolved_turn_detection = (
            turn_detection if is_given(turn_detection) else XAI_DEFAULT_TURN_DETECTION
        )
        resolved_max_session_duration = (
            max_session_duration if is_given(max_session_duration) else None
        )
        init_kwargs: dict = {
            "base_url": resolved_base_url,
            "model": resolved_model,
            "voice": resolved_voice,
            "api_key": api_key,
            "modalities": ["audio", "text"],
            "input_audio_transcription": resolved_transcription,
            "turn_detection": resolved_turn_detection,
            "http_session": http_session,
            "max_session_duration": resolved_max_session_duration,
            "conn_options": conn_options,
        }
        if is_given(reasoning):
            init_kwargs["reasoning"] = reasoning
        if is_given(speed):
            init_kwargs["speed"] = speed
        super().__init__(**init_kwargs)
        self._capabilities.per_response_tool_choice = False
        # client turn-taking is not stable during testing, mark it as unsupported for now
        self._capabilities.can_disable_turn_detection = False
        # xAI force_message drives scripted TTS without a follow-up response.create
        self._capabilities.supports_say = True
        self._provider_label = "xAI Realtime API"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # manual turn-taking is unsupported (can_disable_turn_detection=False)
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(openai.realtime.RealtimeSession):
    """xAI Realtime Session that supports xAI built-in tools and force_message say()."""

    _pending_transcription: ConversationItemInputAudioTranscriptionCompletedEvent | None = None
    _response_spoke: bool = False
    # instance attributes; annotated here so __new__ test doubles can assign them
    _pending_say_event_ids: deque[str]
    _say_tasks: set[asyncio.Task[None]]

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._xai_model: RealtimeModel = realtime_model
        self._session_connected_at: float = 0.0
        self._pending_say_event_ids = deque()
        self._say_tasks = set()
        self.on("openai_server_event_received", self._on_xai_server_event)

    async def _run_ws(self, ws_conn: Any) -> None:
        self._session_connected_at = time.time()
        await super()._run_ws(ws_conn)

    def _reset_input_turn_state(self) -> None:
        self._flush_input_transcription()
        super()._reset_input_turn_state()
        self._response_spoke = False

    async def aclose(self) -> None:
        tasks = list(self._say_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
        # xAI expects voice/turn_detection as top-level session fields
        audio = session.audio
        if isinstance(audio, RealtimeAudioConfig):
            output = audio.output
            if isinstance(output, RealtimeAudioConfigOutput) and "voice" in output.model_fields_set:
                session.voice = output.voice  # type: ignore[attr-defined]
                output.model_fields_set.discard("voice")
            audio_input = audio.input
            if (
                isinstance(audio_input, RealtimeAudioConfigInput)
                and "turn_detection" in audio_input.model_fields_set
            ):
                session.turn_detection = audio_input.turn_detection  # type: ignore[attr-defined]
                audio_input.model_fields_set.discard("turn_detection")
            out_set = isinstance(output, RealtimeAudioConfigOutput) and bool(
                output.model_fields_set
            )
            in_set = isinstance(audio_input, RealtimeAudioConfigInput) and bool(
                audio_input.model_fields_set
            )
            if not out_set and not in_set:
                session.model_fields_set.discard("audio")
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

        task = asyncio.create_task(self._say_task(event_id, text, fut), name="xai-say")
        self._say_tasks.add(task)
        task.add_done_callback(self._say_tasks.discard)
        return fut

    async def _say_task(
        self,
        event_id: str,
        text: str | AsyncIterable[str],
        fut: asyncio.Future[llm.GenerationCreatedEvent],
    ) -> None:
        """Collect text, send force_message, then wait for response.created (or timeout/cancel)."""
        force_message_sent = False
        try:
            full_text = text if isinstance(text, str) else "".join([c async for c in text])
            if fut.done():
                self._response_created_futures.pop(event_id, None)
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
            force_message_sent = True
            # only tag response.created after the force_message is on the wire (FIFO)
            self._ensure_pending_say_tag(event_id)

            if fut.done():
                # cancelled during send: keep the tag for discard-by-id
                if fut.cancelled():
                    self._discard_say(event_id)
                else:
                    self._response_created_futures.pop(event_id, None)
                return

            # timeout covers server RTT only — text collection is already done.
            # use wait() so caller cancel of fut does not CancelledError this task.
            done, _ = await asyncio.wait({fut}, timeout=10.0)
            if not done:
                self._response_created_futures.pop(event_id, None)
                self._discarded_event_ids.add(event_id)
                self._ensure_pending_say_tag(event_id)
                self._schedule_stale_say_cleanup(event_id)
                if not fut.done():
                    fut.set_exception(llm.RealtimeError("say timed out."))
            elif fut.cancelled():
                self._discard_say(event_id)
            else:
                # success or send-path exception: tag already consumed on success
                self._drop_pending_say_tag(event_id)
        except asyncio.CancelledError:
            # aclose() cancels _say_task; always resolve fut so callers do not hang
            self._response_created_futures.pop(event_id, None)
            if force_message_sent:
                self._discard_say(event_id)
            if not fut.done():
                fut.cancel()
            raise
        except Exception as exc:
            self._response_created_futures.pop(event_id, None)
            self._drop_pending_say_tag(event_id)
            if not fut.done():
                fut.set_exception(exc)

    def _ensure_pending_say_tag(self, event_id: str) -> None:
        if event_id not in self._pending_say_event_ids:
            self._pending_say_event_ids.append(event_id)

    def _drop_pending_say_tag(self, event_id: str) -> None:
        try:
            self._pending_say_event_ids.remove(event_id)
        except ValueError:
            pass

    def _discard_say(self, event_id: str) -> None:
        """Cancel server-side and keep the id taggable for a late response.created."""
        self._response_created_futures.pop(event_id, None)
        if event_id not in self._discarded_event_ids:
            self.send_event(ResponseCancelEvent(type="response.cancel"))
            self._discarded_event_ids.add(event_id)
            self._schedule_stale_say_cleanup(event_id)
        self._ensure_pending_say_tag(event_id)

    def _schedule_stale_say_cleanup(self, event_id: str) -> None:
        # if the server never emits response.created, drop the tag so it cannot
        # steal a later unrelated reply
        def _cleanup() -> None:
            if event_id in self._discarded_event_ids:
                self._drop_pending_say_tag(event_id)
                self._discarded_event_ids.discard(event_id)

        asyncio.get_event_loop().call_later(10.0, _cleanup)

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
            event.previous_item_id = self._remote_chat_ctx.tail_id

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
