import os
import time
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
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit.agents import llm
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

XAI_DEFAULT_MODEL: GrokRealtimeModels = "grok-voice-think-fast-1.0"

XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription()

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

        resolved_voice = voice if is_given(voice) else "Ara"
        super().__init__(
            base_url=base_url if is_given(base_url) else XAI_BASE_URL,
            model=model if is_given(model) else XAI_DEFAULT_MODEL,
            voice=resolved_voice,  # type: ignore[arg-type]
            api_key=api_key,
            modalities=["audio", "text"],
            input_audio_transcription=XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
            turn_detection=turn_detection
            if is_given(turn_detection)
            else XAI_DEFAULT_TURN_DETECTION,
            http_session=http_session if is_given(http_session) else None,
            max_session_duration=max_session_duration if is_given(max_session_duration) else None,
            conn_options=conn_options,
        )
        self._capabilities.per_response_tool_choice = False
        # client turn-taking is not stable during testing, mark it as unsupported for now
        self._capabilities.can_disable_turn_detection = False
        self._provider_label = "xAI Realtime API"

    def session(self, *, turn_detection_disabled: bool = False) -> "RealtimeSession":
        # manual turn-taking is unsupported (can_disable_turn_detection=False)
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(openai.realtime.RealtimeSession):
    """xAI Realtime Session that supports xAI built-in tools."""

    # the base __init__ resets the input turn state before the subclass body runs
    _pending_transcription: ConversationItemInputAudioTranscriptionCompletedEvent | None = None
    _response_spoke: bool = False

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._xai_model: RealtimeModel = realtime_model
        self._session_connected_at: float = 0.0

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        self._session_connected_at = time.time()
        await super()._run_ws(ws_conn)

    def _reset_input_turn_state(self) -> None:
        # a reconnect voids every item id, so deliver the held transcript first
        self._flush_input_transcription()
        super()._reset_input_turn_state()
        self._response_spoke = False

    async def aclose(self) -> None:
        # nothing else ends the turn now, and the transcript must not go with the session
        self._flush_input_transcription()

        # emit session duration metrics before closing (for xAI's per-minute billing)
        if self._session_connected_at > 0:
            session_duration = time.time() - self._session_connected_at
            metrics = RealtimeModelMetrics(
                timestamp=time.time(),
                request_id="session_close",
                session_duration=session_duration,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._xai_model.model,
                    model_provider=self._xai_model.provider,
                ),
            )
            self.emit("metrics_collected", metrics)
        await super().aclose()

    def _wrap_session_update(
        self, event_id: str, session: RealtimeSessionCreateRequest
    ) -> SessionUpdateEvent | dict[str, Any]:
        # xAI expects `voice` and `turn_detection` as top-level session fields, whereas the
        # OpenAI base nests them under audio.output / audio.input. Relocate them on the typed
        # request (RealtimeSessionCreateRequest is extra="allow") before the base serializes,
        # so this single seam covers every session.update the base emits — initial config,
        # reconnect, and update_options — with no duplicated logic and no extra events.
        # Gating on model_fields_set keeps an untouched field from leaking as a top-level null.
        audio = session.audio
        if isinstance(audio, RealtimeAudioConfig):
            output = audio.output
            if isinstance(output, RealtimeAudioConfigOutput) and "voice" in output.model_fields_set:
                # voice/turn_detection are set as extra fields (the model is extra="allow")
                session.voice = output.voice  # type: ignore[attr-defined]
                output.model_fields_set.discard("voice")
            audio_input = audio.input
            if (
                isinstance(audio_input, RealtimeAudioConfigInput)
                and "turn_detection" in audio_input.model_fields_set
            ):
                session.turn_detection = audio_input.turn_detection  # type: ignore[attr-defined]
                audio_input.model_fields_set.discard("turn_detection")

            # if the relocation emptied both sub-blocks (a voice/turn_detection-only update),
            # drop the now-hollow audio key instead of sending audio={"input":{},"output":{}}
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

        # inject supported Toolset
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
        # stand the mirror's item in for a held turn, or the diff drops it and moves the reply
        # ahead of it
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
        """Drop a response that xAI killed before it spoke, which nothing else reports."""
        generation = self._current_generation
        if (
            generation is None
            or self._response_spoke
            or isinstance(generation, _DiscardedGeneration)
        ):
            return

        logger.debug("discarding the response xAI left in flight")
        self._close_current_generation()
        # anything still in flight for it must not land on the response that follows
        self._current_generation = _DiscardedGeneration()

    def interrupt(self) -> None:
        # the cancel names no response, so whichever one was in flight is over either way
        super().interrupt()
        self._discard_abandoned_response()

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        # a response that finished clears itself, so anything left here was abandoned
        self._discard_abandoned_response()
        self._close_current_generation()
        self._response_spoke = False
        super()._handle_response_created(event)

    def _handle_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        # xAI reuses the item across a pause, so a different id means the held turn is over
        if self._pending_transcription and self._pending_transcription.item_id != event.item_id:
            self._flush_input_transcription()

        # the turn began at its first speech segment, so a resumed one must not restamp it
        started_at = self._input_speech_started_at.get(event.item_id)
        super()._handle_input_audio_buffer_speech_started(event)
        if started_at is not None:
            self._input_speech_started_at[event.item_id] = started_at

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        # xAI omits previous_item_id, or names an item it never announced, so append rather
        # than lose the item and every later one anchored to it
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
        # xAI's `conversation.item.deleted` event has item_id empty
        # assuming it's the first item in the deletion list
        if event.item_id == "" and self._item_delete_future:
            event.item_id = list(self._item_delete_future.keys())[0]

        super()._handle_conversion_item_deleted(event)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        # xAI re-transcribes the whole item at every commit, and the item outlives a pause,
        # so only the last commit of a turn carries its final
        if getattr(event, "status", None) != "in_progress":
            if self._pending_transcription and self._pending_transcription.item_id != event.item_id:
                self._flush_input_transcription()
            self._pending_transcription = event

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.transcript,
                is_final=False,
            ),
        )

    def _flush_input_transcription(self) -> None:
        """Deliver the held transcript as the single final of its item."""
        if (event := self._pending_transcription) is None:
            return

        self._pending_transcription = None

        # the transcript covers the item from the start, so it replaces the mirrored text
        if (remote_item := self._remote_chat_ctx.get(event.item_id)) and (
            remote_item.item.type == "message"
        ):
            remote_item.item.content = [
                content for content in remote_item.item.content if not isinstance(content, str)
            ]
        super()._handle_conversion_item_input_audio_transcription_completed(event)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        # xAI drops a response the user talks over without a word, so its first output is the
        # only proof that the turn ended
        self._response_spoke = True
        self._flush_input_transcription()
        super()._handle_response_audio_delta(event)

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        # the same boundary, for a response that answers in text
        self._response_spoke = True
        self._flush_input_transcription()
        super()._handle_response_text_delta(event)
