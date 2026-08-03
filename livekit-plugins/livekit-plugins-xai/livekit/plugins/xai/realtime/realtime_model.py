import os
import time
from typing import Any

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    ConversationItemDeletedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeAudioConfigOutput,
    RealtimeConversationItemFunctionCall,
    RealtimeSessionCreateRequest,
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

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._xai_model: RealtimeModel = realtime_model
        self._session_connected_at: float = 0.0

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        self._session_connected_at = time.time()
        await super()._run_ws(ws_conn)

    async def aclose(self) -> None:
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

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        # xAI's `conversation.item.added` event always has the previous_item_id as None
        # replace it with the last item in the remote chat context
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
        # Unlike OpenAI, xAI streams partial transcripts through this same event,
        # distinguishing them with a `status` field ("in_progress" for partials,
        # "completed" for the final transcript). The OpenAI base handler ignores
        # `status` and always emits is_final=True, so every partial would be
        # surfaced as a final transcription (and a duplicate conversation item).
        # Emit interim updates for in-progress transcripts and only delegate to the
        # base handler for the final one.
        if getattr(event, "status", None) == "in_progress":
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=event.item_id,
                    transcript=event.transcript,
                    is_final=False,
                ),
            )
            return

        # audio transcription is included when the item is added
        # clear the content before appending the transcript to avoid duplicates
        if remote_item := self._remote_chat_ctx.get(event.item_id):
            if (
                remote_item.item.type == "message"
                and remote_item.item.raw_text_content == event.transcript
            ):
                remote_item.item.content.clear()
        super()._handle_conversion_item_input_audio_transcription_completed(event)
