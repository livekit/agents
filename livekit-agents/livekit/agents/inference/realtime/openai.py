from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    NoiseReductionType,
    RealtimeAudioInputTurnDetection,
    RealtimeReasoning,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from openai.types.realtime.realtime_audio_config_input import NoiseReduction
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.realtime_session_create_response import Tracing
from openai.types.realtime.realtime_truncation import RealtimeTruncation

from livekit.agents import llm
from livekit.agents.inference._utils import (
    HEADER_INFERENCE_PROVIDER,
    InferenceClass,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
    resolve_credentials,
)
from livekit.agents.log import logger
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils.misc import is_given

from ...llm._realtime.openai import (
    DEFAULT_VOICE,
    RealtimeModel as _RealtimeModel,
    RealtimeSession as _RealtimeSession,
    _DiscardedGeneration,
)
from .._realtime_models import DEPRECATED_REALTIME_MODELS, RealtimeModels, is_duplex_model

_XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription(model="grok-transcribe")
_XAI_DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
    interrupt_response=True,
)


@dataclass
class _InferenceOptions:
    provider: str | None
    api_key: str
    api_secret: str
    is_xai: bool = False


class RealtimeModel(_RealtimeModel):
    """OpenAI-compatible realtime model authenticated through LiveKit Inference."""

    def __init__(
        self,
        model: RealtimeModels | str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: NotGivenOr[InferenceClass | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            NoiseReductionType | NoiseReduction | None
        ] = NOT_GIVEN,
        turn_detection: NotGivenOr[RealtimeAudioInputTurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        truncation: NotGivenOr[RealtimeTruncation | None] = NOT_GIVEN,
        reasoning: NotGivenOr[RealtimeReasoning | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            provider (str, optional): Host that must serve the model, e.g. ``"azure"`` or
                ``"openai"`` for OpenAI models. The session is refused if no host in the
                region matches.
        """
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-realtime'")
        if is_duplex_model(model):
            raise ValueError(f"{model} is a duplex model; use inference.DuplexModel")
        if replacement := DEPRECATED_REALTIME_MODELS.get(model):
            logger.warning(f"{model} is deprecated, use {replacement} instead")
        if is_given(inference_class):
            logger.warning(
                "inference_class is deprecated and will be removed in v2.0. "
                "It has no effect on realtime sessions."
            )

        resolved_api_key, resolved_api_secret = resolve_credentials(api_key, api_secret)

        is_xai = model.startswith("xai/")
        resolved_voice = voice if is_given(voice) else "eve" if is_xai else DEFAULT_VOICE
        resolved_transcription = input_audio_transcription
        resolved_turn_detection = turn_detection
        # Preserve whether xAI's server VAD is a default the framework may disable.
        can_disable_turn_detection = not is_given(turn_detection)
        if is_xai:
            if not is_given(resolved_transcription):
                resolved_transcription = _XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION
            if not is_given(resolved_turn_detection):
                resolved_turn_detection = _XAI_DEFAULT_TURN_DETECTION

        super().__init__(
            model=model,
            voice=resolved_voice,
            modalities=modalities,
            input_audio_transcription=resolved_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=resolved_turn_detection,
            tool_choice=tool_choice,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
            api_key="livekit-inference",
            base_url=base_url or get_default_inference_url(),
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )
        # LiveKit Inference always uses the OpenAI-compatible protocol; ambient Azure
        # settings must not change its URL or session wire format.
        self._opts.is_azure = False
        self._opts.api_version = None
        if is_xai:
            self._capabilities.can_disable_turn_detection = can_disable_turn_detection
        self._inference_opts = _InferenceOptions(
            provider=provider,
            api_key=resolved_api_key,
            api_secret=resolved_api_secret,
            is_xai=is_xai,
        )
        self._provider_label = "LiveKit Inference Realtime"

    @classmethod
    def from_model_string(cls, model: str) -> RealtimeModel:
        """Create a RealtimeModel instance from a model string"""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        sess = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess


class RealtimeSession(_RealtimeSession):
    # the base __init__ resets the input turn state before the subclass body runs
    _is_xai: bool = False
    _pending_transcription: ConversationItemInputAudioTranscriptionCompletedEvent | None = None
    _response_spoke: bool = False

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        turn_detection_disabled: bool = False,
    ) -> None:
        self._inference_model = realtime_model
        self._is_xai = realtime_model._inference_opts.is_xai
        super().__init__(realtime_model, turn_detection_disabled=turn_detection_disabled)

    def _reset_input_turn_state(self) -> None:
        if self._is_xai:
            self._flush_input_transcription()
        super()._reset_input_turn_state()
        self._response_spoke = False

    async def aclose(self) -> None:
        if self._is_xai:
            self._flush_input_transcription()
        await super().aclose()

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        opts = self._inference_model._inference_opts
        url, _ = super()._create_ws_url_and_headers()
        headers = get_inference_headers()
        headers["Authorization"] = f"Bearer {create_access_token(opts.api_key, opts.api_secret)}"
        if opts.provider:
            headers[HEADER_INFERENCE_PROVIDER] = opts.provider
        return url, headers

    def _wrap_session_update(self, event_id: str, session: Any) -> dict[str, Any]:
        event = super()._wrap_session_update(event_id, session)
        if hasattr(event, "model_dump"):
            event = event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
        else:
            event = dict(event)

        session_payload = event.get("session")
        if isinstance(session_payload, dict):
            session_payload.pop("model", None)
        return event

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        pending = self._pending_transcription if self._is_xai else None
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
        if self._is_xai:
            self._discard_abandoned_response()

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        if self._is_xai:
            self._discard_abandoned_response()
            self._close_current_generation()
            self._response_spoke = False
        super()._handle_response_created(event)

    def _handle_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        if not self._is_xai:
            super()._handle_input_audio_buffer_speech_started(event)
            return

        if self._pending_transcription and self._pending_transcription.item_id != event.item_id:
            self._flush_input_transcription()

        started_at = self._input_speech_started_at.get(event.item_id)
        super()._handle_input_audio_buffer_speech_started(event)
        if started_at is not None:
            self._input_speech_started_at[event.item_id] = started_at

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        if self._is_xai:
            if event.previous_item_id and not self._remote_chat_ctx.get(event.previous_item_id):
                logger.warning(
                    "xAI placed a chat history item after an unknown item; appended it instead",
                    extra={"item_id": event.item.id, "previous_item_id": event.previous_item_id},
                )
                event.previous_item_id = None
            if event.previous_item_id is None:
                event.previous_item_id = self._remote_chat_ctx.tail_id
        super()._handle_conversion_item_added(event)

    def _handle_conversion_item_deleted(self, event: ConversationItemDeletedEvent) -> None:
        if self._is_xai and event.item_id == "" and self._item_delete_future:
            event.item_id = list(self._item_delete_future.keys())[0]
        super()._handle_conversion_item_deleted(event)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        if not self._is_xai:
            super()._handle_conversion_item_input_audio_transcription_completed(event)
            return

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
        if self._is_xai:
            self._response_spoke = True
            self._flush_input_transcription()
        super()._handle_response_audio_delta(event)

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        if self._is_xai:
            self._response_spoke = True
            self._flush_input_transcription()
        super()._handle_response_text_delta(event)

    def _is_fatal_error(self, error: object | None) -> bool:
        fatal = {
            "unsupported_transcription_model",
            "unsupported_audio_transport",
            "unsupported_audio_format",
        }
        code = getattr(error, "code", None)
        type_ = getattr(error, "type", None)
        return (
            (isinstance(code, str) and code in fatal)
            or (isinstance(type_, str) and type_ in fatal)
            or super()._is_fatal_error(error)
        )
