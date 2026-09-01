from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from livekit.agents import llm
from livekit.agents.inference._utils import (
    HEADER_INFERENCE_PROVIDER,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils.misc import is_given
from openai.types.realtime import (
    AudioTranscription,
    NoiseReductionType,
    RealtimeAudioInputTurnDetection,
    RealtimeReasoning,
)
from openai.types.realtime.realtime_audio_config_input import NoiseReduction
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from openai.types.realtime.realtime_session_create_response import Tracing
from openai.types.realtime.realtime_truncation import RealtimeTruncation

from .realtime_model import DEFAULT_VOICE, RealtimeModel, RealtimeSession

InferenceClass = Literal["priority", "standard", "low"]

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
    inference_class: InferenceClass | None


class InferenceRealtimeModel(RealtimeModel):
    """OpenAI-compatible realtime model authenticated through LiveKit Inference."""

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
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
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-realtime'")

        resolved_api_key = api_key or os.getenv(
            "LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", "")
        )
        if not resolved_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
            )

        resolved_api_secret = api_secret or os.getenv(
            "LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", "")
        )
        if not resolved_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
            )

        is_xai = model.startswith("xai/")
        resolved_voice = voice if is_given(voice) else "eve" if is_xai else DEFAULT_VOICE
        resolved_transcription = input_audio_transcription
        resolved_turn_detection = turn_detection
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
        self._inference_opts = _InferenceOptions(
            provider=provider,
            api_key=resolved_api_key,
            api_secret=resolved_api_secret,
            inference_class=inference_class,
        )
        self._provider_label = "LiveKit Inference Realtime"

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self, *, turn_detection_disabled: bool = False) -> InferenceRealtimeSession:
        sess = InferenceRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess


class InferenceRealtimeSession(RealtimeSession):
    def __init__(
        self,
        realtime_model: InferenceRealtimeModel,
        *,
        turn_detection_disabled: bool = False,
    ) -> None:
        self._inference_model = realtime_model
        super().__init__(realtime_model, turn_detection_disabled=turn_detection_disabled)

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        opts = self._inference_model._inference_opts
        url, _ = super()._create_ws_url_and_headers()
        headers = get_inference_headers(inference_class=opts.inference_class)
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

    def _is_fatal_error(self, error: object | None) -> bool:
        code = getattr(error, "code", None) or getattr(error, "type", None)
        return code in {
            "unsupported_transcription_model",
            "unsupported_audio_transport",
            "unsupported_audio_format",
            "invalid_audio_payload",
        } or super()._is_fatal_error(error)
