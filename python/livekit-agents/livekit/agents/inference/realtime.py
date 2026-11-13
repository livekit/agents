from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from .._generated.realtime_options import InferenceGateway
from ..llm._realtime import (
    AudioTranscription,
    NoiseReduction,
    RealtimeModel as _RealtimeModel,
    RealtimeSession,
    Reasoning,
    Tracing,
    Truncation,
    TurnDetection,
    Voice,
)
from ..llm.tool_context import ToolChoice
from ..types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from . import InferenceClass, get_default_inference_url, resolve_credentials

if TYPE_CHECKING:
    from .._generated.realtime_wire import Modalities

__all__ = [
    "RealtimeModel",
    "RealtimeSession",
]


class RealtimeModel(_RealtimeModel):
    """OpenAI-compatible realtime model authenticated through LiveKit Inference."""

    _dialect = "inference"
    _provider_label = "livekit.agents.inference.realtime.RealtimeModel"

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
        voice: Voice | None = None,
        modalities: list[Modalities] | None = None,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[NoiseReduction | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        truncation: NotGivenOr[Truncation | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | None] = NOT_GIVEN,
        max_session_duration: NotGivenOr[timedelta | None] = NOT_GIVEN,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-realtime'")
        api_key, api_secret = resolve_credentials(api_key, api_secret)
        self._gateway = InferenceGateway(
            api_secret=api_secret, provider=provider, inference_class=inference_class
        )
        super().__init__(
            model=model,
            voice=voice,
            modalities=modalities,
            tool_choice=tool_choice,
            base_url=base_url or get_default_inference_url(),
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
            max_response_output_tokens=max_response_output_tokens,
            api_key=api_key,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )

    @classmethod
    def from_model_string(cls, model: str) -> RealtimeModel:
        """A RealtimeModel from ``"provider/model"``."""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"
