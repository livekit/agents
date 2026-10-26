from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ... import _core
from ...llm import realtime as llm_realtime
from ...llm._realtime.openai import RealtimeModel as _RealtimeModel, RealtimeSession
from ...llm.tool_context import ToolChoice
from ...types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .._utils import (
    InferenceClass,
    get_default_inference_url,
    get_inference_headers,
    resolve_credentials,
)

if TYPE_CHECKING:
    import aiohttp

__all__ = ["RealtimeModel", "RealtimeSession"]


class RealtimeModel(_RealtimeModel):
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
        input_audio_transcription: NotGivenOr[Any] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Any] = NOT_GIVEN,
        truncation: NotGivenOr[Any] = NOT_GIVEN,
        reasoning: NotGivenOr[Any] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-realtime'")
        api_key, api_secret = resolve_credentials(api_key, api_secret)
        self._inference_class = inference_class
        self._native = _core.NativeRealtimeModel(
            dialect="inference",
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url or get_default_inference_url(),
            model=model,
            voice=voice or None,
            modalities=modalities or None,
            inference_provider=provider,
            inference_class=inference_class,
            timeout=conn_options.timeout,
            max_retry=conn_options.max_retry,
            retry_interval=conn_options.retry_interval,
            label=f"{type(self).__module__}.{type(self).__name__}",
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
            max_session_duration=max_session_duration,
        )
        llm_realtime.RealtimeModel.__init__(self, capabilities=self.capabilities)

    @classmethod
    def from_model_string(cls, model: str) -> RealtimeModel:
        """Create a RealtimeModel instance from a model string"""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"

    def _core_native(self) -> _core.NativeRealtimeModel:
        """The model, identified as the current job's, as a ``LiveSession`` drives it."""
        headers = get_inference_headers(inference_class=self._inference_class)
        self._native.bind_identification(list(headers.items()))
        return self._native
