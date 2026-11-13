from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from .._generated import facades_llm__realtime as _facades, public_llm__realtime as _public
from .._generated.public_llm__realtime import *  # noqa: F403
from ..types import NOT_GIVEN, NotGivenOr
from . import realtime as llm_realtime

if TYPE_CHECKING:
    from .._generated.realtime_wire import Modalities
    from .tool_context import ToolChoice

# kept for plugins that subclass it
RealtimeSession = llm_realtime.RealtimeSession


class _OpenAIProtocolModel(_facades.RealtimeModel):
    """A model speaking the OpenAI realtime protocol. Each provider's model subclasses it and
    declares ``update_options`` with the options its session carries."""

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        raise NotImplementedError(
            f"{type(self).__name__} sessions run in the core; give the model to a VoiceSession"
        )

    async def aclose(self) -> None:
        pass


class RealtimeModel(_OpenAIProtocolModel):
    _provider_label = "livekit.plugins.openai.realtime.realtime_model.RealtimeModel"

    @classmethod
    def with_azure(
        cls,
        *,
        azure_deployment: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        entra_token: str | None = None,
        base_url: str | None = None,
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
    ) -> RealtimeModel:
        """Create a RealtimeModel configured for Azure OpenAI.

        Args:
            azure_deployment: Azure OpenAI deployment name.
            azure_endpoint: Azure endpoint URL; defaults to AZURE_OPENAI_ENDPOINT.
            api_key: Azure API key; defaults to AZURE_OPENAI_API_KEY. Omit with `entra_token`.
            entra_token: Azure Entra token for AAD auth, instead of `api_key`.
            base_url: Explicit base URL, used as-is. Mutually exclusive with `azure_endpoint`.
            voice: Voice used for audio responses.
            modalities: Modalities to enable. Defaults to audio.
            input_audio_transcription: Transcription options; ``whisper-1`` when not given.
            input_audio_noise_reduction: Input noise reduction settings.
            turn_detection: Server-side turn detection; server VAD when not given.
            tool_choice: Tool selection policy.
            speed: Audio playback speed multiplier.
            tracing: Tracing configuration for OpenAI Realtime.
            truncation: How the conversation is truncated when it exceeds the context.
            reasoning: Reasoning config for reasoning-capable models.
            max_response_output_tokens: Maximum output tokens per response; ``None`` is
                unlimited.
            max_session_duration: How long one connection serves before it is recycled.

        Raises:
            ValueError: If credentials are missing, the Azure endpoint cannot be determined, or
                both `base_url` and `azure_endpoint` are given.
        """
        return cls(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            entra_token=entra_token,
            base_url=base_url,
            voice=voice,
            modalities=modalities,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
            max_response_output_tokens=max_response_output_tokens,
            max_session_duration=max_session_duration,
        )

    def update_options(
        self,
        *,
        voice: NotGivenOr[Voice] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[NoiseReduction | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        truncation: NotGivenOr[Truncation | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning | None] = NOT_GIVEN,
    ) -> None:
        """Change the model's options; every live session sends what changed."""
        self._update_options(
            voice=voice,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            max_response_output_tokens=max_response_output_tokens,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
        )


__all__ = ["RealtimeModel", "RealtimeSession"]
__all__ += _public.__all__
