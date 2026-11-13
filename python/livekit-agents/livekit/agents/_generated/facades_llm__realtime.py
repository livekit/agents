# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..types import NOT_GIVEN
from .realtime import RealtimeModel as BaseRealtimeModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import timedelta

    from livekit.agents.llm.tool_context import ToolChoice

    from ..types import NotGivenOr
    from .connect import APIConnectOptions
    from .realtime_options import (
        AudioTranscription,
        InferenceGateway,
        NoiseReduction,
        RealtimeModels,
        RealtimeProvider,
        Reasoning,
        Tracing,
        Truncation,
        TurnDetection,
        Voice,
    )
    from .realtime_wire import Modalities


class RealtimeModel(_core.OpenAIRealtimeModel, BaseRealtimeModel):
    """`RealtimeModel` over the OpenAI Realtime protocol: OpenAI, Azure OpenAI and every provider
    speaking it. A provider subclass sets `_dialect`, `_provider_label` (the name logs and metrics
    give it) and, for LiveKit Inference, `_gateway`.
    """

    _dialect: RealtimeProvider | None = None
    _provider_label: str | None = None
    _gateway: InferenceGateway | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(cls, ("session",), "session.llm_node")

    def __init__(
        self,
        *,
        model: RealtimeModels | str | None = None,
        voice: Voice | None = None,
        modalities: Sequence[Modalities] | None = None,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
        base_url: str | None = None,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[NoiseReduction | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        truncation: NotGivenOr[Truncation | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | None] = NOT_GIVEN,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        azure_endpoint: str | None = None,
        entra_token: str | None = None,
        max_session_duration: NotGivenOr[timedelta | None] = NOT_GIVEN,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        """A realtime model over the OpenAI Realtime protocol.

        Args:
            model: The realtime model; the provider's default when not given.
            voice: The voice of its audio, a name or a voice object; the provider's default
                when not given.
            modalities: ``["text"]`` for text-only replies; audio by default.
            tool_choice: The tool selection policy.
            base_url: The API's base URL; ``OPENAI_BASE_URL`` for OpenAI, from ``azure_endpoint``
                for Azure.
            input_audio_transcription: The caller's transcription; ``None`` turns it off.
            input_audio_noise_reduction: Noise reduction of the caller's audio.
            turn_detection: Server turn detection; ``None`` leaves the turns to the framework.
            speed: The playback speed of its audio.
            tracing: OpenAI's tracing of the session.
            truncation: How the conversation is truncated over the input token limit.
            reasoning: A reasoning model's reasoning, e.g. ``Reasoning(effort="low")``.
            max_response_output_tokens: The tokens of one reply; ``None`` is unlimited.
            api_key: ``OPENAI_API_KEY`` by default, ``AZURE_OPENAI_API_KEY`` on Azure.
            azure_deployment: The Azure OpenAI deployment; giving it selects Azure.
            azure_endpoint: The Azure endpoint; ``AZURE_OPENAI_ENDPOINT`` by default. Exclusive
                with ``base_url``.
            entra_token: A Microsoft Entra token, in place of ``api_key``; selects Azure.
            max_session_duration: How long one connection serves before it is recycled; ``None``
                never recycles.
            conn_options: The connection's retry and timeout options.

        Raises:
            ValueError: If the credentials or the Azure endpoint are missing.
        """
        _core.OpenAIRealtimeModel.__init__(
            self,
            model=model,
            voice=voice,
            modalities=modalities,
            tool_choice=tool_choice,
            base_url=base_url,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            speed=speed,
            tracing=tracing,
            truncation=truncation,
            reasoning=reasoning,
            max_response_output_tokens=max_response_output_tokens,
            api_key=api_key,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            entra_token=entra_token,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
            dialect=self._dialect,
            label=self._provider_label,
            gateway=self._gateway,
        )
        BaseRealtimeModel.__init__(self, capabilities=self.capabilities)

    def _update_options(
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
        """Change the model's options; every live session sends what changed. Each provider's model
        declares `update_options` with the options its session carries.
        """
        _core.OpenAIRealtimeModel._core_update_options(
            self,
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
