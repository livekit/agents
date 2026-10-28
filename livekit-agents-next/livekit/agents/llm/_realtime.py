"""The OpenAI Realtime API, and every provider speaking it (Azure OpenAI, xAI, LiveKit Inference).

The model is the core's (``_core.NativeRealtimeModel``): the socket, the session options, the
reconnects and the conversation mirror are Rust, and a ``VoiceSession`` drives it directly. This
class keeps the plugin's constructor and hands its arguments over as given.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

from .. import _core
from .._generated.realtime_options import NativeRealtimeConfig
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from . import realtime as llm_realtime
from ._realtime_options import realtime_options
from .tool_context import ToolChoice

if TYPE_CHECKING:
    import aiohttp

RealtimeModels = Literal[
    "gpt-realtime",
    "gpt-realtime-1.5",
    "gpt-realtime-2",
    "gpt-realtime-2025-08-28",
    "gpt-4o-realtime-preview",
]


SAMPLE_RATE = 24000


NUM_CHANNELS = 1


OPENAI_BASE_URL = "https://api.openai.com/v1"


DEFAULT_VOICE = "marin"


# the metrics label direct OpenAI sessions had before the client moved into the core
_PLUGIN_LABEL = "livekit.plugins.openai.realtime.realtime_model.RealtimeModel"


# sessions run in the core; the name stays for the plugins that subclass it
RealtimeSession = llm_realtime.RealtimeSession


class RealtimeModel(llm_realtime.RealtimeModel):
    _dialect: str | None = None

    def __init__(
        self,
        *,
        model: RealtimeModels | str = "gpt-realtime",
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Any] = NOT_GIVEN,
        truncation: NotGivenOr[Any] = NOT_GIVEN,
        reasoning: NotGivenOr[Any] = NOT_GIVEN,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        azure_deployment: str | None = None,
        entra_token: str | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        _azure_defaults: bool = False,
    ) -> None:
        """
        Initialize a Realtime model client for OpenAI or Azure OpenAI.

        Args:
            model (str): Realtime model name, e.g., "gpt-realtime".
            voice (str): Voice used for audio responses. Defaults to "marin".
            modalities (list[Literal["text", "audio"]] | NotGiven): Modalities to enable. Defaults to ["text", "audio"] if not provided.
            tool_choice (llm.ToolChoice | None | NotGiven): Tool selection policy for responses.
            base_url (str | NotGiven): HTTP base URL of the OpenAI/Azure API. If not provided, uses OPENAI_BASE_URL for OpenAI; for Azure, constructed from AZURE_OPENAI_ENDPOINT.
            input_audio_transcription (AudioTranscription | None | NotGiven): Options for transcribing input audio.
            input_audio_noise_reduction (NoiseReductionType | NoiseReduction | InputAudioNoiseReduction | None | NotGiven): Input audio noise reduction settings.
            turn_detection (RealtimeAudioInputTurnDetection | None | NotGiven): Server-side turn-detection options.
            speed (float | NotGiven): Audio playback speed multiplier.
            tracing (Tracing | None | NotGiven): Tracing configuration for OpenAI Realtime.
            truncation (RealtimeTruncation | None | NotGiven): Truncation configuration for OpenAI Realtime.
            reasoning (RealtimeReasoning | None | NotGiven): Reasoning config for reasoning-capable models (e.g. ``gpt-realtime-2``), e.g. ``RealtimeReasoning(effort="low")``.
            api_key (str | None): OpenAI API key. If None and not using Azure, read from OPENAI_API_KEY.
            http_session (aiohttp.ClientSession | None): Unused; the core owns the connection.
            azure_deployment (str | None): Azure deployment name. Presence of any Azure-specific option enables Azure mode.
            entra_token (str | None): Azure Entra token auth (alternative to api_key).
            max_session_duration (float | None | NotGiven): Seconds before recycling the connection.
            conn_options (APIConnectOptions): Retry/backoff and connection settings.

        Raises:
            ValueError: If OPENAI_API_KEY is missing in non-Azure mode, or if Azure endpoint cannot be determined when in Azure mode.
        """
        is_azure = entra_token is not None or azure_deployment is not None

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None and not is_azure:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the OPENAI_API_KEY environment variable"
            )
        if not base_url and is_azure:
            if (azure_endpoint := os.getenv("AZURE_OPENAI_ENDPOINT")) is None:
                raise ValueError(
                    "Missing Azure endpoint. Please pass base_url "
                    "or set AZURE_OPENAI_ENDPOINT environment variable."
                )
            base_url = f"{azure_endpoint.rstrip('/')}/openai"

        label = (
            _PLUGIN_LABEL
            if type(self) is RealtimeModel
            else f"{type(self).__module__}.{type(self).__name__}"
        )
        self._native = _core.NativeRealtimeModel(
            NativeRealtimeConfig(
                dialect=self._dialect,
                api_key=api_key,
                base_url=base_url or os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL),
                model=model,
                voice=voice,
                modalities=modalities or None,
                azure_deployment=azure_deployment,
                entra_token=entra_token,
                azure_defaults=_azure_defaults,
                conn_options=conn_options,
                label=label,
                options=realtime_options(
                    input_audio_transcription=input_audio_transcription,
                    input_audio_noise_reduction=input_audio_noise_reduction,
                    turn_detection=turn_detection,
                    tool_choice=tool_choice,
                    speed=speed,
                    tracing=tracing,
                    truncation=truncation,
                    reasoning=reasoning,
                    max_session_duration=max_session_duration,
                ),
            )
        )
        super().__init__(capabilities=self.capabilities)
        self._label = label

    @property
    def model(self) -> str:
        return self._native.model  # type: ignore[no-any-return]

    @property
    def provider(self) -> str:
        return self._native.provider  # type: ignore[no-any-return]

    @property
    def capabilities(self) -> llm_realtime.RealtimeCapabilities:
        return llm_realtime.RealtimeCapabilities(**self._native.capabilities())

    @classmethod
    def with_azure(
        cls,
        *,
        azure_deployment: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        entra_token: str | None = None,
        base_url: str | None = None,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any] = NOT_GIVEN,
        input_audio_noise_reduction: Any = None,
        turn_detection: NotGivenOr[Any] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Any] = NOT_GIVEN,
        reasoning: NotGivenOr[Any] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> RealtimeModel:
        """
        Create a RealtimeModel configured for Azure OpenAI.

        Args:
            azure_deployment (str): Azure OpenAI deployment name.
            azure_endpoint (str | None): Azure endpoint URL; if None, taken from AZURE_OPENAI_ENDPOINT.
            api_key (str | None): Azure API key; if None, taken from AZURE_OPENAI_API_KEY. Omit if using `entra_token`.
            entra_token (str | None): Azure Entra token for AAD auth. Provide instead of `api_key`.
            base_url (str | None): Explicit base URL. Mutually exclusive with `azure_endpoint`. If provided, used as-is.
            voice (str): Voice used for audio responses.
            modalities (list[Literal["text", "audio"]] | NotGiven): Modalities to enable. Defaults to ["text", "audio"] if not provided.
            input_audio_transcription (AudioTranscription | InputAudioTranscription | None | NotGiven): Transcription options; defaults to Azure-optimized values when not provided.
            input_audio_noise_reduction (NoiseReductionType | InputAudioNoiseReduction | None): Input noise reduction settings. Defaults to None.
            turn_detection (RealtimeAudioInputTurnDetection | TurnDetection | None | NotGiven): Server-side VAD; defaults to Azure-optimized values when not provided.
            speed (float | NotGiven): Audio playback speed multiplier.
            tracing (Tracing | None | NotGiven): Tracing configuration for OpenAI Realtime.
            reasoning (RealtimeReasoning | None | NotGiven): Reasoning config for reasoning-capable models, e.g. ``RealtimeReasoning(effort="low")``.
            http_session (aiohttp.ClientSession | None): Unused; the core owns the connection.
            max_session_duration (float | None | NotGiven): Seconds before recycling the connection.

        Raises:
            ValueError: If credentials are missing, Azure endpoint cannot be determined, or both `base_url` and `azure_endpoint` are provided.
        """
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None and entra_token is None:
            raise ValueError(
                "Missing credentials. Please pass one of `api_key`, `entra_token`, "
                "or the `AZURE_OPENAI_API_KEY` environment variable."
            )
        if base_url is None:
            azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint is None:
                raise ValueError(
                    "Missing Azure endpoint. Please pass the `azure_endpoint` "
                    "parameter or set the `AZURE_OPENAI_ENDPOINT` environment variable."
                )
            base_url = f"{azure_endpoint.rstrip('/')}/openai"
        elif azure_endpoint is not None:
            raise ValueError("base_url and azure_endpoint are mutually exclusive")

        return RealtimeModel(
            voice=voice,
            modalities=modalities,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            speed=speed,
            tracing=tracing,
            reasoning=reasoning,
            api_key=api_key,
            azure_deployment=azure_deployment,
            entra_token=entra_token,
            base_url=base_url,
            max_session_duration=max_session_duration,
            _azure_defaults=True,
        )

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Any] = NOT_GIVEN,
        truncation: NotGivenOr[Any] = NOT_GIVEN,
        reasoning: NotGivenOr[Any] = NOT_GIVEN,
    ) -> None:
        """Change the model's options; every live session sends what changed."""
        self._native.update_options(
            realtime_options(
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
        )

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        raise NotImplementedError(
            f"{type(self).__name__} sessions run in the core; give the model to a VoiceSession"
        )

    def _core_native(self) -> _core.NativeRealtimeModel:
        """The model as a ``VoiceSession`` drives it."""
        return self._native

    async def aclose(self) -> None:
        pass
