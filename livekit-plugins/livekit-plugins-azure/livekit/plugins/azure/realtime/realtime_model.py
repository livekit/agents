from __future__ import annotations

import os
from typing import Literal

from livekit.agents import llm
from livekit.agents.llm._realtime import (
    AudioTranscription,
    AzureDeepNoiseSuppression,
    AzureSemanticVad,
    AzureVoice,
    NoiseReductionType,
    RealtimeSession as RealtimeSession,
    ServerVad,
    Tracing,
    _OpenAIProtocolModel,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given

API_VERSION = "2026-04-10"
DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class RealtimeModel(_OpenAIProtocolModel):
    _dialect = "voice_live"
    _provider_label = "livekit.plugins.azure.realtime.realtime_model.RealtimeModel"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        voice: str | AzureVoice = DEFAULT_VOICE,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        modalities: list[Literal["text", "audio"]] | None = None,
        turn_detection: NotGivenOr[ServerVad | AzureSemanticVad | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            AzureDeepNoiseSuppression | NoiseReductionType | None
        ] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
        api_key: str | None = None,
        entra_token: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        """Azure Voice Live: Azure speech, voices and turn detection around a realtime or text model.

        Args:
            endpoint: The Foundry or Speech resource endpoint, e.g.
                ``https://<resource>.services.ai.azure.com``. Defaults to
                ``AZURE_VOICE_LIVE_ENDPOINT``.
            model: ``gpt-realtime``, or a text model such as ``gpt-4.1``. Defaults to
                ``AZURE_VOICE_LIVE_MODEL``, then ``gpt-realtime``.
            voice: An Azure voice name (``en-US-AvaNeural``), an OpenAI voice name (``alloy``), or
                an ``AzureVoice``.
            input_audio_transcription: e.g.
                ``AudioTranscription(model="azure-speech", language="en-US")``; ``None`` disables
                it. Defaults to ``azure-speech`` for text models and ``phi4-mm-realtime``,
                ``whisper-1`` otherwise.
            modalities: ``["text"]`` for text-only responses. Defaults to audio.
            turn_detection: e.g. ``AzureSemanticVad(remove_filler_words=True)``. Defaults to
                server VAD.
            input_audio_noise_reduction: e.g. ``AzureDeepNoiseSuppression()``.
            tool_choice: Tool selection policy.
            speed: The playback speed of its audio.
            tracing: Tracing configuration for the session.
            max_output_tokens: Maximum output tokens per response; ``None`` is unlimited.
            api_key: Defaults to ``AZURE_VOICE_LIVE_API_KEY``. Omit when using ``entra_token``.
            entra_token: A Microsoft Entra token (``https://ai.azure.com/.default`` scope).
            conn_options: Connection retry and timeout options.
        """
        endpoint = endpoint or os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure Voice Live endpoint must be provided via 'endpoint' parameter "
                "or AZURE_VOICE_LIVE_ENDPOINT environment variable"
            )
        api_key = api_key or os.environ.get("AZURE_VOICE_LIVE_API_KEY")
        if api_key is None and entra_token is None:
            raise ValueError(
                "Missing credentials. Please pass one of `api_key`, `entra_token`, "
                "or the `AZURE_VOICE_LIVE_API_KEY` environment variable."
            )
        model = model or os.environ.get("AZURE_VOICE_LIVE_MODEL") or DEFAULT_MODEL
        if not is_given(input_audio_transcription):
            input_audio_transcription = AudioTranscription(
                model=_default_transcription_model(model)
            )

        super().__init__(
            base_url=f"{endpoint.rstrip('/')}/voice-live/realtime?api-version={API_VERSION}",
            model=model,
            voice=_voice(voice, input_audio_transcription),
            modalities=modalities,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            speed=speed,
            tracing=tracing,
            max_response_output_tokens=max_output_tokens,
            api_key=api_key,
            entra_token=entra_token,
            # Voice Live selects the model with the `model` query parameter, Azure's deployment
            azure_deployment=model,
            conn_options=conn_options,
        )

    # Voice Live's session carries no truncation or reasoning
    def update_options(
        self,
        *,
        voice: NotGivenOr[str | AzureVoice] = NOT_GIVEN,
        turn_detection: NotGivenOr[ServerVad | AzureSemanticVad | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[
            AzureDeepNoiseSuppression | NoiseReductionType | None
        ] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        """Change the model's options; every live session sends what changed."""
        self._update_options(
            voice=voice,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            max_response_output_tokens=max_output_tokens,
            speed=speed,
            tracing=tracing,
        )


def _default_transcription_model(model: str) -> str:
    """``azure-speech`` for the text models and ``phi4-mm-realtime``, which reject ``whisper-1``.

    See https://learn.microsoft.com/azure/ai-services/speech-service/voice-live-how-to
    """
    name = model.lower()
    return "azure-speech" if "realtime" not in name or name.startswith("phi") else "whisper-1"


def _voice(voice: str | AzureVoice, transcription: AudioTranscription | None) -> str | AzureVoice:
    """An Azure voice name (it has a hyphen) as a voice object, pinned to a single language."""
    if not isinstance(voice, str) or "-" not in voice:
        return voice
    language = transcription.language if transcription else None
    locale = language if language and "," not in language else None
    return AzureVoice(name=voice, locale=locale)
