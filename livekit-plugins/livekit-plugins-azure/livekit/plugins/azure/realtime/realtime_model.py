"""Azure Voice Live options over the shared OpenAI realtime protocol and request lifecycle."""

from __future__ import annotations

import os
from typing import Any, Literal

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

API_VERSION = "2026-04-10"
DEFAULT_MODEL = "gpt-realtime"
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
RealtimeSession = openai.realtime.RealtimeSession


class RealtimeModel(openai.realtime.RealtimeModel):
    _dialect = "voice_live"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        voice: str | dict[str, Any] = DEFAULT_VOICE,
        input_audio_transcription: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        turn_detection: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        max_output_tokens: int | Literal["inf"] = DEFAULT_MAX_OUTPUT_TOKENS,
        api_key: str | None = None,
        entra_token: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Azure Voice Live: Azure speech, voices and turn detection around a realtime or text model.

        Args:
            endpoint: The Foundry or Speech resource endpoint, e.g.
                ``https://<resource>.services.ai.azure.com``. Defaults to
                ``AZURE_VOICE_LIVE_ENDPOINT``.
            model: ``gpt-realtime``, or a text model such as ``gpt-4.1``. Defaults to
                ``AZURE_VOICE_LIVE_MODEL``, then ``gpt-realtime``.
            voice: An Azure voice name (``en-US-AvaNeural``), an OpenAI voice name (``alloy``), or
                a Voice Live voice object such as ``{"type": "azure-custom", ...}``.
            input_audio_transcription: e.g. ``{"model": "azure-speech", "language": "en-US"}``;
                ``None`` disables it. Defaults to ``azure-speech`` for text models and
                ``phi4-mm-realtime``, ``whisper-1`` otherwise.
            modalities: ``["text"]`` for text-only responses. Defaults to audio.
            turn_detection: e.g. ``{"type": "azure_semantic_vad", "remove_filler_words": True}``.
                Defaults to server VAD.
            input_audio_noise_reduction: e.g. ``{"type": "azure_deep_noise_suppression"}``.
            tool_choice: Tool selection policy.
            max_output_tokens: Maximum output tokens per response.
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
            input_audio_transcription = {"model": _default_transcription_model(model)}

        super().__init__(
            base_url=f"{endpoint.rstrip('/')}/voice-live/realtime?api-version={API_VERSION}",
            model=model,
            voice=_voice(voice, input_audio_transcription),
            modalities=modalities,
            input_audio_transcription=input_audio_transcription,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            api_key=api_key,
            entra_token=entra_token,
            # Voice Live selects the model with the `model` query parameter, Azure's deployment
            azure_deployment=model,
            conn_options=conn_options,
        )
        self.update_options(max_response_output_tokens=max_output_tokens)


def _default_transcription_model(model: str) -> str:
    """``azure-speech`` for the text models and ``phi4-mm-realtime``, which reject ``whisper-1``.

    See https://learn.microsoft.com/azure/ai-services/speech-service/voice-live-how-to
    """
    name = model.lower()
    return "azure-speech" if "realtime" not in name or name.startswith("phi") else "whisper-1"


def _voice(voice: str | dict[str, Any], transcription: dict[str, Any] | None) -> Any:
    """An Azure voice name (it has a hyphen) as a voice object, pinned to a single language."""
    if not isinstance(voice, str) or "-" not in voice:
        return voice
    language = (transcription or {}).get("language")
    locale = {"locale": language} if language and "," not in language else {}
    return {"type": "azure-standard", "name": voice, **locale}
