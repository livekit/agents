"""xAI options over the shared OpenAI realtime protocol and request lifecycle."""

from __future__ import annotations

from datetime import timedelta

from livekit.agents.llm._realtime import (
    AudioTranscription,
    Reasoning,
    SemanticVad,
    ServerVad,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.plugins import openai

from ..types import GrokRealtimeModels, GrokVoices

RealtimeSession = openai.realtime.RealtimeSession


class RealtimeModel(openai.realtime.RealtimeModel):
    _dialect = "xai"
    _provider_label = "livekit.plugins.xai.realtime.realtime_model.RealtimeModel"

    def __init__(
        self,
        *,
        model: GrokRealtimeModels | str | None = None,
        voice: GrokVoices | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        turn_detection: NotGivenOr[ServerVad | SemanticVad | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        max_session_duration: NotGivenOr[timedelta | None] = NOT_GIVEN,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        """An xAI Grok voice model.

        Args:
            model: ``grok-voice-latest`` by default.
            voice: ``Ara`` by default.
            api_key: ``XAI_API_KEY`` by default.
            base_url: xAI's realtime endpoint by default.
            turn_detection: Server VAD by default; ``None`` leaves the turns to the framework.
            input_audio_transcription: ``grok-transcribe`` by default; ``None`` turns it off.
            reasoning: The model's reasoning, e.g. ``Reasoning(effort="low")``.
            speed: The playback speed of its audio.
            max_session_duration: How long one connection serves before it is recycled; never
                by default.
            conn_options: The connection's retry and timeout options.
        """
        super().__init__(
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
            turn_detection=turn_detection,
            input_audio_transcription=input_audio_transcription,
            reasoning=reasoning,
            speed=speed,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )

    # xAI's session carries no tool choice, output limit, noise reduction, tracing or truncation
    def update_options(  # type: ignore[override]
        self,
        *,
        voice: NotGivenOr[GrokVoices | str] = NOT_GIVEN,
        turn_detection: NotGivenOr[ServerVad | SemanticVad | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Change the model's options; every live session sends what changed."""
        super().update_options(
            voice=voice,
            turn_detection=turn_detection,
            input_audio_transcription=input_audio_transcription,
            reasoning=reasoning,
            speed=speed,
        )
