"""xAI options over the shared OpenAI realtime protocol and request lifecycle."""

from __future__ import annotations

import os

from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import AudioTranscription, RealtimeReasoning
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

from ..types import GrokRealtimeModels, GrokVoices

XAI_BASE_URL = "wss://api.x.ai/v1/realtime"
XAI_DEFAULT_MODEL: GrokRealtimeModels = "grok-voice-latest"
XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION = AudioTranscription(model="grok-transcribe")
XAI_DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
    interrupt_response=True,
)
RealtimeSession = openai.realtime.RealtimeSession


class RealtimeModel(openai.realtime.RealtimeModel):
    _dialect = "xai"

    def __init__(
        self,
        *,
        model: NotGivenOr[GrokRealtimeModels | str] = NOT_GIVEN,
        voice: NotGivenOr[GrokVoices | str | None] = "Ara",
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        reasoning: NotGivenOr[RealtimeReasoning | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the XAI_API_KEY environment variable"
            )

        # resolve NotGivenOr values before super().__init__ so mypy does not explode
        # on the OpenAI overload union combinations
        resolved_base_url = base_url if is_given(base_url) else XAI_BASE_URL
        resolved_model = model if is_given(model) else XAI_DEFAULT_MODEL
        resolved_voice = voice if is_given(voice) else "Ara"
        resolved_transcription = (
            input_audio_transcription
            if is_given(input_audio_transcription)
            else XAI_DEFAULT_INPUT_AUDIO_TRANSCRIPTION
        )
        resolved_turn_detection = (
            turn_detection if is_given(turn_detection) else XAI_DEFAULT_TURN_DETECTION
        )
        resolved_max_session_duration = (
            max_session_duration if is_given(max_session_duration) else None
        )
        super().__init__(
            base_url=resolved_base_url,
            model=resolved_model,
            voice=resolved_voice or "Ara",
            api_key=api_key,
            modalities=["audio", "text"],
            input_audio_transcription=resolved_transcription,
            turn_detection=resolved_turn_detection,
            max_session_duration=resolved_max_session_duration,
            conn_options=conn_options,
            reasoning=reasoning,
            speed=speed,
        )
