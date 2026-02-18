from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Metadata(BaseModel):
    model_name: str | None = None
    model_provider: str | None = None


class LLMMetrics(BaseModel):
    type: Literal["llm_metrics"] = "llm_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    ttft: float
    cancelled: bool
    completion_tokens: int
    prompt_tokens: int
    prompt_cached_tokens: int
    total_tokens: int
    tokens_per_second: float
    speech_id: str | None = None
    metadata: Metadata | None = None


class STTMetrics(BaseModel):
    type: Literal["stt_metrics"] = "stt_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    """The request duration in seconds, 0.0 if the STT is streaming."""
    audio_duration: float
    """The duration of the pushed audio in seconds."""
    input_tokens: int = 0
    """Input audio tokens (for token-based billing)."""
    output_tokens: int = 0
    """Output text tokens (for token-based billing)."""
    streamed: bool
    """Whether the STT is streaming (e.g using websocket)."""
    metadata: Metadata | None = None


class TTSMetrics(BaseModel):
    type: Literal["tts_metrics"] = "tts_metrics"
    label: str
    request_id: str
    timestamp: float
    ttfb: float
    duration: float
    audio_duration: float
    cancelled: bool
    characters_count: int
    """Number of characters synthesized (for character-based billing)."""
    input_tokens: int = 0
    """Input text tokens (for token-based billing, e.g., OpenAI TTS)."""
    output_tokens: int = 0
    """Output audio tokens (for token-based billing, e.g., OpenAI TTS)."""
    streamed: bool
    segment_id: str | None = None
    speech_id: str | None = None
    metadata: Metadata | None = None


class VADMetrics(BaseModel):
    type: Literal["vad_metrics"] = "vad_metrics"
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int
    metadata: Metadata | None = None


class EOUMetrics(BaseModel):
    type: Literal["eou_metrics"] = "eou_metrics"
    timestamp: float
    end_of_utterance_delay: float
    """Amount of time between the end of speech from VAD and the decision to end the user's turn.
    Set to 0.0 if the end of speech was not detected.
    """

    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech.
    Set to 0.0 if the end of speech was not detected.
    """

    on_user_turn_completed_delay: float
    """Time taken to invoke the user's `Agent.on_user_turn_completed` callback."""

    speech_id: str | None = None

    metadata: Metadata | None = None


class RealtimeModelMetrics(BaseModel):
    class CachedTokenDetails(BaseModel):
        audio_tokens: int = 0
        text_tokens: int = 0
        image_tokens: int = 0

    class InputTokenDetails(BaseModel):
        audio_tokens: int = 0
        text_tokens: int = 0
        image_tokens: int = 0
        cached_tokens: int = 0
        cached_tokens_details: RealtimeModelMetrics.CachedTokenDetails | None = None

    class OutputTokenDetails(BaseModel):
        text_tokens: int = 0
        audio_tokens: int = 0
        # image_tokens is deprecated, Realtime models no longer emit this metric
        image_tokens: int = 0

    type: Literal["realtime_model_metrics"] = "realtime_model_metrics"
    label: str = ""
    request_id: str
    timestamp: float
    """The timestamp of the response creation."""
    duration: float = 0.0
    """The duration of the response from created to done in seconds."""
    session_duration: float = 0.0
    """The duration of the session connection in seconds (for session-based billing like xAI)."""
    ttft: float = -1
    """Time to first audio token in seconds. -1 if no audio token was sent."""
    cancelled: bool = False
    """Whether the request was cancelled."""
    input_tokens: int = 0
    """The number of input tokens used in the Response, including text and audio tokens."""
    output_tokens: int = 0
    """The number of output tokens sent in the Response, including text and audio tokens."""
    total_tokens: int = 0
    """The total number of tokens in the Response."""
    tokens_per_second: float = 0.0
    """The number of tokens per second."""
    input_token_details: InputTokenDetails
    """Details about the input tokens used in the Response."""
    output_token_details: OutputTokenDetails
    """Details about the output tokens used in the Response."""
    metadata: Metadata | None = None


class InterruptionMetrics(BaseModel):
    type: Literal["interruption_metrics"] = "interruption_metrics"
    timestamp: float
    total_duration: float
    """Latest RTT (Round Trip Time) time taken to perform the inference, in seconds."""
    prediction_duration: float
    """Latest time taken to perform the inference from the model side, in seconds."""
    detection_delay: float
    """Latest total time from the onset of the speech to the final prediction, in seconds."""
    num_interruptions: int
    """Number of interruptions detected, incrementally counted."""
    num_non_interruptions: int
    """Number of non-interruptions detected, incrementally counted."""
    num_requests: int
    """Number of requests sent to the interruption detection model, incrementally counted."""
    metadata: Metadata | None = None


AgentMetrics = (
    STTMetrics
    | LLMMetrics
    | TTSMetrics
    | VADMetrics
    | EOUMetrics
    | RealtimeModelMetrics
    | InterruptionMetrics
)
