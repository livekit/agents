from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


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


class STTMetrics(BaseModel):
    type: Literal["stt_metrics"] = "stt_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    """The request duration in seconds, 0.0 if the STT is streaming."""
    audio_duration: float
    """The duration of the pushed audio in seconds."""
    streamed: bool
    """Whether the STT is streaming (e.g using websocket)."""


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
    streamed: bool
    segment_id: str | None = None
    speech_id: str | None = None


class VADMetrics(BaseModel):
    type: Literal["vad_metrics"] = "vad_metrics"
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int


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


class RealtimeModelMetrics(BaseModel):
    class CachedTokenDetails(BaseModel):
        audio_tokens: int
        text_tokens: int
        image_tokens: int

    class InputTokenDetails(BaseModel):
        audio_tokens: int
        text_tokens: int
        image_tokens: int
        cached_tokens: int
        cached_tokens_details: RealtimeModelMetrics.CachedTokenDetails | None

    class OutputTokenDetails(BaseModel):
        text_tokens: int
        audio_tokens: int
        image_tokens: int

    type: Literal["realtime_model_metrics"] = "realtime_model_metrics"
    label: str
    request_id: str
    timestamp: float
    """The timestamp of the response creation."""
    duration: float
    """The duration of the response from created to done in seconds."""
    ttft: float
    """Time to first audio token in seconds. -1 if no audio token was sent."""
    cancelled: bool
    """Whether the request was cancelled."""
    input_tokens: int
    """The number of input tokens used in the Response, including text and audio tokens."""
    output_tokens: int
    """The number of output tokens sent in the Response, including text and audio tokens."""
    total_tokens: int
    """The total number of tokens in the Response."""
    tokens_per_second: float
    """The number of tokens per second."""
    input_token_details: InputTokenDetails
    """Details about the input tokens used in the Response."""
    output_token_details: OutputTokenDetails
    """Details about the output tokens used in the Response."""


AgentMetrics = Union[
    STTMetrics,
    LLMMetrics,
    TTSMetrics,
    VADMetrics,
    EOUMetrics,
    RealtimeModelMetrics,
]
