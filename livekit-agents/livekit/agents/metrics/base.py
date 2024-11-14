from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class Error:
    pass


@dataclass
class LLMMetrics:
    request_id: str
    timestamp: float
    ttft: float
    duration: float
    label: str
    cancelled: bool
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    tokens_per_second: float
    error: Error | None


@dataclass
class STTMetrics:
    request_id: str
    timestamp: float
    duration: float
    label: str
    audio_duration: float
    streamed: bool
    error: Error | None


@dataclass
class TTSMetrics:
    request_id: str
    timestamp: float
    ttfb: float
    duration: float
    audio_duration: float
    cancelled: bool
    characters_count: int
    label: str
    streamed: bool
    error: Error | None


@dataclass
class VADMetrics:
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int
    label: str


@dataclass
class PipelineSTTMetrics(STTMetrics):
    pass


@dataclass
class PipelineEOUMetrics:
    sequence_id: str
    """Unique identifier shared across different metrics to combine related STT, LLM, and TTS metrics."""

    timestamp: float
    """Timestamp of when the event was recorded."""

    end_of_utterance_delay: float
    """Amount of time between the end of speech from VAD and the decision to end the user's turn."""

    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech.

    May be 0 if the transcript was already available.
    """


@dataclass
class PipelineLLMMetrics(LLMMetrics):
    sequence_id: str
    """Unique identifier shared across different metrics to combine related STT, LLM, and TTS metrics."""


@dataclass
class PipelineTTSMetrics(TTSMetrics):
    sequence_id: str
    """Unique identifier shared across different metrics to combine related STT, LLM, and TTS metrics."""


@dataclass
class PipelineVADMetrics(VADMetrics):
    pass


AgentMetrics = Union[
    STTMetrics,
    LLMMetrics,
    TTSMetrics,
    VADMetrics,
    PipelineSTTMetrics,
    PipelineEOUMetrics,
    PipelineLLMMetrics,
    PipelineTTSMetrics,
    PipelineVADMetrics,
]
