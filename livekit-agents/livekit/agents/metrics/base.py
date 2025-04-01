from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class Error(BaseModel):
    error: str
    retryable: bool
    attempts_remaining: int


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
    total_tokens: int
    tokens_per_second: float
    speech_id: str | None = None
    error: Error | None = None


class STTMetrics(BaseModel):
    type: Literal["stt_metrics"] = "stt_metrics"
    label: str
    request_id: str
    timestamp: float
    duration: float
    audio_duration: float
    streamed: bool
    speech_id: str | None = None
    error: Error | None = None


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
    speech_id: str | None = None
    error: Error | None = None


class VADMetrics(BaseModel):
    type: Literal["vad_metrics"] = "vad_metrics"
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int
    speech_id: str | None = None
    error: Error | None = None


class EOUMetrics(BaseModel):
    type: Literal["eou_metrics"] = "eou_metrics"
    timestamp: float
    end_of_utterance_delay: float
    """Amount of time between the end of speech from VAD and the decision to end the user's turn."""

    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech."""

    speech_id: str | None = None
    error: Error | None = None


AgentMetrics = Union[
    STTMetrics,
    LLMMetrics,
    TTSMetrics,
    VADMetrics,
    EOUMetrics,
]

# @dataclass
# class MultimodalLLMError(Error):
#     type: str | None
#     reason: str | None = None
#     code: str | None = None
#     message: str | None = None


# @dataclass
# class MultimodalLLMMetrics(LLMMetrics):
#     @dataclass
#     class CachedTokenDetails:
#         text_tokens: int
#         audio_tokens: int

#     @dataclass
#     class InputTokenDetails:
#         cached_tokens: int
#         text_tokens: int
#         audio_tokens: int
#         cached_tokens_details: MultimodalLLMMetrics.CachedTokenDetails

#     @dataclass
#     class OutputTokenDetails:
#         text_tokens: int
#         audio_tokens: int

#     input_token_details: InputTokenDetails
#     output_token_details: OutputTokenDetails


# AgentMetrics = Union[
#     STTMetrics,
#     LLMMetrics,
#     TTSMetrics,
#     VADMetrics,
#     PipelineSTTMetrics,
#     PipelineEOUMetrics,
#     PipelineLLMMetrics,
#     PipelineTTSMetrics,
#     PipelineVADMetrics,
#     # MultimodalLLMMetrics,
# ]
