from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class LLMMetrics(BaseModel):
    type: Literal["llm_metrics"] = "llm_metrics"
    timestamp: float
    label: str
    request_id: str = ""
    duration: float = 0.0
    ttft: float = 0.0
    cancelled: bool = False
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    speech_id: str | None = None


class STTMetrics(BaseModel):
    type: Literal["stt_metrics"] = "stt_metrics"
    timestamp: float
    label: str
    request_id: str = ""
    duration: float = 0.0
    audio_duration: float = 0.0
    streamed: bool = False
    speech_id: str | None = None


class TTSMetrics(BaseModel):
    type: Literal["tts_metrics"] = "tts_metrics"
    timestamp: float
    label: str
    request_id: str = ""
    ttfb: float = 0.0
    duration: float = 0.0
    audio_duration: float = 0.0
    cancelled: bool = False
    characters_count: int = 0
    streamed: bool = False
    speech_id: str | None = None


class VADMetrics(BaseModel):
    type: Literal["vad_metrics"] = "vad_metrics"
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int
    speech_id: str | None = None


class EOUMetrics(BaseModel):
    type: Literal["eou_metrics"] = "eou_metrics"
    timestamp: float
    end_of_utterance_delay: float
    """Amount of time between the end of speech from VAD and the decision to end the user's turn."""

    transcription_delay: float
    """Time taken to obtain the transcript after the end of the user's speech."""

    speech_id: str | None = None


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
