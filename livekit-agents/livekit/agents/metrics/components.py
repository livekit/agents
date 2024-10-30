from typing import TypedDict


class LLMMetrics(TypedDict):
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


class STTMetrics(TypedDict):
    request_id: str
    timestamp: float
    duration: float
    label: str
    audio_duration: float
    streamed: bool


class TTSMetrics(TypedDict):
    timestamp: float
    request_id: str
    ttfb: float
    duration: float
    audio_duration: float
    cancelled: bool
    characters_count: int
    label: str
    streamed: bool


class VADMetrics(TypedDict):
    timestamp: float
    inference_duration_total: float
    inference_count: int
    label: str
