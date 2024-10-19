from typing import TypedDict, Literal
from livekit.rtc import EventEmitter


MetricsEventTypes = Literal[
    "vad_metrics_collected",
    "stt_metrics_collected",
    "llm_metrics_collected",
    "tts_metrics_collected",
]


class PipelineMetrics(EventEmitter[MetricsEventTypes]): ...


# VAD Metrics are currently being sent every 1s
class VADMetrics(TypedDict):
    type: Literal["vad_metrics"]
    timestamp: float
    avg_inference_duration: float
    inference_count: int


class STTMetrics(TypedDict):
    type: Literal["stt_metrics"]
    timestamp: float
    speech_id: str
    estimated_ttfb: float
    """
    The estimated time-to-first-byte (TTFB) of the STT service.
    This is calculated using the VAD provided inside the PipelineAgent.
    """


class LLMMetrics(TypedDict):
    type: Literal["llm_metrics"]
    timestamp: float
    speech_id: str
    ttft: float
    duration: float
    cancelled: bool


class TTSMetrics(TypedDict):
    type: Literal["tts_metrics"]
    timestamp: float
    speech_id: str
    ttfb: float
    duration: float
    audio_duration: float
    cancelled: bool
    streamed: bool
