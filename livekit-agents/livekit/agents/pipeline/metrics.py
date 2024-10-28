from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Literal, TypedDict, Union

from ..llm import LLMMetrics
from ..stt import STTMetrics
from ..tts import TTSMetrics
from ..vad import VADMetrics


@dataclass
class SpeechData:
    sequence_id: str


SpeechDataContextVar = contextvars.ContextVar[SpeechData]("voice_assistant_speech_data")


class PipelineSTTMetrics(STTMetrics, TypedDict):
    type: Literal["stt_metrics"]


class PipelineEOUMetrics(TypedDict):
    type: Literal["eou_metrics"]
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


class PipelineLLMMetrics(LLMMetrics, TypedDict):
    type: Literal["llm_metrics"]
    sequence_id: str
    """Unique identifier shared across different metrics to combine related STT, LLM, and TTS metrics."""


class PipelineTTSMetrics(TTSMetrics, TypedDict):
    type: Literal["tts_metrics"]
    sequence_id: str
    """Unique identifier shared across different metrics to combine related STT, LLM, and TTS metrics."""


class PipelineVADMetrics(VADMetrics, TypedDict):
    type: Literal["vad_metrics"]


PipelineMetrics = Union[
    PipelineSTTMetrics,
    PipelineEOUMetrics,
    PipelineLLMMetrics,
    PipelineTTSMetrics,
    PipelineVADMetrics,
]
