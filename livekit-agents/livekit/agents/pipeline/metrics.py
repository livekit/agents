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
    timestamp: float
    duration: float
    transcription_delay: float
    """time it took to get the transcript after the end of the user's speech
    could be 0 if the transcript was already available"""


class PipelineLLMMetrics(LLMMetrics, TypedDict):
    type: Literal["llm_metrics"]
    sequence_id: str


class PipelineTTSMetrics(TTSMetrics, TypedDict):
    type: Literal["tts_metrics"]
    sequence_id: str


class PipelineVADMetrics(VADMetrics, TypedDict):
    type: Literal["vad_metrics"]


PipelineMetrics = Union[
    PipelineSTTMetrics,
    PipelineEOUMetrics,
    PipelineLLMMetrics,
    PipelineTTSMetrics,
    PipelineVADMetrics,
]
