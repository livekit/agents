import contextvars
from dataclasses import dataclass

from ..tts import TTSMetrics


@dataclass
class SpeechData:
    sequence_id: str


SpeechDataContextVar = contextvars.ContextVar[SpeechData]("voice_assistant_speech_data")


class PipelineSTTMetrics(TTSMetrics):
    sequence_id: str


class PipelineLLMMetrics(TTSMetrics):
    sequence_id: str


class PipelineTTSMetrics(TTSMetrics):
    sequence_id: str
