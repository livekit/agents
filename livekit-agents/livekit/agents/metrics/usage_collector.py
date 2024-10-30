from copy import deepcopy
from dataclasses import dataclass

from .base import AgentMetrics, LLMMetrics, STTMetrics, TTSMetrics


@dataclass
class UsageSummary:
    llm_prompt_tokens: int
    llm_completion_tokens: int
    tts_characters_count: int
    stt_audio_duration: float


class UsageCollector:
    def __init__(self) -> None:
        self._summary = UsageSummary(0, 0, 0, 0.0)

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def collect(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, LLMMetrics):
            self._summary.llm_prompt_tokens += metrics.prompt_tokens
            self._summary.llm_completion_tokens += metrics.completion_tokens

        elif isinstance(metrics, TTSMetrics):
            self._summary.tts_characters_count += metrics.characters_count

        elif isinstance(metrics, STTMetrics):
            self._summary.stt_audio_duration += metrics.audio_duration

    def get_summary(self) -> UsageSummary:
        return deepcopy(self._summary)
