from copy import deepcopy
from dataclasses import dataclass

from .base import AgentMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics


@dataclass
class UsageSummary:
    llm_prompt_tokens: int = 0
    llm_prompt_cached_tokens: int = 0
    llm_input_audio_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_output_audio_tokens: int = 0
    tts_characters_count: int = 0
    tts_audio_duration: float = 0.0
    stt_audio_duration: float = 0.0


class UsageCollector:
    def __init__(self) -> None:
        self._summary = UsageSummary()

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def collect(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, LLMMetrics):
            self._summary.llm_prompt_tokens += metrics.prompt_tokens
            self._summary.llm_prompt_cached_tokens += metrics.prompt_cached_tokens
            self._summary.llm_completion_tokens += metrics.completion_tokens

        elif isinstance(metrics, RealtimeModelMetrics):
            self._summary.llm_input_audio_tokens += metrics.input_token_details.audio_tokens
            self._summary.llm_output_audio_tokens += metrics.output_token_details.audio_tokens
            self._summary.llm_prompt_tokens += metrics.input_tokens
            self._summary.llm_prompt_cached_tokens += metrics.input_token_details.cached_tokens
            self._summary.llm_completion_tokens += metrics.output_tokens

        elif isinstance(metrics, TTSMetrics):
            self._summary.tts_characters_count += metrics.characters_count
            self._summary.tts_audio_duration += metrics.audio_duration

        elif isinstance(metrics, STTMetrics):
            self._summary.stt_audio_duration += metrics.audio_duration

    def get_summary(self) -> UsageSummary:
        return deepcopy(self._summary)
