from copy import deepcopy
from dataclasses import dataclass

from .base import AgentMetrics, LLMMetrics, MultimodalLLMMetrics, STTMetrics, TTSMetrics


@dataclass
class UsageSummary:
    llm_prompt_tokens: int
    llm_completion_tokens: int
    tts_characters_count: int
    stt_audio_duration: float

    # Multimodal properties
    cached_text_input_tokens: int
    uncached_text_input_tokens: int
    cached_audio_input_tokens: int
    uncached_audio_input_tokens: int
    output_text_tokens: int
    output_audio_tokens: int


class UsageCollector:
    def __init__(self) -> None:
        self._summary = UsageSummary(0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0)

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def collect(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, MultimodalLLMMetrics):
            self._summary.cached_text_input_tokens += (
                metrics.input_token_details.cached_tokens_details.text_tokens
            )
            self._summary.cached_audio_input_tokens += (
                metrics.input_token_details.cached_tokens_details.audio_tokens
            )
            self._summary.uncached_text_input_tokens += (
                metrics.input_token_details.text_tokens
                - metrics.input_token_details.cached_tokens_details.text_tokens
            )
            self._summary.uncached_audio_input_tokens += (
                metrics.input_token_details.audio_tokens
                - metrics.input_token_details.cached_tokens_details.audio_tokens
            )
            self._summary.output_text_tokens += metrics.output_token_details.text_tokens
            self._summary.output_audio_tokens += metrics.output_token_details.audio_tokens
        elif isinstance(metrics, LLMMetrics):
            self._summary.llm_prompt_tokens += metrics.prompt_tokens
            self._summary.llm_completion_tokens += metrics.completion_tokens

        elif isinstance(metrics, TTSMetrics):
            self._summary.tts_characters_count += metrics.characters_count

        elif isinstance(metrics, STTMetrics):
            self._summary.stt_audio_duration += metrics.audio_duration

    def get_summary(self) -> UsageSummary:
        return deepcopy(self._summary)
