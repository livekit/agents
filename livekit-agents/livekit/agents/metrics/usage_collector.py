import dataclasses
from copy import deepcopy
from dataclasses import dataclass

from .base import AgentMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics


@dataclass
class UsageSummary:
    """Usage summary for a specific model/provider combination."""

    provider: str = ""
    """The provider name (e.g., 'openai', 'deepgram', 'elevenlabs')."""
    model: str = ""
    """The model name (e.g., 'gpt-4o', 'nova-2', 'eleven_turbo_v2')."""

    llm_input_tokens: int = 0
    llm_input_cached_tokens: int = 0
    llm_input_audio_tokens: int = 0
    llm_input_cached_audio_tokens: int = 0
    llm_input_text_tokens: int = 0
    llm_input_cached_text_tokens: int = 0
    llm_input_image_tokens: int = 0
    llm_input_cached_image_tokens: int = 0
    llm_output_tokens: int = 0
    llm_output_audio_tokens: int = 0
    llm_output_image_tokens: int = 0
    llm_output_text_tokens: int = 0
    tts_characters_count: int = 0
    tts_audio_duration: float = 0.0
    stt_audio_duration: float = 0.0

    # backwards-compatible property aliases
    @property
    def llm_prompt_tokens(self) -> int:
        return self.llm_input_tokens

    @llm_prompt_tokens.setter
    def llm_prompt_tokens(self, value: int) -> None:
        self.llm_input_tokens = value

    @property
    def llm_prompt_cached_tokens(self) -> int:
        return self.llm_input_cached_tokens

    @llm_prompt_cached_tokens.setter
    def llm_prompt_cached_tokens(self, value: int) -> None:
        self.llm_input_cached_tokens = value

    @property
    def llm_completion_tokens(self) -> int:
        return self.llm_output_tokens

    @llm_completion_tokens.setter
    def llm_completion_tokens(self, value: int) -> None:
        self.llm_output_tokens = value

    def to_dict(self) -> dict:
        """Returns a dict with only non-zero/non-empty values."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v}

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"UsageSummary({items})"


class UsageCollector:
    """Collects and aggregates usage metrics per model/provider combination."""

    def __init__(self) -> None:
        self._summaries: dict[tuple[str, str], UsageSummary] = {}

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def _get_summary(self, provider: str, model: str) -> UsageSummary:
        """Get or create a UsageSummary for the given provider/model combination."""
        key = (provider, model)
        if key not in self._summaries:
            self._summaries[key] = UsageSummary(provider=provider, model=model)
        return self._summaries[key]

    def _extract_provider_model(
        self, metrics: LLMMetrics | STTMetrics | TTSMetrics | RealtimeModelMetrics
    ) -> tuple[str, str]:
        """Extract provider and model from metrics metadata."""
        provider = ""
        model = ""
        if metrics.metadata:
            provider = metrics.metadata.model_provider or ""
            model = metrics.metadata.model_name or ""
        return provider, model

    def collect(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, LLMMetrics):
            provider, model = self._extract_provider_model(metrics)
            summary = self._get_summary(provider, model)
            summary.llm_input_tokens += metrics.prompt_tokens
            summary.llm_input_cached_tokens += metrics.prompt_cached_tokens
            summary.llm_output_tokens += metrics.completion_tokens

        elif isinstance(metrics, RealtimeModelMetrics):
            provider, model = self._extract_provider_model(metrics)
            summary = self._get_summary(provider, model)
            summary.llm_input_tokens += metrics.input_tokens
            summary.llm_input_cached_tokens += metrics.input_token_details.cached_tokens

            summary.llm_input_text_tokens += metrics.input_token_details.text_tokens
            summary.llm_input_cached_text_tokens += (
                metrics.input_token_details.cached_tokens_details.text_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0
            )
            summary.llm_input_image_tokens += metrics.input_token_details.image_tokens
            summary.llm_input_cached_image_tokens += (
                metrics.input_token_details.cached_tokens_details.image_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0
            )
            summary.llm_input_audio_tokens += metrics.input_token_details.audio_tokens
            summary.llm_input_cached_audio_tokens += (
                metrics.input_token_details.cached_tokens_details.audio_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0
            )

            summary.llm_output_text_tokens += metrics.output_token_details.text_tokens
            summary.llm_output_image_tokens += metrics.output_token_details.image_tokens
            summary.llm_output_audio_tokens += metrics.output_token_details.audio_tokens
            summary.llm_output_tokens += metrics.output_tokens

        elif isinstance(metrics, TTSMetrics):
            provider, model = self._extract_provider_model(metrics)
            summary = self._get_summary(provider, model)
            summary.tts_characters_count += metrics.characters_count
            summary.tts_audio_duration += metrics.audio_duration

        elif isinstance(metrics, STTMetrics):
            provider, model = self._extract_provider_model(metrics)
            summary = self._get_summary(provider, model)
            summary.stt_audio_duration += metrics.audio_duration

    def get_summary(self) -> list[UsageSummary]:
        """Returns a list of usage summaries, one per model/provider combination."""
        return [deepcopy(s) for s in self._summaries.values()]
