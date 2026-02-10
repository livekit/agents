from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .base import AgentMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics


class _BaseModelUsage(BaseModel):
    def __repr__(self) -> str:
        # skip zeros for concise display
        fields = {k: v for k, v in self.model_dump().items() if v != 0 and v != 0.0}
        fields_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fields_str})"


class LLMModelUsage(_BaseModelUsage):
    """Usage summary for LLM models."""

    type: Literal["llm_usage"] = "llm_usage"
    provider: str
    """The provider name (e.g., 'openai', 'anthropic')."""
    model: str
    """The model name (e.g., 'gpt-4o', 'claude-3-5-sonnet')."""

    input_tokens: int = 0
    """Total input tokens."""
    input_cached_tokens: int = 0
    """Input tokens served from cache."""
    input_audio_tokens: int = 0
    """Input audio tokens (for multimodal models)."""
    input_cached_audio_tokens: int = 0
    """Cached input audio tokens."""
    input_text_tokens: int = 0
    """Input text tokens."""
    input_cached_text_tokens: int = 0
    """Cached input text tokens."""
    input_image_tokens: int = 0
    """Input image tokens (for multimodal models)."""
    input_cached_image_tokens: int = 0
    """Cached input image tokens."""

    output_tokens: int = 0
    """Total output tokens."""
    output_audio_tokens: int = 0
    """Output audio tokens (for multimodal models)."""
    output_text_tokens: int = 0
    """Output text tokens."""

    session_duration: float = 0.0
    """Total session connection duration in seconds (for session-based billing like xAI)."""


class TTSModelUsage(_BaseModelUsage):
    """Usage summary for TTS models."""

    type: Literal["tts_usage"] = "tts_usage"
    provider: str
    """The provider name (e.g., 'elevenlabs', 'cartesia')."""
    model: str
    """The model name (e.g., 'eleven_turbo_v2', 'sonic')."""

    input_tokens: int = 0
    """Input text tokens (for token-based TTS billing, e.g., OpenAI TTS)."""
    output_tokens: int = 0
    """Output audio tokens (for token-based TTS billing, e.g., OpenAI TTS)."""
    characters_count: int = 0
    """Number of characters synthesized (for character-based TTS billing)."""
    audio_duration: float = 0.0
    """Duration of generated audio in seconds."""


class STTModelUsage(_BaseModelUsage):
    """Usage summary for STT models."""

    type: Literal["stt_usage"] = "stt_usage"
    provider: str
    """The provider name (e.g., 'deepgram', 'assemblyai')."""
    model: str
    """The model name (e.g., 'nova-2', 'best')."""

    input_tokens: int = 0
    """Input audio tokens (for token-based STT billing)."""
    output_tokens: int = 0
    """Output text tokens (for token-based STT billing)."""
    audio_duration: float = 0.0
    """Duration of processed audio in seconds."""


ModelUsage = LLMModelUsage | TTSModelUsage | STTModelUsage
"""Union type for all model usage types."""


class ModelUsageCollector:
    """Collects and aggregates usage metrics per model/provider combination."""

    def __init__(self) -> None:
        self._llm_usage: dict[tuple[str, str], LLMModelUsage] = {}
        self._tts_usage: dict[tuple[str, str], TTSModelUsage] = {}
        self._stt_usage: dict[tuple[str, str], STTModelUsage] = {}

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

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

    def _get_llm_usage(self, provider: str, model: str) -> LLMModelUsage:
        """Get or create an LLMModelUsage for the given provider/model combination."""
        key = (provider, model)
        if key not in self._llm_usage:
            self._llm_usage[key] = LLMModelUsage(provider=provider, model=model)
        return self._llm_usage[key]

    def _get_tts_usage(self, provider: str, model: str) -> TTSModelUsage:
        """Get or create a TTSModelUsage for the given provider/model combination."""
        key = (provider, model)
        if key not in self._tts_usage:
            self._tts_usage[key] = TTSModelUsage(provider=provider, model=model)
        return self._tts_usage[key]

    def _get_stt_usage(self, provider: str, model: str) -> STTModelUsage:
        """Get or create an STTModelUsage for the given provider/model combination."""
        key = (provider, model)
        if key not in self._stt_usage:
            self._stt_usage[key] = STTModelUsage(provider=provider, model=model)
        return self._stt_usage[key]

    def collect(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, LLMMetrics):
            provider, model = self._extract_provider_model(metrics)
            usage = self._get_llm_usage(provider, model)
            usage.input_tokens += metrics.prompt_tokens
            usage.input_cached_tokens += metrics.prompt_cached_tokens
            usage.output_tokens += metrics.completion_tokens

        elif isinstance(metrics, RealtimeModelMetrics):
            provider, model = self._extract_provider_model(metrics)
            usage = self._get_llm_usage(provider, model)
            usage.input_tokens += metrics.input_tokens
            usage.input_cached_tokens += metrics.input_token_details.cached_tokens

            usage.input_text_tokens += metrics.input_token_details.text_tokens
            usage.input_cached_text_tokens += (
                metrics.input_token_details.cached_tokens_details.text_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0
            )
            usage.input_image_tokens += metrics.input_token_details.image_tokens
            usage.input_cached_image_tokens += (
                metrics.input_token_details.cached_tokens_details.image_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0
            )
            usage.input_audio_tokens += metrics.input_token_details.audio_tokens
            usage.input_cached_audio_tokens += (
                metrics.input_token_details.cached_tokens_details.audio_tokens
                if metrics.input_token_details.cached_tokens_details
                else 0
            )

            usage.output_text_tokens += metrics.output_token_details.text_tokens
            usage.output_audio_tokens += metrics.output_token_details.audio_tokens
            usage.output_tokens += metrics.output_tokens
            usage.session_duration += metrics.session_duration

        elif isinstance(metrics, TTSMetrics):
            provider, model = self._extract_provider_model(metrics)
            tts_usage = self._get_tts_usage(provider, model)
            tts_usage.input_tokens += metrics.input_tokens
            tts_usage.output_tokens += metrics.output_tokens
            tts_usage.characters_count += metrics.characters_count
            tts_usage.audio_duration += metrics.audio_duration

        elif isinstance(metrics, STTMetrics):
            provider, model = self._extract_provider_model(metrics)
            stt_usage = self._get_stt_usage(provider, model)
            stt_usage.input_tokens += metrics.input_tokens
            stt_usage.output_tokens += metrics.output_tokens
            stt_usage.audio_duration += metrics.audio_duration

    def flatten(self) -> list[ModelUsage]:
        """Returns a list of usage summaries, one per model/provider combination."""
        result: list[ModelUsage] = []
        result.extend(u.model_copy(deep=True) for u in self._llm_usage.values())
        result.extend(u.model_copy(deep=True) for u in self._tts_usage.values())
        result.extend(u.model_copy(deep=True) for u in self._stt_usage.values())
        return result
