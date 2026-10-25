# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/metrics/usage/records.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .. import _bindings


class _BaseModelUsage(BaseModel):
    def __repr__(self) -> str:
        return _bindings.model_usage_repr(self)


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
    input_cache_creation_tokens: int = 0
    """Input tokens used to write to the prompt cache (e.g. Anthropic cache writes)."""
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


class InterruptionModelUsage(_BaseModelUsage):
    """Usage summary for interruption detection models."""

    type: Literal["interruption_usage"] = "interruption_usage"
    provider: str
    """The provider name (e.g., 'livekit')."""
    model: str
    """The model name (e.g., 'adaptive')."""
    total_requests: int = 0
    """Total number of requests sent to the interruption detection model."""


class EOTModelUsage(_BaseModelUsage):
    """Usage summary for end-of-turn detection models."""

    type: Literal["eot_usage"] = "eot_usage"
    provider: str
    """The provider name (e.g., 'livekit')."""
    model: str
    """The model name (e.g., 'turn-detector-v1')."""
    total_requests: int = 0
    """Total number of inference requests sent to the EOT model."""


ModelUsage = LLMModelUsage | TTSModelUsage | STTModelUsage | InterruptionModelUsage | EOTModelUsage


@dataclass
class AgentSessionUsage:
    model_usage: list[ModelUsage]


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
_BaseModelUsage.__module__ = "livekit.agents.metrics.usage"
LLMModelUsage.__module__ = "livekit.agents.metrics.usage"
TTSModelUsage.__module__ = "livekit.agents.metrics.usage"
STTModelUsage.__module__ = "livekit.agents.metrics.usage"
InterruptionModelUsage.__module__ = "livekit.agents.metrics.usage"
EOTModelUsage.__module__ = "livekit.agents.metrics.usage"
AgentSessionUsage.__module__ = "livekit.agents.metrics.usage"
