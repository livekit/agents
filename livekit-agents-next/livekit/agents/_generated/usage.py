# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/usage.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .. import _bindings


class TokenUsage(BaseModel):
    """Token counts in one vocabulary. Totals are inclusive: the cache, reasoning and modality
    breakdowns are subsets of them and are never added to them. A breakdown is ``None`` when the
    provider did not report it; a reported zero stays zero.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    """All input tokens, including cache reads and writes and every modality."""
    output_tokens: int = 0
    """All output tokens, including hidden reasoning and every modality."""
    cache_read_tokens: int | None = None
    """Input tokens read from cache."""
    cache_write_tokens: int | None = None
    """Input tokens written to cache."""
    output_reasoning_tokens: int | None = None
    """Reasoning tokens included in output."""
    input_text_tokens: int | None = None
    input_audio_tokens: int | None = None
    input_image_tokens: int | None = None
    output_text_tokens: int | None = None
    output_audio_tokens: int | None = None
    output_image_tokens: int | None = None
    cache_read_text_tokens: int | None = None
    cache_read_audio_tokens: int | None = None
    cache_read_image_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """``input_tokens + output_tokens``."""
        return _bindings.token_usage_total(self)


class LLMUsage(BaseModel):
    """An LLM's usage, realtime models included."""

    model_config = ConfigDict(frozen=True)

    type: Literal["llm"] = "llm"
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    unmeasured_requests: int = 0
    """Requests whose expected usage measurement is unavailable."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    connection_duration: float | None = None
    """Metered session-connection seconds."""


class STTUsage(BaseModel):
    """A speech-to-text model's usage."""

    model_config = ConfigDict(frozen=True)

    type: Literal["stt"] = "stt"
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    unmeasured_requests: int = 0
    """Requests whose expected usage measurement is unavailable."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    audio_duration: float | None = None
    """Processed input audio seconds."""


class TTSUsage(BaseModel):
    """A text-to-speech model's usage."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tts"] = "tts"
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    unmeasured_requests: int = 0
    """Requests whose expected usage measurement is unavailable."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    audio_duration: float | None = None
    """Synthesized output audio seconds."""
    characters_count: int | None = None
    """Synthesized characters."""


class EOTUsage(BaseModel):
    """An end-of-turn model's usage, priced per request."""

    model_config = ConfigDict(frozen=True)

    type: Literal["eot"] = "eot"
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    unmeasured_requests: int = 0
    """Requests whose expected usage measurement is unavailable."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())


class InterruptionUsage(BaseModel):
    """An interruption detector's usage, priced per request."""

    model_config = ConfigDict(frozen=True)

    type: Literal["interruption"] = "interruption"
    provider: str | None = None
    model: str | None = None
    requests: int = 0
    unmeasured_requests: int = 0
    """Requests whose expected usage measurement is unavailable."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())


ModelUsage: TypeAlias = Annotated[
    LLMUsage | STTUsage | TTSUsage | EOTUsage | InterruptionUsage,
    Field(discriminator="type"),
]
"""One model's usage, keyed by ``(type, provider, model)``."""


class Usage(BaseModel):
    """What an operation consumed: a text run, a text-session turn or a voice session. An immutable
    snapshot; the operation's accessor returns the latest one.
    """

    model_config = ConfigDict(frozen=True)

    models: tuple[ModelUsage, ...] = Field(default_factory=lambda: tuple())
    """Per-model usage, in first-observed order."""
    tokens: TokenUsage = Field(default_factory=lambda: TokenUsage())
    """The aggregate of ``models``' tokens."""
    requests: int = 0
    """The sum of the per-model request counts."""
    tool_calls: int = 0
    """Tool-body executions started in this scope."""
    unmeasured_requests: int = 0
    """Requests whose expected usage measurement is unavailable."""
    settled: bool = False
    """The operation finished local execution and accounting cleanup."""
    complete: bool = False
    """The operation is settled and no expected usage measurement is missing."""

    @property
    def input_tokens(self) -> int:
        """``tokens.input_tokens``."""
        return _bindings.usage_input_tokens(self)

    @property
    def output_tokens(self) -> int:
        """``tokens.output_tokens``."""
        return _bindings.usage_output_tokens(self)

    @property
    def total_tokens(self) -> int:
        """``tokens.total_tokens``."""
        return _bindings.usage_total_tokens(self)


TokenUsage.__module__ = "livekit.agents"
LLMUsage.__module__ = "livekit.agents"
STTUsage.__module__ = "livekit.agents"
TTSUsage.__module__ = "livekit.agents"
EOTUsage.__module__ = "livekit.agents"
InterruptionUsage.__module__ = "livekit.agents"
Usage.__module__ = "livekit.agents"
