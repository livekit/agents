# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/metrics/usage_collector/records.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass

from .. import _bindings


@dataclass
class UsageSummary:
    """.. deprecated::
        Use :class:`LLMModelUsage`, :class:`TTSModelUsage`, or :class:`STTModelUsage` instead.
    """

    llm_prompt_tokens: int = 0
    llm_prompt_cached_tokens: int = 0
    llm_input_audio_tokens: int = 0
    llm_input_cached_audio_tokens: int = 0
    llm_input_text_tokens: int = 0
    llm_input_cached_text_tokens: int = 0
    llm_input_image_tokens: int = 0
    llm_input_cached_image_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_output_audio_tokens: int = 0
    llm_output_image_tokens: int = 0
    llm_output_text_tokens: int = 0
    tts_characters_count: int = 0
    tts_audio_duration: float = 0.0
    stt_audio_duration: float = 0.0

    @property
    def llm_input_tokens(self) -> int:
        """Alias for ``llm_prompt_tokens``: prompt and input are one counter under two names."""
        return _bindings.usage_summary_llm_input_tokens(self)

    @llm_input_tokens.setter
    def llm_input_tokens(self, value: int) -> None:
        _bindings.usage_summary_set_llm_input_tokens(self, value)

    @property
    def llm_output_tokens(self) -> int:
        """Alias for ``llm_completion_tokens``."""
        return _bindings.usage_summary_llm_output_tokens(self)

    @llm_output_tokens.setter
    def llm_output_tokens(self, value: int) -> None:
        _bindings.usage_summary_set_llm_output_tokens(self, value)

    def __post_init__(self) -> None:
        _bindings.usage_summary_post_init(self)


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
UsageSummary.__module__ = "livekit.agents.metrics.usage_collector"
