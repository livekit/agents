# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/models.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal, TypeAlias

ReasoningEffort: TypeAlias = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


OpenAIModels: TypeAlias = Literal["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5.1", "openai/gpt-5.1-chat-latest", "openai/gpt-5.2", "openai/gpt-5.2-chat-latest", "openai/gpt-5.3-chat-latest", "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano", "openai/gpt-5.5", "openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/chat-latest", "openai/gpt-oss-120b"]


GoogleModels: TypeAlias = Literal["google/gemini-3.1-pro", "google/gemini-3-flash", "google/gemini-3.1-flash-lite", "google/gemini-3.5-flash", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]


KimiModels: TypeAlias = Literal["moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6"]


DeepSeekModels: TypeAlias = Literal["deepseek-ai/deepseek-v3", "deepseek-ai/deepseek-v3.2"]


ZAIModels: TypeAlias = Literal["zai/glm-5.1"]


XAIModels: TypeAlias = Literal["xai/grok-4-1-fast-non-reasoning", "xai/grok-4-1-fast-reasoning", "xai/grok-4.20-0309-non-reasoning", "xai/grok-4.20-0309-reasoning", "xai/grok-4.20-multi-agent-0309", "xai/grok-4.3", "xai/grok-4.5"]


OpenAIRealtimeModels: TypeAlias = Literal["openai/gpt-realtime", "openai/gpt-realtime-mini", "openai/gpt-realtime-1.5", "openai/gpt-realtime-2", "openai/gpt-realtime-2.1", "openai/gpt-realtime-2.1-mini"]


XAIRealtimeModels: TypeAlias = Literal["xai/grok-voice", "xai/grok-voice-latest", "xai/grok-voice-think-fast-2.0"]


LLMModels: TypeAlias = (
    OpenAIModels | GoogleModels | KimiModels | DeepSeekModels | ZAIModels | XAIModels
)


RealtimeModels: TypeAlias = OpenAIRealtimeModels | XAIRealtimeModels
