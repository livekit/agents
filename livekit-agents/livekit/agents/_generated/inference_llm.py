# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/inference/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..llm import ToolChoice
    from openai.types.chat import ChatCompletionPredictionContentParam, completion_create_params
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import Metadata

OpenAIModels = Literal[
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.1",
    "openai/gpt-5.1-chat-latest",
    "openai/gpt-5.2",
    "openai/gpt-5.2-chat-latest",
    "openai/gpt-5.3-chat-latest",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.5",
    "openai/chat-latest",
    "openai/gpt-oss-120b",
]


GoogleModels = Literal[
    "google/gemini-3.1-pro",
    "google/gemini-3-flash",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
]


KimiModels = Literal["moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6"]


DeepSeekModels = Literal["deepseek-ai/deepseek-v3", "deepseek-ai/deepseek-v3.2"]


ZAIModels = Literal["zai/glm-5.1"]


XAIModels = Literal[
    "xai/grok-4-1-fast-non-reasoning",
    "xai/grok-4-1-fast-reasoning",
    "xai/grok-4.20-0309-non-reasoning",
    "xai/grok-4.20-0309-reasoning",
    "xai/grok-4.20-multi-agent-0309",
    "xai/grok-4.3",
    "xai/grok-4.5",
]


LLMModels = OpenAIModels | GoogleModels | KimiModels | DeepSeekModels | ZAIModels | XAIModels


InferenceClass = Literal["priority", "standard", "low"]
"""Scheduling class for a request. ``low`` yields to voice traffic, so it is only
appropriate for work no caller is waiting on.
"""


class ChatCompletionOptions(TypedDict, total=False):
    frequency_penalty: float | None
    logit_bias: dict[str, int] | None
    logprobs: bool | None
    max_completion_tokens: int | None
    max_tokens: int | None
    metadata: Metadata | None
    modalities: list[Literal["text", "audio"]] | None
    n: int | None
    parallel_tool_calls: bool
    prediction: ChatCompletionPredictionContentParam | None
    presence_penalty: float | None
    prompt_cache_key: str
    prompt_cache_retention: Literal["in_memory", "24h"] | None
    reasoning_effort: ReasoningEffort | None
    safety_identifier: str
    seed: int | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    stop: str | None | list[str] | None
    store: bool | None
    temperature: float | None
    top_logprobs: int | None
    top_p: float | None
    user: str
    verbosity: Literal["low", "medium", "high"] | None
    web_search_options: completion_create_params.WebSearchOptions
    tool_choice: ToolChoice


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
ChatCompletionOptions.__module__ = "livekit.agents.inference.llm"
