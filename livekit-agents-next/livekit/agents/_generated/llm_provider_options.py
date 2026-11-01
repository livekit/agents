# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/inference.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import Literal

    from openai.types.chat import ChatCompletionPredictionContentParam
    from openai.types.chat.completion_create_params import WebSearchOptions
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import Metadata

    from livekit.agents.llm import ToolChoice


class ChatCompletionOptions(TypedDict, total=False):
    frequency_penalty: float | None
    logit_bias: dict[str, int] | None
    logprobs: bool | None
    max_completion_tokens: int | None
    max_tokens: int | None
    metadata: Metadata | None
    modalities: list[Literal['text', 'audio']] | None
    n: int | None
    parallel_tool_calls: bool
    prediction: ChatCompletionPredictionContentParam | None
    presence_penalty: float | None
    prompt_cache_key: str
    prompt_cache_retention: Literal['in_memory', '24h'] | None
    reasoning_effort: ReasoningEffort | None
    safety_identifier: str
    seed: int | None
    service_tier: Literal['auto', 'default', 'flex', 'scale', 'priority'] | None
    stop: str | None | list[str] | None
    store: bool | None
    temperature: float | None
    top_logprobs: int | None
    top_p: float | None
    user: str
    verbosity: Literal['low', 'medium', 'high'] | None
    web_search_options: WebSearchOptions
    tool_choice: ToolChoice


ChatCompletionOptions.__module__ = "livekit.agents.inference.llm"
