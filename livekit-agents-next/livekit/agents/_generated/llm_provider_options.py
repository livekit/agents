# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/inference.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionPredictionContentParam
    from openai.types.chat.completion_create_params import WebSearchOptions
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import Metadata

    from livekit.agents.llm import ToolChoice


@dataclass(frozen=True)
class ProviderEndpoint:
    """An OpenAI-compatible provider the client calls directly with the provider's own key, in place
    of the LiveKit Inference gateway; no LiveKit header is sent.
    """

    base_url: str
    api_key: str
    strict_tool_schema: bool = True
    """Tool schemas in OpenAI's strict form; some OpenAI-compatible providers refuse it."""
    gzip_body: bool = False
    """Gzip-compressed request bodies, for a provider that accepts them."""
    msgpack_body: bool = False
    """Msgpack request bodies, for a provider that accepts them."""


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
    web_search_options: WebSearchOptions
    tool_choice: ToolChoice


ProviderEndpoint.__annotations__ = get_annotations(ProviderEndpoint, eval_str=True)
ProviderEndpoint.__module__ = "livekit.agents.inference.llm"
ChatCompletionOptions.__module__ = "livekit.agents.inference.llm"
