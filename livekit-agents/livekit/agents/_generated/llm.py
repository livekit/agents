# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/llm/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .. import _bindings
from ..llm.chat_context import ChatRole


class CompletionUsage(BaseModel):
    completion_tokens: int
    """The number of tokens in the completion."""
    prompt_tokens: int
    """The number of input tokens used (includes cached tokens)."""
    prompt_cached_tokens: int = 0
    """The number of cached input tokens used."""
    cache_creation_tokens: int = 0
    """The number of tokens used to create the cache."""
    cache_read_tokens: int = 0
    """The number of tokens read from the cache."""
    total_tokens: int
    """The total number of tokens used (completion + prompt tokens)."""
    service_tier: str | None = None
    """The service tier used for processing the request (e.g. 'default', 'priority', 'flex').
    Returned by providers that support tiered processing (e.g. OpenAI).
    """


class FunctionToolCall(BaseModel):
    type: Literal["function"] = "function"
    name: str
    arguments: str
    call_id: str
    extra: dict[str, Any] | None = None
    """Provider-specific extra data (e.g., Google thought signatures)."""


class ChoiceDelta(BaseModel):
    role: ChatRole | None = None
    content: str | None = None
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)
    extra: dict[str, Any] | None = None
    """Provider-specific extra data (e.g., Google thought signatures)."""


class ChatChunk(BaseModel):
    id: str
    delta: ChoiceDelta | None = None
    usage: CompletionUsage | None = None

    def has_response(self) -> bool:
        """Whether this chunk delivered generation the caller can see.

        Token counts and provider metadata (a gateway deployment stamp, a thought
        signature) reach the caller without being output: they neither start the
        clock on time-to-first-token nor give a retry anything to duplicate.
        """
        return _bindings.chat_chunk_has_response(self)


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
CompletionUsage.__module__ = "livekit.agents.llm.llm"
FunctionToolCall.__module__ = "livekit.agents.llm.llm"
ChoiceDelta.__module__ = "livekit.agents.llm.llm"
ChatChunk.__module__ = "livekit.agents.llm.llm"
