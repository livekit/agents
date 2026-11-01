# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/fallback.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from livekit.agents._core import (
    DuplexRealtimeModel,
    HostRealtimeModel,
    OpenAIRealtimeModel,
    RealtimeFallbackModel,
)
from livekit.agents.llm.realtime import RealtimeModel


@dataclass(frozen=True, kw_only=True, slots=True)
class AvailabilityChanged:
    """`AvailabilityChangedEvent`, identical in all three fallback adapters: an entry, by position in
    the host's list, went unavailable or recovered. The host maps the index back to its instance.
    """

    index: int
    label: str
    available: bool


RealtimeFallbackMember: TypeAlias = (
    DuplexRealtimeModel | HostRealtimeModel | OpenAIRealtimeModel | RealtimeFallbackModel
    | RealtimeModel
)
"""One realtime fallback member: a native model or a host provider. Host providers open a fresh
adapter on every attempt, so a failed session is never reused by a later attempt.
"""
