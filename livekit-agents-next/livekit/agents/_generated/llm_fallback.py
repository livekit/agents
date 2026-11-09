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
from typing import TYPE_CHECKING, TypeAlias

from livekit.agents._core import (
    DuplexRealtimeModel,
    HostRealtimeModel,
    OpenAIRealtimeModel,
    RealtimeFallbackModel,
)
from livekit.agents.llm.realtime import RealtimeModel

if TYPE_CHECKING:
    from builtins import object as _HostHostChatContext, object as _HostHostTools


@dataclass(kw_only=True, slots=True)
class AvailabilityChanged:
    """`AvailabilityChangedEvent`, identical in all three fallback adapters: an entry, by position in
    the host's list, went unavailable or recovered. The host maps the index back to its instance.
    """

    index: int
    label: str
    available: bool


HostTools: TypeAlias = "_HostHostTools"


HostChatContext: TypeAlias = "_HostHostChatContext"


RealtimeLeg: TypeAlias = (
    DuplexRealtimeModel | HostRealtimeModel | OpenAIRealtimeModel | RealtimeFallbackModel
    | RealtimeModel
)
"""A realtime model as the core takes it (a voice session's, a fallback member): a native model
or a host provider. A fallback opens a host provider afresh on every attempt, so a failed
session is never reused by a later attempt.
"""
