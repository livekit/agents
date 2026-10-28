# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/fallback.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class AvailabilityChanged:
    """`AvailabilityChangedEvent`, identical in all three fallback adapters: an entry, by position in
    the host's list, went unavailable or recovered. The host maps the index back to its instance.
    """

    index: int
    label: str
    available: bool
