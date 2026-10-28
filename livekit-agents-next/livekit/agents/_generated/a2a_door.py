# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/door.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .a2a_card import EndpointDescriptor
    from .storage_options import TurnOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class DoorConfig:
    lane: str
    holder: str
    endpoint: EndpointDescriptor
    options: TurnOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class DoorFrame:
    json: str
    sse: str
    is_error: bool
