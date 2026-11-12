# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/search.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from pydantic import JsonValue

if TYPE_CHECKING:
    from livekit.agents._generated.toolset_events import SearchItem as _HostSearchFields

SearchFields: TypeAlias = "_HostSearchFields"


@dataclass
class ToolProxyCall:
    """The tool a `call_tool` names and the parameters it passes (an object, or its JSON string)."""

    name: str
    parameters: JsonValue
