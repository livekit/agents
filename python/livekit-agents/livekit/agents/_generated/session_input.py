# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/session/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .delegation_remote import RemoteTaskRef


@dataclass(kw_only=True, slots=True)
class InputRequestRefAgent:
    agent_id: str
    request_id: str


@dataclass(kw_only=True, slots=True)
class InputRequestRefRemoteTask:
    remote_id: str
    task: RemoteTaskRef
    request_id: str


InputRequestRef: TypeAlias = InputRequestRefAgent | InputRequestRefRemoteTask
"""Owner-qualified input occurrence. Local continuation and remote transport remain distinct."""
