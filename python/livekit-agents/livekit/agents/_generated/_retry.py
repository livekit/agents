# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/types/retry.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(kw_only=True, slots=True)
class _core_RetryDecisionStop:
    pass


@dataclass(kw_only=True, slots=True)
class _core_RetryDecisionRaise:
    pass


@dataclass(kw_only=True, slots=True)
class _core_RetryDecisionExhausted:
    message: str


@dataclass(kw_only=True, slots=True)
class _core_RetryDecisionRetry:
    delay: float


_core_RetryDecision: TypeAlias = (
    _core_RetryDecisionStop
    | _core_RetryDecisionRaise
    | _core_RetryDecisionExhausted
    | _core_RetryDecisionRetry
)
