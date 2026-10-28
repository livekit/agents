# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/types/retry.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class _core_RetryDecisionStop:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class _core_RetryDecisionRaise:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class _core_RetryDecisionExhausted:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class _core_RetryDecisionRetry:
    delay: float


_core_RetryDecision: TypeAlias = (
    _core_RetryDecisionStop
    | _core_RetryDecisionRaise
    | _core_RetryDecisionExhausted
    | _core_RetryDecisionRetry
)
