# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/turn/keyterms.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class KeytermUpdate:
    """One LLM answer: three disjoint lists of terms."""

    pending: list[str]
    """Distinctive terms seen but not yet trusted -- tracked, not applied."""
    confirm: list[str]
    """Pending terms the transcript has now corroborated -- applied."""
    remove: list[str]
    """Only a spelling the user corrected away; applied terms are otherwise sticky."""


@dataclass(frozen=True, kw_only=True, slots=True)
class KeytermsChanged:
    """The effective list changed, and how."""

    keyterms: list[str]
    """What the STT should now be biased toward."""
    added: list[str]
    removed: list[str]
