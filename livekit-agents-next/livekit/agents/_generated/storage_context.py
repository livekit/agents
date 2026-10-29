# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/storage/context.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class ContextField:
    """The host's declared field contract and its already validated JSON value."""

    name: str
    schema: object
    value: object


@dataclass(frozen=True, kw_only=True, slots=True)
class ContextManifest:
    """One application's declaration, retained to detect removed or renamed fields on reload."""

    identity: str
    fields: list[str]


@dataclass(frozen=True, kw_only=True, slots=True)
class ContextScope:
    """One scope's committed data. Revisions and schema metadata are separate from app values."""

    scope_id: str
    revision: int
    fields: dict[str, object]
    contracts: dict[str, object]
    manifests: list[ContextManifest]


@dataclass(frozen=True, kw_only=True, slots=True)
class ContextScopeWrite:
    """A scope replacement conditional on the revision this activation loaded."""

    expected_revision: int
    scope: ContextScope


@dataclass(frozen=True, kw_only=True, slots=True)
class AppStatement:
    """One application SQL statement a managed transaction staged for the turn's commit. `params`
    are JSON scalars: null, booleans, numbers and strings.
    """

    sql: str
    params: list[object]


@dataclass(frozen=True, kw_only=True, slots=True)
class ContextValues:
    """Values to hydrate in the host after the core accepted their schema contracts."""

    revision: int
    fields: dict[str, object]
