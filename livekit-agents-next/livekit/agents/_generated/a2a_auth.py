# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/auth.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class AuthGrant:
    all: bool = False
    scope: str = ""
    thread_ids: list[str]
    list: bool = False
    create: bool = False
    write: bool = False
    read: bool = False
    delete: bool = False


AuthOperation: TypeAlias = Literal["create", "list", "read", "write", "delete"]


PrincipalKind: TypeAlias = Literal["project", "grant"]


@dataclass(frozen=True, kw_only=True, slots=True)
class AuthPrincipal:
    kind: PrincipalKind
    grant: AuthGrant | None
    identity: str
    claims: dict[str, object]
    endpoint_call: bool
    """An endpoint call grant does not itself permit reading or changing a thread."""
