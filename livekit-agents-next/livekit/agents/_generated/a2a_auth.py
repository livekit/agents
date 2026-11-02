# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/auth.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from inspect import get_annotations
from typing import Literal, TypeAlias

from .storage_class import ThreadInfo


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentThreadGrant:
    """A grant's authoring surface; the native core owns every permission decision."""

    all: bool = False
    scope: str = ""
    thread_ids: Sequence[str] = field(default_factory=tuple)
    list: bool = False
    create: bool = False
    write: bool = False
    read: bool = False
    delete: bool = False

    def __post_init__(self) -> None:
        from .. import _bindings

        _bindings.auth_grant_post_init(self)

    def can_create(self, scope: str | None = None) -> bool:
        from .. import _bindings

        return _bindings.auth_grant_can_create(self, scope)

    def can_list(self, scope: str) -> bool:
        from .. import _bindings

        return _bindings.auth_grant_can_list(self, scope)

    def can_read(self, thread: ThreadInfo) -> bool:
        from .. import _bindings

        return _bindings.auth_grant_can_read(self, thread)

    def can_write(self, thread: ThreadInfo) -> bool:
        from .. import _bindings

        return _bindings.auth_grant_can_write(self, thread)

    def can_delete(self, thread: ThreadInfo) -> bool:
        from .. import _bindings

        return _bindings.auth_grant_can_delete(self, thread)


AuthOperation: TypeAlias = Literal["create", "list", "read", "write", "delete"]


PrincipalKind: TypeAlias = Literal["project", "grant"]


@dataclass(frozen=True, kw_only=True, slots=True)
class Principal:
    """The verified endpoint caller, with endpoint and thread authority kept independent."""

    kind: PrincipalKind
    grant: AgentThreadGrant | None
    identity: str
    claims: dict[str, object]
    endpoint_call: bool
    """An endpoint call grant does not itself permit reading or changing a thread."""


AgentThreadGrant.__annotations__ = get_annotations(AgentThreadGrant, eval_str=True)
AgentThreadGrant.__module__ = "livekit.agents.a2a.auth"
Principal.__module__ = "livekit.agents.a2a.auth"
