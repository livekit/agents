# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/llm.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal, NotRequired

from typing_extensions import TypedDict


class ToolSpec(TypedDict):
    """Plain data: reflection and invocation stay Python. The host passes the dict of these keys."""

    name: str
    description: NotRequired[str]
    """Defaults to the empty string only when absent; null is invalid."""
    json_schema: object
    """Explicit JSON, including JSON null; a missing key raises."""
    ignore_on_enter: NotRequired[bool]
    """Must not fire on the turn its `beta/` task is entered (`dob.py`). Defaults on omission."""
    provider_tool: NotRequired[ProviderToolDeclaration]
    """Provider-owned wire declaration; absent for a client function. `name` keeps its tool ID."""


class ProviderToolDeclaration(TypedDict):
    """Native transport ownership of an external tool declaration. Unknown/custom host tools
    have no native provider or payload; the canonical object remains in the host tool pool.
    """

    provider: str | None
    payload: object


class ToolFlags(TypedDict, total=False):
    """A struct, not an enum, so axes don't exclude each other."""

    ignore_on_enter: bool
    """Must not fire on the turn its `beta/` task is entered (`dob.py`). Defaults on omission."""


class NamedToolChoice(TypedDict):
    """The named branch of the public tool-choice union, not a provider-owned payload."""

    type: Literal["function"]
    function: Function


class Function(TypedDict):
    name: str


NamedToolChoice.__module__ = "livekit.agents.llm.tool_context"
Function.__module__ = "livekit.agents.llm.tool_context"
