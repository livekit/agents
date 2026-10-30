# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/workflows.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from ..types import NotGivenOr


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkflowInstructions:
    """`WorkflowInstructions` as the host passes it: the two fields a workflow's template reads."""

    persona: NotGivenOr[str] = NOT_GIVEN
    """`NOT_GIVEN` keeps the workflow's built-in persona; `""` removes the section."""
    extra: str = ""


InstructionsArg: TypeAlias = WorkflowInstructions | str
"""A workflow's `instructions: WorkflowInstructions | str`: a
`WorkflowInstructions` fills the built-in template, anything else is fixed instructions.
"""
