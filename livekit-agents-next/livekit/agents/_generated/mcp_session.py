# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/mcp.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

McpCleanupStatus: TypeAlias = Literal["pending", "clean", "incomplete"]
"""An observed cleanup outcome; a deadline never implies successful cleanup."""


@dataclass(frozen=True, kw_only=True, slots=True)
class McpCleanup:
    """Evidence retained by the session owner after close, including native child wait receipts."""

    status: McpCleanupStatus
    service_settled: bool
    calls_settled: bool
    reader_settled: bool
    child_exit_code: int | None
    child_exit_signal: int | None
    child_reaped: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class McpInvocation:
    """The host context captured at the MCP operation's entry."""

    host_context: object | None
