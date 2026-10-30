# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/cli.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class CliArgsStart:
    log_level: str
    dev: bool
    cli_addr: str | None
    simulation: bool
    ws_url: str | None
    api_key: str | None
    api_secret: str | None
    shutdown_timeout: float | None
    drain_timeout: float | None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliArgsConsole:
    connect_addr: str
    record: bool


CliArgs: TypeAlias = CliArgsStart | CliArgsConsole
"""The command accepted by the process launcher."""


@dataclass(frozen=True, kw_only=True, slots=True)
class CliParseResultParsed:
    args: CliArgs


@dataclass(frozen=True, kw_only=True, slots=True)
class CliParseResultDisplay:
    message: str
    exit_code: int
    stderr: bool


CliParseResult: TypeAlias = CliParseResultParsed | CliParseResultDisplay
"""Parsing never prints or exits the hosting process."""


@dataclass(frozen=True, kw_only=True, slots=True)
class LogRecord:
    """A language host's normalized process log record."""

    level: str
    logger: str
    process_id: int | None
    timestamp: float
    """Unix seconds."""
    message: str
    fields: dict[str, object]
    exception: str | None
    stack: str | None
