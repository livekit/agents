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
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .worker_options import ProcessOptions, ServerOptions

LogLevel: TypeAlias = Literal["trace", "debug", "info", "warn", "error", "fatal", "off"]
"""The launcher's `--log-level`; `warning` and `critical` are aliases of `warn` and `fatal`."""


@dataclass(kw_only=True, slots=True)
class CliArgsStart:
    log_level: LogLevel
    dev: bool
    cli_addr: str | None
    simulation: bool
    options: ServerOptions
    """The server options the command line gives (`--url`, `--api-key`, `--api-secret`,
    `--shutdown-timeout`, `--drain-timeout`), over the `AgentServer`'s own.
    """
    process: ProcessOptions
    """The process tuning the command line gives."""


@dataclass(kw_only=True, slots=True)
class CliArgsConsole:
    connect_addr: str
    record: bool


CliArgs: TypeAlias = CliArgsStart | CliArgsConsole
"""The command accepted by the process launcher."""


@dataclass(kw_only=True, slots=True)
class CliParseResultParsed:
    args: CliArgs


@dataclass(kw_only=True, slots=True)
class CliParseResultDisplay:
    message: str
    exit_code: int
    stderr: bool


CliParseResult: TypeAlias = CliParseResultParsed | CliParseResultDisplay
"""Parsing never prints or exits the hosting process."""


LogStyle: TypeAlias = Literal["json", "dev", "console"]
"""How a log line is written: `json` in production, colored lines for a person reading them."""


@dataclass(kw_only=True, slots=True)
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
