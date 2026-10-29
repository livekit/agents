# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/cli.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class CliArgsStart:
    entrypoint: str | None
    log_level: str
    log_format: str
    log_style: str
    dev: bool
    cli_addr: str | None
    simulation: bool
    ws_url: str | None
    api_key: str | None
    api_secret: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliArgsConsole:
    entrypoint: str | None
    connect_addr: str
    record: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class CliArgsDownloadFiles:
    pass


CliArgs: TypeAlias = CliArgsStart | CliArgsConsole | CliArgsDownloadFiles
"""The command accepted by the process launcher."""


@dataclass(frozen=True, kw_only=True, slots=True)
class LogFields:
    """What a log line is made of: the record's level, logger, process and time, a `dict` message
    when one was logged, and its extra fields (the record's attributes outside the standard ones),
    as JSON values.
    """

    level: str
    name: str
    pid: int | None
    created: float
    msg: dict[str, object] | None
    extra: dict[str, object]
