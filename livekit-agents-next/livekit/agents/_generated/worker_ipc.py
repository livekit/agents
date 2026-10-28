# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/ipc.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class IpcFailureIo:
    kind: str
    os_error: int | None
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class IpcFailureMalformed:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class IpcFailureTooLarge:
    size: int


@dataclass(frozen=True, kw_only=True, slots=True)
class IpcFailureUnexpected:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class IpcFailureVersion:
    received: int
    expected: int


IpcFailure: TypeAlias = (
    IpcFailureIo
    | IpcFailureMalformed
    | IpcFailureTooLarge
    | IpcFailureUnexpected
    | IpcFailureVersion
)


@dataclass(frozen=True, kw_only=True, slots=True)
class ProcessCompletion:
    exit_code: int | None
    failure: IpcFailure | None
