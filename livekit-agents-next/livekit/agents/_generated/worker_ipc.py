# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/ipc.rs
#     rust/crates/lk-agents-abi/src/worker/job.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

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
class JobCompletion:
    exit_code: int | None
    failure: IpcFailure | None


@dataclass(frozen=True, kw_only=True, slots=True)
class StartJobFields:
    """The process's one job, shared by the generated host boundary and the IPC wire protocol."""

    job: bytes
    name: str
    identity: str
    metadata: str
    url: str
    token: str
    worker_id: str
    fake_job: bool
    received_at: float
    accepted_at: float
    assigned_at: float
    launched_at: float
    traceparent: str | None = None
