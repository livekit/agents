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
from enum import Enum, unique
from typing import TypeAlias


@dataclass(kw_only=True, slots=True)
class IpcFailureIo:
    kind: str
    os_error: int | None
    message: str


@dataclass(kw_only=True, slots=True)
class IpcFailureMalformed:
    message: str


@dataclass(kw_only=True, slots=True)
class IpcFailureTooLarge:
    size: int


@dataclass(kw_only=True, slots=True)
class IpcFailureUnexpected:
    message: str


@dataclass(kw_only=True, slots=True)
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


@dataclass(kw_only=True, slots=True)
class JobCompletion:
    exit_code: int | None
    failure: IpcFailure | None


@unique
class JobStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


@dataclass(kw_only=True, slots=True)
class JobRunnerOptions:
    """How the worker supervises one job runner. The defaults are the production values."""

    thread: bool = False
    """The runner is a thread of this process, else a process. A thread has no pong deadline
    nor memory limit, whatever is set here: it cannot be stopped.
    """
    initialize_timeout: float
    """How long the runner has to be ready, and later to take its job."""
    close_timeout: float
    """How long a closing runner has to end before it is stopped."""
    memory_warn_mb: float = 0.0
    """The footprint past which a warning is logged; 0: none."""
    memory_limit_mb: float = 0.0
    """The footprint past which the process is killed; 0: none."""
    ping_interval: float = 2.5
    """Seconds between pings; 0: none."""
    ping_timeout: float = 60.0
    """A process silent this long is killed; 0: a late pong is only reported."""
    high_ping_threshold: float = 0.5
    """A pong later than this is reported."""
    http_proxy: str | None = None
    """The job's HTTP proxy, handed over at initialization."""
    asyncio_debug: bool = False
    """The runner's event loop runs in debug mode."""


@dataclass(kw_only=True, slots=True)
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


JobStatus.__module__ = "livekit.agents.ipc"
JobRunnerOptions.__module__ = "livekit.agents.ipc"
