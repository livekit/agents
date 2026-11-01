# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/worker.rs
#     rust/crates/lk-agents-abi/src/worker/server_host.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .worker_endpoint import EndpointKind


@dataclass(frozen=True, kw_only=True, slots=True)
class JobInfo:
    """The scalar half of a `livekit.agent.Job`."""

    id: str
    dispatch_id: str
    type: int
    room: str | None
    """The room's name, flattened out of `Job.room`."""
    room_sid: str | None
    agent_name: str
    metadata: str
    deployment: str
    enable_recording: bool
    enable_redaction: bool
    proto: bytes
    """The exact `agent.Job` bytes: the fields above are lossy, and this crosses IPC."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RunningJobInfo:
    """`RunningJobInfo`: what a launched job is handed."""

    accept_arguments: AcceptArguments
    job: JobInfo
    url: str
    token: str
    worker_id: str
    fake_job: bool
    received_at: float
    """The dispatch timeline in unix seconds, 0.0 for a job no dispatch offered (simulated):
    the `AvailabilityRequest` arrived,
    """
    accepted_at: float
    """the request was accepted,"""
    assigned_at: float
    """and the `JobAssignment` arrived."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptArguments:
    """Accepted participant fields; application attributes retain their own string keys."""

    name: str
    identity: str
    metadata: str
    attributes: list[tuple[str, str]]
    """Merged after `lk.agent.name`, so a user attribute of that name wins."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ProcessPoolOptions:
    """Process-pool settings resolved before the host launch callback."""

    num_idle_processes: int
    initialize_timeout: float
    close_timeout: float
    session_end_timeout: float
    memory_warn_mb: float
    memory_limit_mb: float
    http_proxy: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class ProcessStartOptions:
    """Resolved server settings handed to the process launch adapter."""

    devmode: bool
    job_processes: bool
    env: list[tuple[str, str]]
    job_executor_type: str
    multiprocessing_context: str
    pool: ProcessPoolOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointRoute:
    """One route of the endpoint manifest the worker registers."""

    path: str
    methods: list[str]
    kind: EndpointKind
    """`"http"`, or `"text"` for a session endpoint."""
    public: bool
