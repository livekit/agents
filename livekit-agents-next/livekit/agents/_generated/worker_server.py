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
    from .job_records import JobExecutorType
    from .worker_endpoint import EndpointKind
    from .worker_options import MultiprocessingContext


@dataclass(kw_only=True, slots=True)
class ProcessPoolOptions:
    """Process-pool settings resolved before the host launch callback."""

    num_idle_processes: int
    initialize_timeout: float
    close_timeout: float
    session_end_timeout: float
    memory_warn_mb: float
    memory_limit_mb: float
    http_proxy: str | None


@dataclass(kw_only=True, slots=True)
class ProcessStartOptions:
    """Resolved server settings handed to the process launch adapter."""

    devmode: bool
    job_processes: bool
    env: list[tuple[str, str]]
    job_executor_type: JobExecutorType
    multiprocessing_context: MultiprocessingContext
    pool: ProcessPoolOptions


@dataclass(kw_only=True, slots=True)
class EndpointRoute:
    """One route of the endpoint manifest the worker registers."""

    path: str
    methods: list[str]
    kind: EndpointKind
    """`"http"`, or `"text"` for a session endpoint."""
    public: bool
