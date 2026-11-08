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
    from .worker_options import MultiprocessingContext


@dataclass(kw_only=True, slots=True)
class ProcessStartOptions:
    """What the host's process launcher needs, resolved by the server before it starts the job
    processes.
    """

    job_processes: bool
    """The server runs jobs: the host returns its launcher factory."""
    env: list[tuple[str, str]]
    """The job processes' environment, beside this process's."""
    multiprocessing_context: MultiprocessingContext
    session_end_timeout: float
    """Seconds a session's end callbacks have to finish."""


@dataclass(kw_only=True, slots=True)
class EndpointRoute:
    """One route of the endpoint manifest the worker registers."""

    path: str
    methods: list[str]
    kind: EndpointKind
    """`"http"`, or `"text"` for a session endpoint."""
    public: bool
