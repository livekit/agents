# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence

from .worker_cli import CliArgs

def job_timeline(
    received_at: float,
    accepted_at: float,
    assigned_at: float,
    launched_at: float,
    entrypoint_started_at: float,
    server_started_at: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, float]]]:
    """A job's dispatch timeline, for its root span: one event per known stage instant (unix
    nanoseconds) and the seconds between adjacent stages as attributes (they sum to the dispatch
    latency). A zero instant is unknown (a simulation, the console, a resumed job) and skipped.
    `server_started_at` is the server's own start (`JobState.started_at`, unix nanoseconds, or
    milliseconds or seconds should that change).
    """

def cpu_count() -> float:
    """Logical CPUs, fractional under a cgroup quota. Reads, starts nothing."""

def endpoint_url_base(ws_url: str, agent_name: str, deployment: str) -> str | None:
    """`https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
    the URL and the agent are known; a worker registered without a deployment is `default`.
    """

def parse_cli(argv: Sequence[str]) -> CliArgs: ...

def entrypoint_module(path: str | None) -> tuple[str, str]: ...

class ExitSignals:
    """The process launcher's signal and final event-loop cleanup owner."""
    def __init__(self, escalate: bool = True) -> None: ...
    def signal(self, exit_served: bool) -> str: ...
    def served(self) -> None: ...
    def draining(self) -> None: ...
    def closing(self) -> None:
        """Arm the native deadline before the host cancels tasks or closes its executor."""
    def closed(self) -> None:
        """Disarm the deadline only after the host event loop has actually closed."""
