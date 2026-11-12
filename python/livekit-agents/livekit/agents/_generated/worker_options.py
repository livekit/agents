# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/worker.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

from ..types import NOT_GIVEN, NotGivenOr
from .job_records import JobExecutorType

MultiprocessingContext: TypeAlias = Literal["spawn", "forkserver"]
"""How job processes start: `forkserver` (Linux only; elsewhere it falls back to `spawn`) or
`spawn`.
"""


class Listeners(TypedDict, total=False):
    """`AgentServer(listeners=)`: public traffic and operational information never share a port
    (docs/server.md "Listeners"). A listener left out keeps its default.
    """

    public: PublicListener
    """The agent's endpoints and HTTP routes, where the worker serves them (a self-hosted
    LiveKit server, the console); `0.0.0.0:8080`.
    """
    health: ListenerAddress
    """`/`, `/ready` and `/worker`; `0.0.0.0:8081`."""
    prometheus: ListenerAddress
    """`/metrics`; off unless given, then `0.0.0.0:9090`."""


class ListenerAddress(TypedDict, total=False):
    """Where a listener binds; a key left out keeps its default."""

    host: str
    port: int


class PublicListener(TypedDict, total=False):
    """The public listener's address and how it reads the client."""

    host: str
    port: int
    proxy_headers: bool
    """Read the client address and scheme from `X-Forwarded-For` and `X-Forwarded-Proto`; only
    behind a proxy that always sets them.
    """


@dataclass
class ServerOptions:
    """`AgentServer(...)`'s options: a sparse update to the server configuration, where omission and
    an explicit null remain distinct. An option left out keeps its default.
    """

    ws_url: NotGivenOr[str] = NOT_GIVEN
    """The LiveKit server URL; default `LIVEKIT_URL`."""
    api_key: NotGivenOr[str] = NOT_GIVEN
    """Default `LIVEKIT_API_KEY`."""
    api_secret: NotGivenOr[str] = NOT_GIVEN
    """Default `LIVEKIT_API_SECRET`."""
    agent_name: NotGivenOr[str] = NOT_GIVEN
    """The agent's name, its partition in a thread and the first segment of its endpoints; default
    `LIVEKIT_AGENT_NAME`. `LIVEKIT_AGENT_NAME_OVERRIDE` wins over it.
    """
    listeners: NotGivenOr[Listeners] = NOT_GIVEN
    """The worker's `public`, `health` and `prometheus` listeners (docs/server.md "Listeners")."""
    load_threshold: NotGivenOr[float] = NOT_GIVEN
    """Load at which the worker stops taking jobs; 0.7 in production."""
    num_idle_processes: NotGivenOr[int] = NOT_GIVEN
    """Job processes kept warm ahead of dispatch; `min(ceil(cpus), 4)` in production, none in
    development.
    """
    drain_timeout: NotGivenOr[float] = NOT_GIVEN
    """Maximum seconds to wait for active work after admission stops; 3600. Zero skips waiting."""
    shutdown_timeout: NotGivenOr[float] = NOT_GIVEN
    """Total process shutdown budget in seconds, draining and cleanup included; 3610. Must
    exceed `drain_timeout`.
    """
    job_memory_warn_mb: NotGivenOr[float] = NOT_GIVEN
    """A job process's memory warning threshold in MB; 1000."""
    job_memory_limit_mb: NotGivenOr[float] = NOT_GIVEN
    """A job process's memory kill threshold in MB; 0, no limit."""


class ServerOptionsKwargs(TypedDict, total=False):
    """`ServerOptions`'s fields as keyword arguments."""

    ws_url: str
    """The LiveKit server URL; default `LIVEKIT_URL`."""
    api_key: str
    """Default `LIVEKIT_API_KEY`."""
    api_secret: str
    """Default `LIVEKIT_API_SECRET`."""
    agent_name: str
    """The agent's name, its partition in a thread and the first segment of its endpoints; default
    `LIVEKIT_AGENT_NAME`. `LIVEKIT_AGENT_NAME_OVERRIDE` wins over it.
    """
    listeners: Listeners
    """The worker's `public`, `health` and `prometheus` listeners (docs/server.md "Listeners")."""
    load_threshold: float
    """Load at which the worker stops taking jobs; 0.7 in production."""
    num_idle_processes: int
    """Job processes kept warm ahead of dispatch; `min(ceil(cpus), 4)` in production, none in
    development.
    """
    drain_timeout: float
    """Maximum seconds to wait for active work after admission stops; 3600. Zero skips waiting."""
    shutdown_timeout: float
    """Total process shutdown budget in seconds, draining and cleanup included; 3610. Must
    exceed `drain_timeout`.
    """
    job_memory_warn_mb: float
    """A job process's memory warning threshold in MB; 1000."""
    job_memory_limit_mb: float
    """A job process's memory kill threshold in MB; 0, no limit."""


@dataclass
class ProcessOptions:
    """The server's process tuning: the launcher sets it (`CliArgs::Start`), never `AgentServer(...)`.
    An option left out keeps its default.
    """

    max_retry: NotGivenOr[int] = NOT_GIVEN
    """Connection attempts to LiveKit before the worker fails; 16."""
    http_proxy: NotGivenOr[str | None] = NOT_GIVEN
    """The proxy the job processes' HTTP clients use; default `HTTPS_PROXY`, else `HTTP_PROXY`."""
    job_executor_type: NotGivenOr[JobExecutorType] = NOT_GIVEN
    """Where jobs run; `"process"`."""
    multiprocessing_context: NotGivenOr[MultiprocessingContext] = NOT_GIVEN
    """How job processes start; `"forkserver"` on Linux, else `"spawn"`."""
    shutdown_process_timeout: NotGivenOr[float] = NOT_GIVEN
    """Seconds a job process has to exit once its job ended; 10."""
    session_end_timeout: NotGivenOr[float] = NOT_GIVEN
    """Seconds a session's end callbacks have to finish; 300."""
    initialize_process_timeout: NotGivenOr[float] = NOT_GIVEN
    """Seconds a job process has to initialize (`prewarm`); 10."""
    ping_timeout: NotGivenOr[float] = NOT_GIVEN
    """Seconds a job process may leave a ping unanswered before it is killed; 60."""


Listeners.__module__ = "livekit.agents"
ListenerAddress.__module__ = "livekit.agents"
PublicListener.__module__ = "livekit.agents"
