# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/worker.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

from ..types import NOT_GIVEN, NotGivenOr
from .job_records import JobExecutorType


@dataclass
class WorkerPermissions:
    """`WorkerPermissions`, field-for-field, including the defaults."""

    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_update_metadata: bool = True
    can_publish_sources: list[int] = field(default_factory=list)
    """`models.TrackSource` values, as raw enum numbers."""
    hidden: bool = False


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
    """A sparse update to the server configuration; omission and an explicit null remain distinct."""

    ws_url: NotGivenOr[str] = NOT_GIVEN
    api_key: NotGivenOr[str] = NOT_GIVEN
    api_secret: NotGivenOr[str] = NOT_GIVEN
    agent_name: NotGivenOr[str] = NOT_GIVEN
    listeners: NotGivenOr[Listeners] = NOT_GIVEN
    load_threshold: NotGivenOr[float] = NOT_GIVEN
    num_idle_processes: NotGivenOr[int] = NOT_GIVEN
    drain_timeout: NotGivenOr[float] = NOT_GIVEN
    shutdown_timeout: NotGivenOr[float] = NOT_GIVEN
    max_retry: NotGivenOr[int] = NOT_GIVEN
    permissions: NotGivenOr[WorkerPermissions] = NOT_GIVEN
    http_proxy: NotGivenOr[str | None] = NOT_GIVEN
    version: NotGivenOr[str] = NOT_GIVEN
    job_executor_type: NotGivenOr[JobExecutorType] = NOT_GIVEN
    multiprocessing_context: NotGivenOr[MultiprocessingContext] = NOT_GIVEN
    job_memory_warn_mb: NotGivenOr[float] = NOT_GIVEN
    job_memory_limit_mb: NotGivenOr[float] = NOT_GIVEN
    shutdown_process_timeout: NotGivenOr[float] = NOT_GIVEN
    session_end_timeout: NotGivenOr[float] = NOT_GIVEN
    initialize_process_timeout: NotGivenOr[float] = NOT_GIVEN


Listeners.__module__ = "livekit.agents"
ListenerAddress.__module__ = "livekit.agents"
PublicListener.__module__ = "livekit.agents"
