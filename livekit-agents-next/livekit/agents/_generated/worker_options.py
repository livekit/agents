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

from pydantic import JsonValue

from ..types import NOT_GIVEN, NotGivenOr


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


@dataclass
class ServerOptions:
    """A sparse update to the server configuration; omission and an explicit null remain distinct."""

    ws_url: NotGivenOr[str] = NOT_GIVEN
    api_key: NotGivenOr[str] = NOT_GIVEN
    api_secret: NotGivenOr[str] = NOT_GIVEN
    agent_name: NotGivenOr[str] = NOT_GIVEN
    listeners: NotGivenOr[JsonValue] = NOT_GIVEN
    """`{"public": {...}, "health": {...}, "prometheus": {...}}`, each `host` and `port`
    (`public` also `proxy_headers`); the worker validates the keys.
    """
    load_threshold: NotGivenOr[float] = NOT_GIVEN
    num_idle_processes: NotGivenOr[int] = NOT_GIVEN
    drain_timeout: NotGivenOr[float] = NOT_GIVEN
    shutdown_timeout: NotGivenOr[float] = NOT_GIVEN
    max_retry: NotGivenOr[int] = NOT_GIVEN
    permissions: NotGivenOr[WorkerPermissions] = NOT_GIVEN
    http_proxy: NotGivenOr[str | None] = NOT_GIVEN
    prometheus_multiproc_dir: NotGivenOr[str | None] = NOT_GIVEN
    version: NotGivenOr[str] = NOT_GIVEN
    job_executor_type: NotGivenOr[str] = NOT_GIVEN
    multiprocessing_context: NotGivenOr[str] = NOT_GIVEN
    job_memory_warn_mb: NotGivenOr[float] = NOT_GIVEN
    job_memory_limit_mb: NotGivenOr[float] = NOT_GIVEN
    shutdown_process_timeout: NotGivenOr[float] = NOT_GIVEN
    session_end_timeout: NotGivenOr[float] = NOT_GIVEN
    initialize_process_timeout: NotGivenOr[float] = NOT_GIVEN
