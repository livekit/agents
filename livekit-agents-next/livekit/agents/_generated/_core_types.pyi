# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from livekit.agents.types import APIConnectOptions

from ._retry import _core_RetryDecision
from .stt import SpeechEvent

def api_connect_options_check(
    max_retry: int,
    retry_interval: float,
    pool_timeout: float | None,
    connect_timeout: float | None,
    write_timeout: float | None,
    read_timeout: float | None,
    total_timeout: float | None,
) -> None:
    """The record as the host built it, checked at the boundary; the error names the field."""

def api_retry_interval(retry_interval: float, num_retries: int) -> float:
    """`APIConnectOptions.interval_for_retry` for the host's retry loops; only `retry_interval`
    affects the answer.
    """

class _core_ProviderAttempts:
    """Host callbacks execute attempts authorized by the shared inference request machine."""
    def __init__(self, kind: str, conn_options: APIConnectOptions, now_ns: int) -> None: ...
    def external_retry_owner(self) -> None: ...
    def ready(self, now_ns: int) -> bool: ...
    def failure(
        self,
        error: BaseException,
        produced_output: bool,
        now_ns: int,
    ) -> _core_RetryDecision: ...
    def wake_at(self) -> int | None: ...
    def capacity_exceeded(self, input: bool, now_ns: int) -> None: ...
    def expect_response(self, now_ns: int) -> None: ...
    def input_ended(self, now_ns: int) -> None: ...
    def observe(self, event: SpeechEvent, now_ns: int) -> None: ...
    def observe_response(self, observation: str, now_ns: int) -> None: ...
    def succeeded(self, now_ns: int) -> None: ...
    @property
    def cleanup_timeout(self) -> float: ...
    def cancel(self, now_ns: int) -> None: ...
