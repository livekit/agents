# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from typing import Self

from livekit.agents._bindings import AttemptHost

from .connect import APIConnectOptions
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

class ProviderAttempts:
    """The attempts of one host provider request (`livekit.agents.ProviderAttempts`).

    `run()` drives them: it waits until an attempt may start, runs the host's attempt, races it
    against the read and total deadlines, and on failure retries, raises or stops as the shared
    attempt policy decides. Inside a session the session owns retries (`core = True`): `run()`
    then runs exactly one attempt, still bounded by its deadlines.

    The observation methods feed the read deadline while an attempt runs: a provider calls
    `expect_response()` when it sent something that owes an answer, `progress()` when response
    data arrived (heartbeats are not progress), `audio_received()` for synthesized audio and
    `complete_response()` when nothing is owed any more.
    """
    def __new__(cls, kind: str, conn_options: APIConnectOptions, host: AttemptHost) -> Self: ...
    @property
    def core(self) -> bool:
        """Whether the session owns retries; set before the request starts."""
    @core.setter
    def core(self, core: bool) -> None: ...
    async def run(self) -> object | None:
        """Run the request's attempts. Returns the successful attempt's value (`None` when the
        provider stopped the request, a 499); raises the attempt's own error when it is not
        retried, or `APIConnectionError` from the last error once the retries are exhausted.
        Each attempt runs under the caller's current span.
        """
    def expect_response(self, *, final: bool = False) -> None:
        """A request was sent that owes a response; `final`: it was the last input."""
    def input_ended(self) -> None:
        """The input ended: its final response is owed."""
    def progress(self) -> None:
        """Response data arrived: the read deadline restarts."""
    def audio_received(self) -> None:
        """Synthesized audio arrived: it settles the first response, and after the final input it
        is progress.
        """
    def complete_response(self) -> None:
        """No response is outstanding: the read deadline disarms."""
    def observe(self, event: SpeechEvent) -> None:
        """A recognition event, for the stream's read deadline and transcript completion."""
    @property
    def exhausted(self) -> bool:
        """Whether retries were exhausted: `run()` then raised `APIConnectionError` from the last
        attempt's error, which the request's ending `error` event carries.
        """
    async def settled(self) -> None:
        """After `run()` was cancelled and the host cancelled its running attempt: wait until that
        attempt finished its cleanup, at most `cleanup_timeout` (a stalled cleanup is logged and
        keeps running). `_bindings.run_attempts` does both.
        """
    @property
    def cleanup_timeout(self) -> float:
        """How long a cancelled attempt's cleanup may take before it is reported as stalled."""
