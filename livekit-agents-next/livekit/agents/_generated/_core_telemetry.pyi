# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from .._core import ChatItem
from .telemetry import TelemetryWorkObservation

class LoopMonitor:
    """A loop's watchdog: runtime owns thresholds, timing, admission and cleanup, while the host
    supplies asyncio scheduling, stack samples and OpenTelemetry emission.
    """
    def __init__(
        self,
        event_loop: object,
        *,
        warn_threshold: float,
        error_threshold: float,
        tick_interval: float | None,
        name: str,
        emit_spans: bool,
        report: object,
        span_of: object,
    ) -> None: ...
    @property
    def warn_threshold(self) -> float: ...
    @property
    def error_threshold(self) -> float: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

def _core_loop_monitor_thresholds_from_env() -> tuple[float, float] | None: ...

class _core_SessionReportUpload:
    """Credentials for the session recording endpoint and host OTLP exporters."""
    def __init__(self, api_key: str, api_secret: str) -> None: ...
    def bearer(self) -> str:
        """A fresh LiveKit observability credential. No token policy lives in the host adapter."""
    def content_policy(
        self,
        transcript_requested: bool,
        audio_requested: bool,
        capture_content: bool,
        redaction_enabled: bool,
    ) -> tuple[bool, bool]:
        """Recording options are explicit consent for audio; transcript also requires GenAI content
        capture. Project redaction excludes both raw content formats.
        """
    async def transcript(self, items: list[ChatItem]) -> str:
        """Project and serialize a native history on bounded runtime workers, with no host objects
        or Python JSON allocation on the session loop.
        """
    async def upload(
        self,
        base_url: str,
        room_id: str,
        job_id: str,
        simulated: bool,
        redaction_enabled: bool,
        start_time_millis: int,
        transcript: str | None,
        audio_path: str | None,
        disabled: bool,
    ) -> bool:
        """Returns false only when the project disabled recording. Export failures are reported once
        to the caller, which decides where to surface them without failing the conversation.
        """

class _core_TelemetryWorker:
    """Process-owned SDK work. Constructing another handle shares the same queue and worker ceiling."""
    def __init__(self) -> None: ...
    def submit(self, key: str, callback: object, retained_bytes: int) -> bool:
        """Refuses overload or a duplicate live key before retaining the callback. `retained_bytes`
        includes the immutable SDK batch captured by the callback; no producer waits for capacity.
        """
    def reserve(self, key: str, retained_bytes: int) -> bool:
        """Retention admission for an SDK batch queue, independent of cleanup callback slots."""
    def release(self, key: str) -> None: ...
    def record_drop(self, kind: str, count: int) -> None:
        """`kind` is one of `dropped`, `failed`, `conflict`; unknown categories are refused."""
    def admit_metric(self, name: str, attributes: list[tuple[str, str]]) -> bool:
        """Refuse unbounded labels/series before host SDK instrument creation."""
    def wait(self, key: str, timeout: float) -> bool:
        """Off-loop cleanup only. A deadline never detaches work or releases its capacity early."""
    async def wait_async(self, key: str, timeout: float) -> bool: ...
    def drain_observations(self) -> list[TelemetryWorkObservation]:
        """Drain bounded lifecycle observations; timed-out work keeps its captured key and owner."""
    @property
    def diagnostics(self) -> list[tuple[str, int]]:
        """Nonrecursive counters, read without exporting/logging them through this same pipeline."""
