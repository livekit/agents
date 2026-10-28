# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from .recorder import RecordingSummary

class SessionRecorder:
    """One admitted runtime worker owns opening, encoding, writes and finalization."""
    def __init__(
        self,
        path: str,
        *,
        sample_rate: int = 48000,
        t0: float | None = None,
        bitrate: int = 64000,
    ) -> None: ...
    async def ready(self) -> None:
        """Wait until the native file and encoder are open before attaching the tap."""
    @property
    def bytes_written(self) -> int: ...
    @property
    def started_at(self) -> float | None: ...
    @property
    def dropped_samples(self) -> tuple[int, int]: ...
    async def flush(self, now: float) -> RecordingSummary: ...
    async def finish(self, now: float | None = None) -> None: ...
