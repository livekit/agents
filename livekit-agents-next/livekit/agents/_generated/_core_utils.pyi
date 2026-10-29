# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from livekit.rtc import AudioFrame

from ..types import NotGivenOr

class AudioByteStream:
    """Buffers bytes and emits whole frames: fixed-size, or progressive (20 ms first, doubling to
    the target) to cut TTS time-to-first-audio.
    """
    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int | None = None,
        progressive: bool = False,
    ) -> None:
        """``samples_per_channel=None`` means 100 ms, as in the reference."""
    def push(self, data: bytes | bytearray | memoryview) -> list[AudioFrame]:
        """Buffer ``data`` and emit every whole frame it completes."""
    def write(self, data: bytes | bytearray | memoryview) -> list[AudioFrame]:
        """Public alias of ``push`` in the reference."""
    def flush(self) -> list[AudioFrame]:
        """Emit the buffered whole samples, short of a frame or not; a trailing partial sample stays
        for the next push.
        """
    @property
    def buffered_duration(self) -> float:
        """Seconds of audio waiting for the next frame."""
    def reset_progressive(self) -> None:
        """Restart progressive sizing, keeping the buffered audio: after a mid-stream flush, the
        next burst starts with small frames without losing a partial sample.
        """
    def clear(self) -> None:
        """Drop everything buffered and restart progressive sizing (called on interruption, so the
        next word is not a full frame late).
        """

class ConnectionPoolState:
    def __init__(self, maximum_age: float | None, refresh_on_get: bool) -> None: ...
    @property
    def generation(self) -> int: ...
    @property
    def last_acquire_time(self) -> float: ...
    @property
    def last_connection_reused(self) -> bool: ...
    def connected(
        self,
        id: int,
        now: float,
        generation: int,
        acquire_time: float | None,
    ) -> bool: ...
    def empty(self) -> bool: ...
    def get(self, now: float) -> int | None: ...
    def put(self, id: int) -> None: ...
    def remove(self, id: int) -> None: ...
    def invalidate(self) -> None: ...
    def close_all(self) -> None: ...
    def next_closing(self) -> int | None: ...
    def closed(self, id: int) -> None: ...

class ExpFilter:
    def __init__(
        self,
        alpha: float,
        max_val: NotGivenOr[float] = ...,
        min_val: NotGivenOr[float] = ...,
        initial: NotGivenOr[float] = ...,
    ) -> None: ...
    def reset(
        self,
        alpha: NotGivenOr[float] = ...,
        initial: NotGivenOr[float] = ...,
        min_val: NotGivenOr[float] = ...,
        max_val: NotGivenOr[float] = ...,
    ) -> None: ...
    def apply(self, exp: float, sample: NotGivenOr[float] = ...) -> float: ...
    @property
    def value(self) -> float | None: ...
    def update_base(self, alpha: float) -> None: ...

class MovingAverage:
    def __init__(self, window_size: int) -> None: ...
    def add_sample(self, sample: float) -> None: ...
    def get_avg(self) -> float: ...
    def reset(self) -> None: ...
    def size(self) -> int: ...
