# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from typing import Self

from livekit.agents.http import Multiplexer, SocketProtocol

from ..types import NotGivenOr
from .connect import APIConnectOptions
from .rtc_frames import AudioFrame

class AudioByteStream:
    """Buffers bytes and emits whole frames: fixed-size, or progressive (20 ms first, doubling to
    the target) to cut TTS time-to-first-audio.
    """
    def __new__(
        cls,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int | None = None,
        progressive: bool = False,
    ) -> Self:
        """``samples_per_channel=None`` means 100 ms, as in the reference."""
    def push(self, data: bytes | bytearray | memoryview) -> list[AudioFrame]:
        """Buffer ``data`` and emit every whole frame it completes."""
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
    def __new__(cls, maximum_age: float | None, refresh_on_get: bool) -> Self: ...
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
    def take_to_close(self) -> list[int]:
        """Every connection queued for closing, taken off the queue: the caller now owns each close."""

class ExpFilter:
    def __new__(
        cls,
        alpha: float,
        max_val: NotGivenOr[float] = ...,
        min_val: NotGivenOr[float] = ...,
        initial: NotGivenOr[float] = ...,
    ) -> Self: ...
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
    def __new__(cls, window_size: int) -> Self: ...
    def add_sample(self, sample: float) -> None: ...
    def get_avg(self) -> float: ...
    def reset(self) -> None: ...
    def size(self) -> int: ...

class AudioConverter:
    """`utils.media.AudioConverter`: frames of any rate and channel count, converted to one format
    (resampled, and averaged to mono or copied across channels). A frame already in that format
    passes through untouched; a change of input format ends the previous stream first.
    """
    def __new__(cls, sample_rate: int, num_channels: int) -> Self: ...
    def push(self, frame: AudioFrame) -> list[AudioFrame]:
        """The converted audio `frame` completes, after the tail of a stream it ends."""
    def flush(self) -> list[AudioFrame]:
        """The audio still held by the resampler, at the end of a segment or stream."""

class SocketSession:
    """`_core.SocketSession`: one provider socket stream. `run()` connects through the protocol,
    sends its input, hands it each received frame, keeps the connection alive and rotates it;
    failures retry under `conn_options`.
    """
    def __new__(
        cls,
        host: SocketProtocol,
        conn_options: APIConnectOptions,
        label: str,
        *,
        mux: SocketMux | None = None,
        context_id: str | None = None,
    ) -> Self:
        """`label` names the provider in retries. With `mux`, the stream is the context
        `context_id` on one of its shared sockets.
        """
    def reconnect(self, reason: str) -> None:
        """Rotate the connection now: goodbye, drain, close, connect again. No retry is spent."""
    async def run(self) -> None:
        """Run until the input ended and the provider finished, or the failure that ended it."""
    async def aclose(self) -> None:
        """Stop: the socket is closed, bounded, and `run()` returns."""

class SocketMux:
    """`_core.SocketMux`: the shared sockets of a multiplexing provider, opened on demand, each
    carrying up to `max_contexts` streams and replaced after `max_session_duration`.
    """
    def __new__(cls, host: Multiplexer, *, max_contexts: int | None = None) -> Self: ...
    def invalidate(self) -> None:
        """Options changed: idle sockets close, busy ones close once their streams end."""
    async def aclose(self) -> None:
        """Close every socket, bounded."""

class StaleHandle(RuntimeError):
    """A handle whose originating host no longer owns a live entry."""
