# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from livekit import rtc
from livekit.agents.utils.aio import Chan

class AudioEmitter:
    def __init__(self, *, label: str, dst_ch: Chan) -> None: ...
    def initialize(
        self,
        *,
        request_id: str,
        sample_rate: int,
        num_channels: int,
        mime_type: str,
        frame_size_ms: int = 200,
        stream: bool = False,
    ) -> None: ...
    def start_segment(self, *, segment_id: str) -> None: ...
    def end_segment(self) -> None: ...
    def push(self, data: bytes | bytearray | memoryview | list[int]) -> None: ...
    def push_frame(self, frame: rtc.AudioFrame) -> None: ...
    def push_timed_transcript(
        self,
        delta_text: livekit.agents.types.TimedString | list[livekit.agents.types.TimedString],
    ) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def join(self) -> None:
        """Waits for compressed input to finish decoding; reports a decode failure."""
    async def aclose(self) -> None:
        """Discards pending audio and observes native settlement; never reports a synthesis failure."""
    def pushed_duration(self, idx: int = -1) -> float: ...
    @property
    def num_segments(self) -> int: ...
    def _note_provider_request_id(self, context_id: str) -> None: ...
    @property
    def output_closed(self) -> bool: ...

class AudioDecoder:
    """Container bytes decoded on one admitted native worker: input is queued without waiting for
    the decoder, and decoded frames are awaited.
    """
    def __init__(
        self,
        mime: str,
        *,
        sample_rate: int | None = 48000,
        num_channels: int = 1,
    ) -> None: ...
    def push(self, data: bytes | bytearray | memoryview | list[int]) -> None:
        """Refused after `end_input`, `aclose` or a decode failure."""
    def end_input(self) -> None:
        """No more input: `next` returns `None` once everything pushed is decoded."""
    async def next(self) -> rtc.AudioFrame | None:
        """The next decoded frame, at the requested format; `None` at the end of input."""
    async def aclose(self) -> None:
        """Cancel decoding and observe release of the native decoder's worker admission."""

class StreamBuffer:
    """A byte pipe: normal EOF drains buffered bytes; close discards them immediately."""
    def __init__(self) -> None: ...
    def write(self, data: bytes | bytearray | memoryview | list[int]) -> None: ...
    def end_input(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    def read(self, size: int = -1) -> bytes: ...

def split_sentences(
    text: str,
    min_sentence_len: int = 20,
    *,
    retain_format: bool = False,
) -> list[tuple[str, int, int]]: ...
