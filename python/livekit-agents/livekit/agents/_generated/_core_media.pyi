# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from typing import Literal, Self

from livekit.agents.tts import SynthesizedAudio
from livekit.agents.utils.aio import Chan
from livekit.rtc import VideoFrame

from .chat_items import InferenceDetail
from .llm import SerializedImage
from .rtc_frames import AudioFrame
from .transcript import TranscriptChunk
from .usage import TokenUsage

async def encode_image(
    frame: VideoFrame,
    *,
    format: Literal["JPEG", "PNG"] = "JPEG",
    quality: int | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
    resize_strategy: Literal["center_aspect_fit", "center_aspect_cover", "scale_aspect_fit", "scale_aspect_cover", "skew"] | None = None,
) -> bytes: ...

async def serialize_image_frame(
    frame: VideoFrame,
    inference_detail: InferenceDetail,
    inference_width: int | None = None,
    inference_height: int | None = None,
) -> SerializedImage:
    """A frame as JPEG bytes, resized to its inference size when it has one."""

async def serialize_image_url(
    url: str,
    inference_detail: InferenceDetail,
    mime_type: str | None,
) -> SerializedImage:
    """A URL's inline bytes and MIME type, or the URL itself when it is external."""

class AudioEmitter:
    def __new__(cls, *, label: str, dst_ch: Chan[SynthesizedAudio]) -> Self: ...
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
    def push_frame(self, frame: AudioFrame) -> None: ...
    def push_timed_transcript(
        self,
        delta_text: TranscriptChunk | list[TranscriptChunk],
    ) -> None: ...
    def report_usage(
        self,
        usage: TokenUsage | None = None,
        *,
        characters_count: int | None = None,
    ) -> None:
        """What the provider reported it billed for the open segment, or the next one when none is
        open, carried on the segment's final frame. Reports add up, one per provider request;
        providers that report nothing keep the framework's own measure (characters sent, audio
        received).
        """
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

class AudioStreamDecoder:
    """`utils.media.AudioStreamDecoder`: container bytes decoded on one admitted native worker into
    PCM frames, which come out as soon as they are decoded. Input is queued without waiting for
    the decoder. One decoder decodes one stream. A failure (an unknown format, no worker admitted,
    invalid audio) is logged and ends the stream.
    """
    def __new__(
        cls,
        *,
        sample_rate: int | None = 48000,
        num_channels: int = 1,
        format: str | None = None,
    ) -> Self:
        """`sample_rate` is the output rate (the source's when `None`), `format` the input's media
        type (`"audio/mpeg"`), probed when `None`.
        """
    def push(self, chunk: bytes | bytearray | memoryview | list[int]) -> None:
        """Queue `chunk`; ignored after `end_input`, `aclose` or the end of the stream."""
    def end_input(self) -> None:
        """No more input: the stream ends once everything pushed is decoded."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> AudioFrame:
        """The next decoded frame, at the requested format; `None` ends the stream."""
    async def aclose(self) -> None:
        """Cancel decoding and observe release of the native decoder's worker admission."""
