# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/rtc/mod.rs
#     rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs
#     rust/crates/lk-agents-abi/src/voice/rtc/media.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from inspect import get_annotations
from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import TypedDict

from livekit.rtc import (
    Transcription as _HostTranscription,
    TranscriptionSegment as _HostTranscriptionSegment,
)

if TYPE_CHECKING:
    from livekit.agents.voice.room_io import AudioSource, TrackPublication
    from livekit.rtc import VideoFrameEvent as _HostVideoFrameRef

    from .rtc_frames import AudioFrame


@dataclass(kw_only=True, slots=True)
class TrackPublishOptions:
    """Everything `_ParticipantAudioOutput` needs to publish its track, translated once at construction."""

    name: str
    """`track_name` on the wire; absent keeps the default (`roomio_audio`)."""
    source: int
    """`rtc.TrackSource`'s number, as `rtc.TrackPublishOptions(source=)` takes it."""
    sample_rate: int
    num_channels: int
    queue_size_ms: int
    """`rtc.AudioSource(queue_size_ms=)`: the backpressure bound that paces `_forward_audio`
    against playout (`room_io` passes 200). Not on the wire; a zero default would mean no bound.
    """


@dataclass(kw_only=True, slots=True)
class AudioStreamOptions:
    """`rtc.AudioStream.from_track(...)`'s parameters: what `_core_rtc.TrackPublication.
    _core_open_audio_stream` opens its `rtc.AudioStream` (and, with `auto_gain_control`, its
    `AudioProcessingModule`) with.
    """

    participant_identity: str
    """The participant selected by the room coordinator, for the host's per-track selector callback."""
    sample_rate: int
    """`AudioInputOptions.sample_rate`, default 24000."""
    num_channels: int
    frame_size_ms: int | None
    """`AudioInputOptions.frame_size_ms`, default 50."""
    noise_cancellation: object | None
    """The host-side `rtc.FrameProcessor` / `rtc.NoiseCancellationOptions`, an opaque host object
    Rust never inspects.
    """
    auto_gain_control: bool
    """`AudioInputOptions.auto_gain_control`, default **true**. Performed host-side, per stream."""
    queue_size_ms: int
    """Bounds the SDK queue, as audio, before frames reach the core's equally bounded input
    channel; the host divides it by `frame_size_ms` into its frame count.
    """


@dataclass(kw_only=True, slots=True)
class ProcessorFailure:
    """An input audio processor (noise cancellation, gain control) that raised on a frame."""

    label: str
    """The processor's class, `module.qualname`: what its failing span is tracked by."""
    processor: object
    error: object
    """The exception it raised, handed back as `AudioProcessingError.error`."""
    message: str


@dataclass(kw_only=True, slots=True)
class AudioStreamItemFrame:
    frame: AudioFrame


@dataclass(kw_only=True, slots=True)
class AudioStreamItemProcessorFailed:
    failure: ProcessorFailure


AudioStreamItem: TypeAlias = AudioStreamItemFrame | AudioStreamItemProcessorFailed
"""One item off an input audio stream: a frame, or a processor that failed on the frame that
follows, which arrives as that processor received it.
"""


VideoFrameRef: TypeAlias = "_HostVideoFrameRef"


class StreamTextOptions(TypedDict):
    """`LocalParticipant.stream_text(...)`'s keyword arguments."""

    topic: str
    sender_identity: str | None
    """`None`: the writer defaults to the local participant's identity."""
    attributes: dict[str, str]


class StreamBytesOptions(TypedDict):
    """`LocalParticipant.stream_bytes(...)`'s keyword arguments."""

    name: str
    topic: str
    destination_identities: list[str]
    attributes: dict[str, str]


TranscriptionSegment = _HostTranscriptionSegment


Transcription = _HostTranscription


@dataclass(kw_only=True, slots=True)
class PlaybackInfo:
    """What one `wait_for_playout` covered, measured since the previous one returned: what a host
    audio source's `PlayoutLedger` hands back, and the core reads back unchanged.
    """

    played: timedelta
    """Audio that reached the speaker."""
    discarded: timedelta
    """Audio captured and then thrown away by `clear_queue`: non-zero means the playout was
    interrupted.
    """


@dataclass
class AudioOutputCapabilities:
    pause: bool


@dataclass
class PublishedAudioTrack:
    """A published audio source and the publication controlling its lifetime."""

    source: AudioSource
    publication: TrackPublication


@dataclass(kw_only=True, slots=True)
class VideoTrackOptions:
    name: str
    source: int
    width: int
    height: int


AudioOutputCapabilities.__annotations__ = get_annotations(AudioOutputCapabilities, eval_str=True)
AudioOutputCapabilities.__module__ = "livekit.agents.voice.io"
