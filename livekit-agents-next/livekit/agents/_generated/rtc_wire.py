# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/rtc.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import get_type_hints


@dataclass(frozen=True, kw_only=True, slots=True)
class TrackPublishOptions:
    """Everything `_ParticipantAudioOutput` needs to publish its track, translated once at construction."""

    name: str
    """`track_name` on the wire; absent keeps the default (`roomio_audio`)."""
    source: str
    """Crosses as its name; the host adapter maps it back to `rtc.TrackSource`."""
    sample_rate: int
    num_channels: int
    queue_size_ms: int
    """`rtc.AudioSource(queue_size_ms=)`: the backpressure bound that paces `_forward_audio`
    against playout (`room_io` passes 200). Not on the wire; a zero default would mean no bound.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
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
    session: object | None
    """The host's session the stream feeds, told when an input processor fails on a span; an
    opaque host object Rust never inspects.
    """
    auto_gain_control: bool
    """`AudioInputOptions.auto_gain_control`, default **true**. Performed host-side, per stream."""
    queue_capacity: int
    """Bounds the SDK queue before frames reach the core's equally bounded input queue."""


@dataclass(frozen=True, kw_only=True, slots=True)
class VideoFrameRef:
    """One frame off an `rtc.VideoStream`, as a host `io.VideoInput` answers `_core_next_frame()`: a
    [`MediaRef`](lk_agents_abi::MediaRef) (the pixels never cross), the size read at the seam to
    reject non-frames, and the frame's own capture stamp.
    """

    frame: object
    """The side-table key; the host frame is released when the last clone drops."""
    width: int
    height: int
    timestamp_us: int = 0
    """`VideoFrameEvent.timestamp_us`, the host's."""


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamTextOptions:
    """`LocalParticipant.stream_text(...)`'s keyword arguments."""

    topic: str
    sender_identity: str | None
    """Absent means omit the key: the writer then defaults to the local participant's identity."""
    attributes: dict[str, str]


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamBytesOptions:
    """`LocalParticipant.stream_bytes(...)`'s keyword arguments."""

    name: str
    topic: str
    destination_identities: list[str]
    attributes: dict[str, str]


@dataclass(frozen=True, kw_only=True, slots=True)
class TranscriptionSegment:
    """`rtc.TranscriptionSegment`."""

    id: str
    text: str
    start_time: int
    """Milliseconds. `room_io` always writes 0 for both; kept because the field is on the wire."""
    end_time: int
    language: str
    final: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class Transcription:
    """`rtc.Transcription`, the argument to `publish_transcription`."""

    participant_identity: str
    track_sid: str
    segments: list[TranscriptionSegment]


@dataclass(frozen=True, kw_only=True, slots=True)
class PlaybackInfo:
    """What one `wait_for_playout` covered, measured since the previous one returned: what a host
    audio source's `PlayoutLedger` hands back, and the core reads back unchanged.
    """

    played: float
    """Audio that reached the speaker."""
    discarded: float
    """Audio captured and then thrown away by `clear_queue`: non-zero means the playout was
    interrupted.
    """


@dataclass
class AudioOutputCapabilities:
    pause: bool


AudioOutputCapabilities.__annotations__ = get_type_hints(AudioOutputCapabilities)
AudioOutputCapabilities.__module__ = "livekit.agents.voice.io"
