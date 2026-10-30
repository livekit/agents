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
