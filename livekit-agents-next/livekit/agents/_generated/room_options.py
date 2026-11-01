# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import NotRequired

from typing_extensions import TypedDict


class AudioInputConfig(TypedDict):
    """`AudioInputOptions`. Every key is required: the one producer, `RoomOptions._to_core()`,
    writes them all, and a silent default once turned AGC off.
    """

    sample_rate: int
    """24 kHz, as in the reference; the host's stream converts once."""
    num_channels: int
    frame_size_ms: int
    noise_cancellation: object | None
    """A [`HostRef`](lk_agents_abi::HostRef) so the side-table entry lives as long as the config."""
    auto_gain_control: bool
    """Resolved by `_to_core`, applied host-side (`rtc.AudioProcessingModule`), forwarded as
    [`AudioStreamOptions::auto_gain_control`](crate::voice::rtc::AudioStreamOptions).
    """
    pre_connect_audio: bool
    pre_connect_audio_timeout: float
    """Seconds on the wire, clamped at zero."""


class AudioOutputConfig(TypedDict):
    """The host's flat dict over `TrackPublishOptions`: `track_source` is the protobuf int of
    `rtc.TrackSource`, an absent `track_name` keeps the default, and the queue bound is not on the
    wire. Default: 24 kHz mono `roomio_audio` on the microphone source, with the 200 ms queue.
    """

    track_source: int
    track_name: NotRequired[str]
    sample_rate: int
    num_channels: int


class TextOutputConfig(TypedDict):
    """`TextOutputOptions` as the core reads it; `next_in_chain` is the host's alone."""

    sync_transcription: bool | None
    """`None` is `NOT_GIVEN`: synchronise if there is an audio output. `Some(false)` disables it."""
    transcription_speed_factor: float
    json_format: bool
    """JSON per chunk, with start and end timestamps for a `TranscriptChunk`."""


class RoomIoConfig(TypedDict):
    """After `get_*_options()` resolved every `NotGivenOr`; `None` means a leg is off. Read by
    the generated room-I/O boundary converter.
    """

    participant_identity: str | None
    """`None` links to the first acceptable one."""
    participant_kinds: list[int]
    """Empty means `DEFAULT_PARTICIPANT_KINDS` (`[]` is falsy in the reference). A malformed value
    is refused, never read as empty.
    """
    close_on_disconnect: bool
    delete_room_on_close: bool
    text_input: bool
    video_input: bool
    audio_input: AudioInputConfig | None
    audio_output: AudioOutputConfig | None
    text_output: TextOutputConfig | None


class AudioOutputOptionsWire(TypedDict):
    """The SDK's numeric source and public option names, before native publish defaults are applied."""

    track_source: int
    track_name: NotRequired[str]
    sample_rate: int
    num_channels: int
