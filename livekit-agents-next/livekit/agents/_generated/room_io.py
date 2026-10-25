# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field

from livekit import rtc

from ..types import NOT_GIVEN
from ..voice.room_io._text_input import _default_text_input_cb


@dataclass
class AudioInputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    frame_size_ms: int = 50
    """The frame size in milliseconds for the audio input."""
    noise_cancellation: rtc.NoiseCancellationOptions | NoiseCancellationSelector | rtc.FrameProcessor[rtc.AudioFrame] | None = None
    auto_gain_control: bool = True
    """Enable automatic gain control (AGC) on the input audio. Enabled by default."""
    pre_connect_audio: bool = True
    """Pre-connect audio enabled or not."""
    pre_connect_audio_timeout: float = 3.0
    """The pre-connect audio will be ignored if it doesn't arrive within this time."""


@dataclass
class TextOutputOptions:
    sync_transcription: NotGivenOr[bool] = NOT_GIVEN
    """False to disable transcription synchronization with audio output.
    Otherwise, transcription is emitted as quickly as available.
    """
    transcription_speed_factor: float = 1.0
    """Speed factor of transcription synchronization with audio output.
    Only effective if `sync_transcription` is True.
    """
    next_in_chain: TextOutput | None = None
    """The next text output in the chain for the agent. If provided, the agent's transcription will be passed to it."""
    json_format: bool = False
    """Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TimedString."""


@dataclass
class NoiseCancellationParams:
    participant: rtc.Participant
    track: rtc.Track


@dataclass
class TextInputOptions:
    text_input_cb: TextInputCallback = _default_text_input_cb


@dataclass
class VideoInputOptions:
    pass


@dataclass
class AudioOutputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    track_publish_options: rtc.TrackPublishOptions = field(default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""


@dataclass
class RoomInputOptions:
    text_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to True."""
    audio_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to True."""
    video_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to False."""
    audio_sample_rate: int = 24000
    audio_num_channels: int = 1
    audio_frame_size_ms: int = 50
    """The frame size in milliseconds for the audio input."""
    noise_cancellation: rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None = None
    text_input_cb: TextInputCallback = _default_text_input_cb
    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted for auto subscription. If not provided,
    accept `DEFAULT_PARTICIPANT_KINDS`.
    """
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to link to. If not provided, link to the first participant.
    Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`.
    """
    pre_connect_audio: bool = True
    """Pre-connect audio enabled or not."""
    pre_connect_audio_timeout: float = 3.0
    """The pre-connect audio will be ignored if it doesn't arrive within this time."""
    close_on_disconnect: bool = True
    """Close the AgentSession if the linked participant disconnects with reasons in
    CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED.
    """
    delete_room_on_close: bool = False
    """Delete the room when the AgentSession is closed, default to False"""


@dataclass
class RoomOutputOptions:
    transcription_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to True."""
    audio_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to True."""
    audio_sample_rate: int = 24000
    audio_num_channels: int = 1
    audio_publish_options: rtc.TrackPublishOptions = field(default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    audio_track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""
    sync_transcription: NotGivenOr[bool] = NOT_GIVEN
    """False to disable transcription synchronization with audio output.
    Otherwise, transcription is emitted as quickly as available.
    """
    transcription_speed_factor: float = 1.0
    """Speed factor of transcription synchronization with audio output.
    Only effective if `sync_transcription` is True.
    """


@dataclass
class TextInputEvent:
    text: str
    info: rtc.TextStreamInfo | None = None
    participant: rtc.RemoteParticipant | None = None


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
AudioInputOptions.__module__ = "livekit.agents.voice.room_io.types"
TextOutputOptions.__module__ = "livekit.agents.voice.room_io.types"
NoiseCancellationParams.__module__ = "livekit.agents.voice.room_io.types"
TextInputOptions.__module__ = "livekit.agents.voice.room_io.types"
VideoInputOptions.__module__ = "livekit.agents.voice.room_io.types"
AudioOutputOptions.__module__ = "livekit.agents.voice.room_io.types"
RoomInputOptions.__module__ = "livekit.agents.voice.room_io.types"
RoomOutputOptions.__module__ = "livekit.agents.voice.room_io.types"
TextInputEvent.__module__ = "livekit.agents.voice.room_io.types"
