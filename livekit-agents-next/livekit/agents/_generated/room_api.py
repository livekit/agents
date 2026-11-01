# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import livekit.rtc

from .._bindings import default_text_input_cb
from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from livekit import rtc
    from livekit.agents.voice.io import TextOutput

    from ..types import NotGivenOr
    from ..voice.agent_session import VoiceSession as HostVoiceSession
    from .room_input_events import NoiseCancellationParams, TextInputEvent


@dataclass
class TextInputOptions:
    """An application callback for incoming room text; its default runs on the owning session."""
    text_input_cb: Callable[[HostVoiceSession, TextInputEvent], Coroutine[None, None, None] | None] = default_text_input_cb


@dataclass
class AudioInputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    frame_size_ms: int = 50
    """The frame size in milliseconds for the audio input."""
    noise_cancellation: rtc.NoiseCancellationOptions | Callable[[NoiseCancellationParams], rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None] | rtc.FrameProcessor[rtc.AudioFrame] | None = None
    auto_gain_control: NotGivenOr[bool] = NOT_GIVEN
    """Enable automatic gain control (AGC) on the input audio.
        If not given, disabled when noise cancellation is configured directly.
        Set explicitly when using a noise cancellation selector.
    """
    pre_connect_audio: bool = True
    """Pre-connect audio enabled or not."""
    pre_connect_audio_timeout: float = 3.0
    """The pre-connect audio will be ignored if it doesn't arrive within this time."""


@dataclass
class VideoInputOptions:
    pass


@dataclass
class AudioOutputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    track_publish_options: rtc.TrackPublishOptions = field(default_factory=lambda: livekit.rtc.TrackPublishOptions(source=livekit.rtc.TrackSource.SOURCE_MICROPHONE))
    track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""


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
    """Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TranscriptChunk."""


@dataclass
class RoomOptions:
    text_input: NotGivenOr[TextInputOptions | bool] = NOT_GIVEN
    """The text input options. If not provided, default to True."""
    audio_input: NotGivenOr[AudioInputOptions | bool] = NOT_GIVEN
    """The audio input options. If not provided, default to True."""
    video_input: NotGivenOr[VideoInputOptions | bool] = NOT_GIVEN
    """The video input options. If not provided, default to False."""
    audio_output: NotGivenOr[AudioOutputOptions | bool] = NOT_GIVEN
    """The audio output options. If not provided, default to True."""
    text_output: NotGivenOr[TextOutputOptions | bool] = NOT_GIVEN
    """The transcription output options. If not provided, default to True."""
    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted for auto subscription. If not provided,
        accept `DEFAULT_PARTICIPANT_KINDS`.
    """
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to link to. If not provided, link to the first participant.
        Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`.
    """
    close_on_disconnect: bool = True
    """Close the VoiceSession if the linked participant disconnects with reasons in
        CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED.
    """
    delete_room_on_close: bool = False
    """Delete the room when the VoiceSession is closed, default to False"""

    def get_text_input_options(self) -> TextInputOptions | None:
        from .. import _bindings

        return _bindings.room_options_get_text_input_options(self)

    def get_audio_input_options(self) -> AudioInputOptions | None:
        from .. import _bindings

        return _bindings.room_options_get_audio_input_options(self)

    def get_video_input_options(self) -> VideoInputOptions | None:
        from .. import _bindings

        return _bindings.room_options_get_video_input_options(self)

    def get_audio_output_options(self) -> AudioOutputOptions | None:
        from .. import _bindings

        return _bindings.room_options_get_audio_output_options(self)

    def get_text_output_options(self) -> TextOutputOptions | None:
        from .. import _bindings

        return _bindings.room_options_get_text_output_options(self)

    def _to_core(self) -> dict[str, Any]:
        from .. import _bindings

        return _bindings.room_options_to_core(self)


TextInputOptions.__module__ = "livekit.agents.voice.room_io"
AudioInputOptions.__module__ = "livekit.agents.voice.room_io"
VideoInputOptions.__module__ = "livekit.agents.voice.room_io"
AudioOutputOptions.__module__ = "livekit.agents.voice.room_io"
TextOutputOptions.__module__ = "livekit.agents.voice.room_io"
RoomOptions.__module__ = "livekit.agents.voice.room_io"
