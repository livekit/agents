from __future__ import annotations

from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from ...types import NOT_GIVEN, NotGivenOr

if TYPE_CHECKING:
    from ..agent_session import AgentSession


DEFAULT_PARTICIPANT_KINDS: list[rtc.ParticipantKind.ValueType] = [
    rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
    rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
]

DEFAULT_CLOSE_ON_DISCONNECT_REASONS: list[rtc.DisconnectReason.ValueType] = [
    rtc.DisconnectReason.CLIENT_INITIATED,
    rtc.DisconnectReason.ROOM_DELETED,
    rtc.DisconnectReason.USER_REJECTED,
]


@dataclass
class TextInputEvent:
    text: str
    info: rtc.TextStreamInfo
    participant: rtc.RemoteParticipant


TextInputCallback = Callable[
    ["AgentSession", TextInputEvent], Optional[Coroutine[None, None, None]]
]


def _default_text_input_cb(sess: AgentSession, ev: TextInputEvent) -> None:
    sess.interrupt()
    sess.generate_reply(user_input=ev.text)


class _BaseOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextInputOptions(_BaseOptions):
    enabled: bool = True
    text_input_cb: TextInputCallback = _default_text_input_cb


class AudioInputOptions(_BaseOptions):
    enabled: bool = True
    sample_rate: int = 24000
    num_channels: int = 1
    noise_cancellation: rtc.NoiseCancellationOptions | None = None
    pre_connect_audio: bool = True
    """Pre-connect audio enabled or not."""
    pre_connect_audio_timeout: float = 3.0
    """The pre-connect audio will be ignored if it doesn't arrive within this time."""


class VideoInputOptions(_BaseOptions):
    enabled: bool = True


class AudioOutputOptions(_BaseOptions):
    enabled: bool = True
    sample_rate: int = 24000
    num_channels: int = 1
    track_publish_options: rtc.TrackPublishOptions = Field(
        default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""


class TranscriptionOutputOptions(_BaseOptions):
    enabled: bool = True
    sync_transcription: NotGivenOr[bool] = NOT_GIVEN
    """False to disable transcription synchronization with audio output.
    Otherwise, transcription is emitted as quickly as available."""
    transcription_speed_factor: float = 1.0
    """Speed factor of transcription synchronization with audio output.
    Only effective if `sync_transcription` is True."""


class RoomOptions(_BaseOptions):
    text_input: NotGivenOr[TextInputOptions | bool] = NOT_GIVEN
    """The text input options. If not provided, default to True."""
    audio_input: NotGivenOr[AudioInputOptions | bool] = NOT_GIVEN
    """The audio input options. If not provided, default to True."""
    video_input: NotGivenOr[VideoInputOptions | bool] = NOT_GIVEN
    """The video input options. If not provided, default to False."""
    audio_output: NotGivenOr[AudioOutputOptions | bool] = NOT_GIVEN
    """The audio output options. If not provided, default to True."""
    transcription_output: NotGivenOr[TranscriptionOutputOptions | bool] = NOT_GIVEN
    """The transcription output options. If not provided, default to True."""

    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted for auto subscription. If not provided,
    accept `DEFAULT_PARTICIPANT_KINDS`."""
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to link to. If not provided, link to the first participant.
    Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`."""
    close_on_disconnect: bool = True
    """Close the AgentSession if the linked participant disconnects with reasons in
    CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED."""

    def ensure_text_input_options(self) -> TextInputOptions:
        if isinstance(self.text_input, TextInputOptions):
            return self.text_input
        return TextInputOptions(enabled=self.text_input is not False)

    def ensure_audio_input_options(self) -> AudioInputOptions:
        if isinstance(self.audio_input, AudioInputOptions):
            return self.audio_input
        return AudioInputOptions(enabled=self.audio_input is not False)

    def ensure_video_input_options(self) -> VideoInputOptions:
        if isinstance(self.video_input, VideoInputOptions):
            return self.video_input
        return VideoInputOptions(enabled=self.video_input is True)

    def ensure_audio_output_options(self) -> AudioOutputOptions:
        if isinstance(self.audio_output, AudioOutputOptions):
            return self.audio_output
        return AudioOutputOptions(enabled=self.audio_output is not False)

    def ensure_transcription_output_options(self) -> TranscriptionOutputOptions:
        if isinstance(self.transcription_output, TranscriptionOutputOptions):
            return self.transcription_output
        return TranscriptionOutputOptions(enabled=self.transcription_output is not False)

    @classmethod
    def _from_legacy(
        cls,
        input_options: NotGivenOr[RoomInputOptions],
        output_options: NotGivenOr[RoomOutputOptions],
    ) -> RoomOptions:
        from ..agent_session import AgentSession  # noqa: F401

        opts = cls()
        if input_options:
            opts.text_input = TextInputOptions(
                enabled=input_options.text_enabled is not False,
                text_input_cb=input_options.text_input_cb,
            )
            opts.audio_input = AudioInputOptions(
                enabled=input_options.audio_enabled is not False,
                sample_rate=input_options.audio_sample_rate,
                num_channels=input_options.audio_num_channels,
                noise_cancellation=input_options.noise_cancellation,
                pre_connect_audio=input_options.pre_connect_audio,
                pre_connect_audio_timeout=input_options.pre_connect_audio_timeout,
            )
            opts.video_input = VideoInputOptions(
                enabled=input_options.video_enabled is True,
            )
            opts.close_on_disconnect = input_options.close_on_disconnect

        if output_options:
            opts.audio_output = AudioOutputOptions(
                enabled=output_options.audio_enabled is not False,
                sample_rate=output_options.audio_sample_rate,
                num_channels=output_options.audio_num_channels,
                track_publish_options=output_options.audio_publish_options,
                track_name=output_options.audio_track_name,
            )
            opts.transcription_output = TranscriptionOutputOptions(
                enabled=output_options.transcription_enabled is not False,
                sync_transcription=output_options.sync_transcription,
                transcription_speed_factor=output_options.transcription_speed_factor,
            )
        return opts


# RoomInputOptions and RoomOutputOptions are deprecated


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
    noise_cancellation: rtc.NoiseCancellationOptions | None = None
    text_input_cb: TextInputCallback = _default_text_input_cb
    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted for auto subscription. If not provided,
    accept `DEFAULT_PARTICIPANT_KINDS`."""
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to link to. If not provided, link to the first participant.
    Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`."""
    pre_connect_audio: bool = True
    """Pre-connect audio enabled or not."""
    pre_connect_audio_timeout: float = 3.0
    """The pre-connect audio will be ignored if it doesn't arrive within this time."""
    close_on_disconnect: bool = True
    """Close the AgentSession if the linked participant disconnects with reasons in
    CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED."""


@dataclass
class RoomOutputOptions:
    transcription_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to True."""
    audio_enabled: NotGivenOr[bool] = NOT_GIVEN
    """If not given, default to True."""
    audio_sample_rate: int = 24000
    audio_num_channels: int = 1
    audio_publish_options: rtc.TrackPublishOptions = field(
        default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    audio_track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""
    sync_transcription: NotGivenOr[bool] = NOT_GIVEN
    """False to disable transcription synchronization with audio output.
    Otherwise, transcription is emitted as quickly as available."""
    transcription_speed_factor: float = 1.0
    """Speed factor of transcription synchronization with audio output.
    Only effective if `sync_transcription` is True."""


# DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()
# DEFAULT_ROOM_OUTPUT_OPTIONS = RoomOutputOptions()
