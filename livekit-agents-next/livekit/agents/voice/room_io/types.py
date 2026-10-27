from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from livekit import rtc

from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ..io import TextOutput

if TYPE_CHECKING:
    from ..agent_session import LiveSession

DEFAULT_CLOSE_ON_DISCONNECT_REASONS: list[rtc.DisconnectReason.ValueType] = [
    rtc.DisconnectReason.CLIENT_INITIATED,
    rtc.DisconnectReason.ROOM_DELETED,
    rtc.DisconnectReason.USER_REJECTED,
]


@dataclass
class TextInputEvent:
    text: str
    info: rtc.TextStreamInfo | None = None
    participant: rtc.RemoteParticipant | None = None


TextInputCallback = Callable[["LiveSession", TextInputEvent], Coroutine[None, None, None] | None]


@dataclass
class NoiseCancellationParams:
    participant: rtc.Participant
    track: rtc.Track


NoiseCancellationSelector: TypeAlias = Callable[
    [NoiseCancellationParams],
    rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None,
]


async def _default_text_input_cb(sess: LiveSession, ev: TextInputEvent) -> None:
    async with sess._claim_user_turn():
        await sess.interrupt()
        sess.generate_reply(user_input=ev.text)


@dataclass
class TextInputOptions:
    text_input_cb: TextInputCallback = _default_text_input_cb


@dataclass
class AudioInputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    frame_size_ms: int = 50
    """The frame size in milliseconds for the audio input."""
    noise_cancellation: (
        rtc.NoiseCancellationOptions
        | NoiseCancellationSelector
        | rtc.FrameProcessor[rtc.AudioFrame]
        | None
    ) = None
    auto_gain_control: NotGivenOr[bool] = NOT_GIVEN
    """Enable automatic gain control (AGC) on the input audio.
    If not given, disabled when noise cancellation is configured directly.
    Set explicitly when using a noise cancellation selector."""
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
    track_publish_options: rtc.TrackPublishOptions = field(
        default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""


@dataclass
class TextOutputOptions:
    sync_transcription: NotGivenOr[bool] = NOT_GIVEN
    """False to disable transcription synchronization with audio output.
    Otherwise, transcription is emitted as quickly as available."""
    transcription_speed_factor: float = 1.0
    """Speed factor of transcription synchronization with audio output.
    Only effective if `sync_transcription` is True."""
    next_in_chain: TextOutput | None = None
    """The next text output in the chain for the agent. If provided, the agent's transcription will be passed to it."""
    json_format: bool = False
    """Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TimedString."""


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
    accept `DEFAULT_PARTICIPANT_KINDS`."""
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to link to. If not provided, link to the first participant.
    Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`."""
    close_on_disconnect: bool = True
    """Close the LiveSession if the linked participant disconnects with reasons in
    CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED."""
    delete_room_on_close: bool = False
    """Delete the room when the LiveSession is closed, default to False"""

    def get_text_input_options(self) -> TextInputOptions | None:
        if isinstance(self.text_input, TextInputOptions):
            return self.text_input
        # if text_input is not given, default to enabled
        return TextInputOptions() if self.text_input is not False else None

    def get_audio_input_options(self) -> AudioInputOptions | None:
        if isinstance(self.audio_input, AudioInputOptions):
            return self.audio_input
        # if audio_input is not given, default to enabled
        return AudioInputOptions() if self.audio_input is not False else None

    def get_video_input_options(self) -> VideoInputOptions | None:
        if isinstance(self.video_input, VideoInputOptions):
            return self.video_input
        # if video_input is not given, default to disabled
        return VideoInputOptions() if self.video_input is True else None

    def get_audio_output_options(self) -> AudioOutputOptions | None:
        if isinstance(self.audio_output, AudioOutputOptions):
            return self.audio_output
        return AudioOutputOptions() if self.audio_output is not False else None

    def get_text_output_options(self) -> TextOutputOptions | None:
        if isinstance(self.text_output, TextOutputOptions):
            return self.text_output
        return TextOutputOptions() if self.text_output is not False else None

    def _to_core(self) -> dict[str, Any]:
        """The ``RoomIoConfig`` the core reads: every leg resolved, ``None`` when it is off."""
        audio_in = self.get_audio_input_options()
        audio_out = self.get_audio_output_options()
        text_out = self.get_text_output_options()
        return {
            "participant_identity": (
                self.participant_identity if is_given(self.participant_identity) else None
            ),
            "participant_kinds": (
                list(self.participant_kinds) if is_given(self.participant_kinds) else []
            ),
            "close_on_disconnect": self.close_on_disconnect,
            "delete_room_on_close": self.delete_room_on_close,
            "text_input": self.get_text_input_options() is not None,
            "video_input": self.get_video_input_options() is not None,
            "audio_input": audio_in
            and {
                "sample_rate": audio_in.sample_rate,
                "num_channels": audio_in.num_channels,
                "frame_size_ms": audio_in.frame_size_ms,
                "noise_cancellation": audio_in.noise_cancellation,
                "auto_gain_control": (
                    audio_in.auto_gain_control if is_given(audio_in.auto_gain_control) else True
                ),
                "pre_connect_audio": audio_in.pre_connect_audio,
                "pre_connect_audio_timeout": audio_in.pre_connect_audio_timeout,
            },
            "audio_output": audio_out
            and {
                "sample_rate": audio_out.sample_rate,
                "num_channels": audio_out.num_channels,
                "track_source": audio_out.track_publish_options.source,
                **({"track_name": audio_out.track_name} if is_given(audio_out.track_name) else {}),
            },
            "text_output": text_out
            and {
                "sync_transcription": (
                    text_out.sync_transcription if is_given(text_out.sync_transcription) else None
                ),
                "transcription_speed_factor": text_out.transcription_speed_factor,
                "json_format": text_out.json_format,
            },
        }
