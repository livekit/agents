# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
# and the `#[host(export)]` / `#[host(object)]` items in `EXPORT_SOURCES`.
#
# The stub of every `_core` class and function the binding generates rather than
# writes. `_core.pyi` re-exports this file.
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from typing import Any

WARM_TRANSFER_IDENTITY: str
TWILIO_RINGING_TIMEOUT: float
def rtc_wire_names() -> tuple[list[str], list[str], list[str], list[str], list[str]]: ...
def video_queue_frames() -> int: ...
TELEMETRY_MAX_DEPTH: int
TELEMETRY_MAX_EVENTS: int
SDK_RECORD_BYTES: int
SDK_PENDING_RECORDS: int
SDK_BATCH_ITEMS: int
SDK_SCHEDULE_DELAY_MILLIS: int
SDK_EXPORT_TIMEOUT_MILLIS: int
SDK_MAX_PROVIDERS: int
SDK_SHUTDOWN_TIMEOUT_SECONDS: float
UPLOAD_TIMEOUT_SECONDS: int
REDACTED_EXCEPTION: str
REDACTED_LOG_BODY: str


class TrackPublishOptions:
    """The host's view: what `_core_rtc.LocalParticipant._core_publish_audio_track` builds its
    `rtc.AudioSource`, `LocalAudioTrack` and `rtc.TrackPublishOptions` from.
    """

    @property
    def name(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def queue_size_ms(self) -> int: ...


class AudioStreamOptions:
    """The host's view: what `_core_rtc.TrackPublication._core_open_audio_stream` opens its
    `rtc.AudioStream` (and, with `auto_gain_control`, its `AudioProcessingModule`) with.
    """

    @property
    def participant_identity(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def frame_size_ms(self) -> int | None: ...
    @property
    def queue_capacity(self) -> int: ...
    @property
    def noise_cancellation(self) -> Any | None: ...
    @property
    def auto_gain_control(self) -> bool: ...


class VideoFrameRef:
    """What a host `io.VideoInput` answers `_core_next_frame()` with: its frame (interned; only the key
    crosses back) and the size it read off it.
    """

    def __init__(self, frame: Any, width: int, height: int, timestamp_us: int = 0) -> None: ...


class StreamTextOptions:
    """The host's view: `_core_rtc.LocalParticipant._core_stream_text` spreads it into
    `stream_text(..)`, leaving `sender_identity` out when it is `None` (an explicit `None` would
    publish the user's transcript as the agent).
    """

    @property
    def topic(self) -> str: ...
    @property
    def sender_identity(self) -> str | None: ...
    @property
    def attributes(self) -> dict[str, str]: ...


class StreamBytesOptions:
    """The host's view: `_core_rtc.LocalParticipant._core_stream_bytes` spreads it into
    `stream_bytes(..)`.
    """

    @property
    def name(self) -> str: ...
    @property
    def topic(self) -> str: ...
    @property
    def destination_identities(self) -> list[str]: ...
    @property
    def attributes(self) -> dict[str, str]: ...


class Transcription:
    """The host's view: `_core_rtc.LocalParticipant._core_publish_transcription` builds the
    `rtc.Transcription` and its `rtc.TranscriptionSegment`s (keyword dicts, `final` spelled plainly).
    """

    @property
    def participant_identity(self) -> str: ...
    @property
    def track_sid(self) -> str: ...
    @property
    def segments(self) -> list[dict[str, Any]]: ...


class PlaybackInfo:
    """What a host audio source's `PlayoutLedger` hands back, and the core reads back unchanged."""

    @property
    def played(self) -> float: ...
    @property
    def discarded(self) -> float: ...


class AudioOutputCapabilities:
    """What a host `io.AudioOutput` answers its `_core_capabilities()` with."""

    def __init__(self, pause: bool) -> None: ...
    @property
    def pause(self) -> bool: ...


class RoomEventBus:
    @property
    def subscriber_count(self) -> int: ...


class ChatRequest:
    """The request as a host `LLM._core_chat` reads it: a `_core.ChatRequest` handle over a clone. The
    host objects cross as their side-table keys (`_core.resolve_host_ref`), and the context as its
    `ChatContext.to_dict()` form, read only when the request carries no host context of its own.
    """

    @property
    def chat_ctx_host(self) -> int | None: ...
    @property
    def chat_ctx(self) -> Any: ...
    @property
    def tool_names(self) -> list[str]: ...
    @property
    def function_schemas(self) -> list[dict[str, Any]]: ...
    @property
    def tools_host(self) -> int | None: ...
    @property
    def tools_host_extra(self) -> list[int]: ...
    @property
    def parallel_tool_calls(self) -> bool | None: ...
    @property
    def tool_choice(self) -> Any | None: ...
    @property
    def extra(self) -> dict[str, Any] | None: ...
    @property
    def inference_class(self) -> str | None: ...


class StaleHandle(RuntimeError):
    """A handle whose originating host no longer owns a live entry."""


class SessionClosing(RuntimeError):
    """`wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing."""
