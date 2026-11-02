//! `RoomOptions` as the host resolves it, the room-I/O events, and `PlaybackFinishedEvent`. `RoomIo`
//! and the participant-linking rules live in `lk-agents-voice/src/room_io`.

// `pub use`: an extract record's field types are copied verbatim into the generated module,
// which globs this module to resolve them.
pub use crate::voice::rtc::{
    DisconnectReason, ParticipantIdentity, ParticipantKind, StreamInfo, TrackPublishOptions,
    TrackSource,
};
use std::time::Duration;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "room_io", path = "livekit.agents.voice.room_io"))]
const _: () = ();

// Resolved legs of the public RoomOptions. Generated models own their fields; explicit
// host hooks resolve SDK defaults and callbacks before these boundary converters read them.

/// A normalized `AudioInputOptions` view. Generated extraction reads its declared attributes.
#[crate::host(record(
    module = "room_options",
    host_type = "livekit.agents.voice.room_io.AudioInputOptions",
    boundary,
    extract
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioInputConfig {
    /// 24 kHz, as in the reference; the host's stream converts once.
    pub sample_rate: u32,
    pub num_channels: u16,
    pub frame_size_ms: u32,
    /// A [`HostRef`](lk_agents_abi::HostRef) so the side-table entry lives as long as the config.
    pub noise_cancellation: Option<lk_agents_abi::HostRef>,
    /// Resolved by `_to_core`, applied host-side (`rtc.AudioProcessingModule`), forwarded as
    /// [`AudioStreamOptions::auto_gain_control`](crate::voice::rtc::AudioStreamOptions).
    pub auto_gain_control: bool,
    pub pre_connect_audio: bool,
    /// Seconds on the wire, clamped at zero.
    #[host(read_as = f64, normalize = crate::voice::room_io::clamped_audio_timeout)]
    pub pre_connect_audio_timeout: std::time::Duration,
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        AudioInputConfig {
            sample_rate: 24_000,
            num_channels: 1,
            frame_size_ms: 50,
            noise_cancellation: None,
            auto_gain_control: true,
            pre_connect_audio: true,
            pre_connect_audio_timeout: Duration::from_secs(3),
        }
    }
}

/// Public `AudioOutputOptions`, resolved to the native publication configuration.
#[crate::host(record(
    module = "room_options",
    host_type = "livekit.agents.voice.room_io.AudioOutputOptions",
    boundary,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioOutputConfig {
    #[host(flatten, read_as = AudioOutputOptionsWire, try_normalize = crate::voice::room_io::publish_options_from_wire)]
    pub publish: TrackPublishOptions,
}

/// `TextOutputOptions` as the core reads it; `next_in_chain` is the host's alone.
#[crate::host(record(
    module = "room_options",
    host_type = "livekit.agents.voice.room_io.TextOutputOptions",
    boundary,
    extract
))]
#[derive(Debug, Clone, PartialEq)]
pub struct TextOutputConfig {
    /// `None` is `NOT_GIVEN`: synchronise if there is an audio output. `Some(false)` disables it.
    #[host(read_as = crate::Maybe<bool>, normalize = crate::voice::room_io::sync_transcription_from_wire)]
    pub sync_transcription: Option<bool>,
    pub transcription_speed_factor: f64,
    /// JSON per chunk, with start and end timestamps for a `TranscriptChunk`.
    pub json_format: bool,
}

impl Default for TextOutputConfig {
    fn default() -> Self {
        TextOutputConfig {
            sync_transcription: None,
            transcription_speed_factor: 1.0,
            json_format: false,
        }
    }
}

/// After `get_*_options()` resolved every `NotGivenOr`; `None` means a leg is off. Read by
/// the generated room-I/O boundary converter.
#[crate::host(record(
    module = "room_options",
    host_type = "livekit.agents.voice.room_io.RoomOptions",
    boundary,
    extract
))]
#[derive(Debug, Clone, PartialEq)]
pub struct RoomIoConfig {
    /// `None` links to the first acceptable one.
    #[host(read_as = Option<String>, normalize = crate::voice::room_io::participant_identity_from_wire)]
    pub participant_identity: Option<ParticipantIdentity>,
    /// Empty means `DEFAULT_PARTICIPANT_KINDS` (`[]` is falsy in the reference). A malformed value
    /// is refused, never read as empty.
    #[host(read_as = Vec<i32>, try_normalize = crate::voice::room_io::participant_kinds_from_wire)]
    pub participant_kinds: Vec<ParticipantKind>,
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
    pub text_input: bool,
    pub video_input: bool,
    pub audio_input: Option<AudioInputConfig>,
    pub audio_output: Option<AudioOutputConfig>,
    pub text_output: Option<TextOutputConfig>,
}

impl Default for RoomIoConfig {
    fn default() -> Self {
        RoomIoConfig {
            participant_identity: None,
            participant_kinds: Vec::new(),
            close_on_disconnect: true,
            delete_room_on_close: false,
            text_input: true,
            video_input: false,
            audio_input: Some(AudioInputConfig::default()),
            audio_output: Some(AudioOutputConfig::default()),
            text_output: Some(TextOutputConfig::default()),
        }
    }
}

#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticipantLinked {
    pub identity: ParticipantIdentity,
}

#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticipantUnlinked {
    pub identity: ParticipantIdentity,
    // Crosses as its name: the host only logs it.
    pub reason: DisconnectReason,
}

/// `text_input`'s payload: the host builds its `TextInputEvent` from it, with the participant and
/// `info` as its SDK objects.
#[crate::host(record(module = "session_wire", name = "TextInputWire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextInput {
    pub text: String,
    pub participant: ParticipantIdentity,
    pub info: StreamInfo,
}

#[crate::host(record(module = "session_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteRoomRequested {
    pub room: String,
}

/// `playback_position` is a `Duration`: it is summed and compared against deadlines.
// The default is `AudioOutput.__last_playback_ev`'s initial value.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlaybackFinished {
    pub playback_position: Duration,
    /// A `clear_buffer` on the audio output cut the playout short.
    pub interrupted: bool,
    /// Possibly partial; `None` when the sink does not synchronise one.
    pub synchronized_transcript: Option<String>,
}

/// `started_at` is converted against the session's anchor at the seam.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaybackProgressed {
    /// Host `time.time()` seconds.
    pub started_at: f64,
    pub offset: Duration,
    pub duration: Duration,
}

/// The SDK's numeric source, read structurally without copying SDK publication fields.
#[crate::host(record(
    module = "room_options",
    host_type = "livekit.rtc.TrackPublishOptions",
    boundary,
    extract
))]
pub struct AudioTrackPublishOptions {
    pub source: i32,
}

/// Public audio options before native publish defaults are applied.
#[crate::host(record(
    module = "room_options",
    host_type = "livekit.agents.voice.room_io.AudioOutputOptions",
    boundary,
    extract
))]
pub struct AudioOutputOptionsWire {
    pub track_publish_options: AudioTrackPublishOptions,
    pub track_name: crate::Maybe<String>,
    pub sample_rate: u32,
    pub num_channels: u16,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
#[crate::host(error(kind = "RuntimeError"))]
pub struct RoomOptionsError(pub String);

pub fn clamped_audio_timeout(value: f64) -> Duration {
    Duration::from_secs_f64(value.max(0.0))
}

pub fn publish_options_from_wire(
    value: AudioOutputOptionsWire,
) -> Result<TrackPublishOptions, RoomOptionsError> {
    Ok(TrackPublishOptions {
        name: value.track_name.unwrap_or(TrackPublishOptions::default().name),
        source: TrackSource::try_from(value.track_publish_options.source)
            .map_err(|e| RoomOptionsError(e.to_string()))?,
        sample_rate: value.sample_rate,
        num_channels: value.num_channels,
        ..Default::default()
    })
}

pub fn sync_transcription_from_wire(value: crate::Maybe<bool>) -> Option<bool> {
    value.given()
}

pub fn participant_identity_from_wire(value: Option<String>) -> Option<ParticipantIdentity> {
    value.map(ParticipantIdentity)
}

pub fn participant_kinds_from_wire(
    values: Vec<i32>,
) -> Result<Vec<ParticipantKind>, RoomOptionsError> {
    values.into_iter().map(|value| ParticipantKind::try_from(value).map_err(|error| RoomOptionsError(format!(
        "{error}: it would widen the auto-link accept list to the default rather than narrow it, so it is refused instead"
    )))).collect()
}

/// An application callback for incoming room text; its default runs on the owning session.
#[crate::host(record(
    module = "room_api",
    path = "livekit.agents.voice.room_io",
    dataclass,
    deferred,
    host_fields(text_input_cb(
        ty = "collections.abc.Callable[[..voice.agent_session.VoiceSession, .room_input_events.TextInputEvent], collections.abc.Coroutine[None, None, None] | None]",
        default = ".._bindings.default_text_input_cb"
    ))
))]
pub struct TextInputOptions {}

// Public data contracts. Python behavior remains in its adapters.

/// An input audio processor failed on a span; the session forwarded that span unprocessed
///     and kept listening.
#[crate::host(record(
    module = "audio_errors",
    path = "livekit.agents.voice.room_io",
    pydantic,
    arbitrary_types,
    constant(type = "audio_processing_error")
))]
pub struct AudioProcessingError {
    #[host(default_factory = "time.time")]
    pub timestamp: f64,
    pub label: String,
    #[host(annotation = "typing.Annotated[Exception, pydantic.Field(exclude=True)]")]
    pub error: crate::HostRef,
    #[host(default = true)]
    pub recoverable: bool,
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "room_input_events",
    path = "livekit.agents.voice.room_io",
    dataclass
))]
pub struct TextInputEvent {
    pub text: String,
    #[host(default = None, annotation = "rtc.TextStreamInfo | None")]
    pub info: Option<crate::HostRef>,
    #[host(default = None, annotation = "rtc.RemoteParticipant | None")]
    pub participant: Option<crate::HostRef>,
}

#[crate::host(record(
    module = "room_input_events",
    path = "livekit.agents.voice.room_io",
    dataclass
))]
pub struct NoiseCancellationParams {
    #[host(annotation = "rtc.Participant")]
    pub participant: crate::HostRef,
    #[host(annotation = "rtc.Track")]
    pub track: crate::HostRef,
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "room_api",
    path = "livekit.agents.voice.room_io",
    dataclass,
    deferred
))]
pub struct AudioInputOptions {
    #[host(default = 24000)]
    pub sample_rate: i64,
    #[host(default = 1)]
    pub num_channels: i64,
    /// The frame size in milliseconds for the audio input.
    #[host(default = 50)]
    pub frame_size_ms: i64,
    #[host(default = None, annotation = "rtc.NoiseCancellationOptions | collections.abc.Callable[[.room_input_events.NoiseCancellationParams], rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None] | rtc.FrameProcessor[rtc.AudioFrame] | None")]
    pub noise_cancellation: Option<crate::HostRef>,
    /// Enable automatic gain control (AGC) on the input audio.
    ///     If not given, disabled when noise cancellation is configured directly.
    ///     Set explicitly when using a noise cancellation selector.
    pub auto_gain_control: crate::Maybe<bool>,
    /// Pre-connect audio enabled or not.
    #[host(default = true)]
    pub pre_connect_audio: bool,
    /// The pre-connect audio will be ignored if it doesn't arrive within this time.
    #[host(default = 3.0)]
    pub pre_connect_audio_timeout: f64,
}

#[crate::host(record(
    module = "room_api",
    path = "livekit.agents.voice.room_io",
    dataclass,
    deferred
))]
pub struct VideoInputOptions {}

#[crate::host(record(
    module = "room_api",
    path = "livekit.agents.voice.room_io",
    dataclass,
    deferred
))]
pub struct AudioOutputOptions {
    #[host(default = 24000)]
    pub sample_rate: i64,
    #[host(default = 1)]
    pub num_channels: i64,
    #[host(
        default_factory = "livekit.rtc.TrackPublishOptions(source=livekit.rtc.TrackSource.SOURCE_MICROPHONE)",
        annotation = "rtc.TrackPublishOptions"
    )]
    pub track_publish_options: crate::HostRef,
    /// The name of the audio track to publish. If not provided, default to "roomio_audio".
    pub track_name: crate::Maybe<String>,
}

#[crate::host(record(
    module = "room_api",
    path = "livekit.agents.voice.room_io",
    dataclass,
    deferred
))]
pub struct TextOutputOptions {
    /// False to disable transcription synchronization with audio output.
    ///     Otherwise, transcription is emitted as quickly as available.
    pub sync_transcription: crate::Maybe<bool>,
    /// Speed factor of transcription synchronization with audio output.
    ///     Only effective if `sync_transcription` is True.
    #[host(default = 1.0)]
    pub transcription_speed_factor: f64,
    /// The next text output in the chain for the agent. If provided, the agent's transcription will be passed to it.
    #[host(default = None, annotation = "livekit.agents.voice.io.TextOutput | None")]
    pub next_in_chain: Option<crate::HostRef>,
    /// Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TranscriptChunk.
    #[host(default = false)]
    pub json_format: bool,
}

#[crate::host(record(
    module = "room_api",
    path = "livekit.agents.voice.room_io",
    dataclass,
    deferred
))]
#[python(method(
    name = "get_text_input_options",
    ret = "TextInputOptions | None",
    hook = "room_options_get_text_input_options"
))]
#[python(method(
    name = "get_audio_input_options",
    ret = "AudioInputOptions | None",
    hook = "room_options_get_audio_input_options"
))]
#[python(method(
    name = "get_video_input_options",
    ret = "VideoInputOptions | None",
    hook = "room_options_get_video_input_options"
))]
#[python(method(
    name = "get_audio_output_options",
    ret = "AudioOutputOptions | None",
    hook = "room_options_get_audio_output_options"
))]
#[python(method(
    name = "get_text_output_options",
    ret = "TextOutputOptions | None",
    hook = "room_options_get_text_output_options"
))]
#[python(method(name = "_to_core", ret = "RoomOptions", hook = "room_options_to_core"))]
pub struct RoomOptions {
    /// The text input options. If not provided, default to True.
    #[host(annotation = "..types.NotGivenOr[TextInputOptions | bool]")]
    pub text_input: crate::Maybe<crate::HostRef>,
    /// The audio input options. If not provided, default to True.
    #[host(annotation = "..types.NotGivenOr[AudioInputOptions | bool]")]
    pub audio_input: crate::Maybe<crate::HostRef>,
    /// The video input options. If not provided, default to False.
    #[host(annotation = "..types.NotGivenOr[VideoInputOptions | bool]")]
    pub video_input: crate::Maybe<crate::HostRef>,
    /// The audio output options. If not provided, default to True.
    #[host(annotation = "..types.NotGivenOr[AudioOutputOptions | bool]")]
    pub audio_output: crate::Maybe<crate::HostRef>,
    /// The transcription output options. If not provided, default to True.
    #[host(annotation = "..types.NotGivenOr[TextOutputOptions | bool]")]
    pub text_output: crate::Maybe<crate::HostRef>,
    /// Participant kinds accepted for auto subscription. If not provided,
    ///     accept `DEFAULT_PARTICIPANT_KINDS`.
    #[host(annotation = "..types.NotGivenOr[list[rtc.ParticipantKind.ValueType]]")]
    pub participant_kinds: crate::Maybe<Vec<i32>>,
    /// The participant to link to. If not provided, link to the first participant.
    ///     Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`.
    pub participant_identity: crate::Maybe<String>,
    /// Close the VoiceSession if the linked participant disconnects with reasons in
    ///     CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED.
    #[host(default = true)]
    pub close_on_disconnect: bool,
    /// Delete the room when the VoiceSession is closed, default to False
    #[host(default = false)]
    pub delete_room_on_close: bool,
}
