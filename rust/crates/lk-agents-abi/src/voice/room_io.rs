//! `RoomOptions` as the host resolves it, the room-I/O events, and `PlaybackFinishedEvent`. `RoomIo`
//! and the participant-linking rules live in `lk-agents-voice/src/room_io`.

// `pub use`: an extract record's field types are copied verbatim into the generated module,
// which globs this module to resolve them.
pub use crate::voice::rtc::{
    DisconnectReason, ParticipantIdentity, ParticipantKind, StreamInfo, TrackPublishOptions,
    TrackSource,
};
use lk_agents_host_derive::Host;
use std::time::Duration;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "room_io", path = "livekit.agents.voice.room_io.types"))]
const _: () = ();

// `voice/room_io/types.py`'s option dataclasses; a `///` is the reference's docstring verbatim,
// Rust notes are `//`. `RoomOptions` itself stays hand-written: its resolvers are behaviour.

// `AudioInputOptions`. Every key is required: the one producer, `RoomOptions._to_rust_config()`,
// writes them all, and a silent default once turned AGC off.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    name = "AudioInputOptions"
)]
#[node(interface)]
#[binding(extract, from_items)]
pub struct AudioInputConfig {
    // 24 kHz, as in the reference; the host's stream converts once.
    #[python(default = "24000")]
    pub sample_rate: u32,
    #[python(default = "1")]
    pub num_channels: u16,
    /// The frame size in milliseconds for the audio input.
    #[python(default = "50")]
    pub frame_size_ms: u32,
    // A [`HostRef`](lk_agents_abi::HostRef) so the side-table entry lives as long as the config.
    #[python(
        ty = "rtc.NoiseCancellationOptions | NoiseCancellationSelector | rtc.FrameProcessor[rtc.AudioFrame] | None",
        import = "livekit:rtc",
        default = "None"
    )]
    #[node(ty = "unknown")]
    pub noise_cancellation: Option<lk_agents_abi::HostRef>,
    /// Enable automatic gain control (AGC) on the input audio. Enabled by default.
    //
    // Honoured host-side (`rtc.AudioProcessingModule`), forwarded as
    // [`AudioStreamOptions::auto_gain_control`](crate::voice::rtc::AudioStreamOptions).
    #[python(default = "True")]
    pub auto_gain_control: bool,
    /// Pre-connect audio enabled or not.
    #[python(default = "True")]
    pub pre_connect_audio: bool,
    /// The pre-connect audio will be ignored if it doesn't arrive within this time.
    //
    // Seconds on the wire, clamped at zero by the converter.
    #[python(ty = "float", default = "3.0")]
    #[node(ty = "number")]
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

/// The host's flat dict over `TrackPublishOptions`. Default: 24 kHz mono `roomio_audio` on the
/// microphone source, with the 200 ms queue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioOutputConfig {
    pub publish: TrackPublishOptions,
}

/// `track_source` is the protobuf int of `rtc.TrackSource`; absent `track_name` keeps the default.
/// The queue bound is not on the wire: it keeps `TrackPublishOptions`' own default.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AudioOutputConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{field_error, item, item_with};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "AudioOutputConfigFromPy";
        let obj = &*obj;
        let name = match obj.get_item("track_name") {
            Ok(v) => v.extract::<String>().map_err(|e| field_error(obj.py(), e, OWNER, "track_name"))?,
            Err(_) => TrackPublishOptions::default().name,
        };
        Ok(AudioOutputConfig {
            publish: TrackPublishOptions {
                name,
                source: item_with(obj, OWNER, "track_source", |v| {
                    TrackSource::try_from(v.extract::<i32>()?).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
                })?,
                sample_rate: item(obj, OWNER, "sample_rate")?,
                num_channels: item(obj, OWNER, "num_channels")?,
                ..TrackPublishOptions::default()
            },
        })
    }
}

// `next_in_chain` is the host's own `TextOutput`, a `host_field`.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    dataclass,
    name = "TextOutputOptions",
    host_field(
        name = "next_in_chain",
        ty = "TextOutput | None",
        default = "None",
        doc = "The next text output in the chain for the agent. If provided, the agent's transcription will be passed to it."
    ),
    order = "sync_transcription, transcription_speed_factor, next_in_chain, json_format"
)]
#[node(interface)]
#[binding(extract, from_items)]
pub struct TextOutputConfig {
    /// False to disable transcription synchronization with audio output.
    /// Otherwise, transcription is emitted as quickly as available.
    //
    // `None` is `NOT_GIVEN`: synchronise if there is an audio output. `Some(false)` disables it.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub sync_transcription: Option<bool>,
    /// Speed factor of transcription synchronization with audio output.
    /// Only effective if `sync_transcription` is True.
    #[python(default = "1.0")]
    pub transcription_speed_factor: f64,
    /// Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TimedString.
    #[python(default = "False")]
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

// What a `NoiseCancellationSelector` is called with, host-side.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    name = "NoiseCancellationParams"
)]
#[node(interface)]
pub struct NoiseCancellationParams {
    #[python(ty = "rtc.Participant", import = "livekit:rtc")]
    #[node(ty = "unknown")]
    pub participant: crate::HostRef,
    #[python(ty = "rtc.Track", import = "livekit:rtc")]
    #[node(ty = "unknown")]
    pub track: crate::HostRef,
}

// `_default_text_input_cb` is host behaviour (`voice/room_io/_text_input.py`).
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    name = "TextInputOptions"
)]
#[node(interface)]
pub struct TextInputOptions {
    #[python(
        ty = "TextInputCallback",
        import = "..voice.room_io._text_input:_default_text_input_cb",
        default = "_default_text_input_cb"
    )]
    #[node(ty = "unknown")]
    pub text_input_cb: Option<crate::HostRef>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    dataclass,
    name = "VideoInputOptions"
)]
#[node(interface)]
pub struct VideoInputOptions {}

// `AudioOutputOptions` as written, before the host flattens it. Only `source` of
// `rtc.TrackPublishOptions` crosses.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    name = "AudioOutputOptions"
)]
#[node(interface)]
pub struct AudioOutputOptions {
    #[python(default = "24000")]
    pub sample_rate: u32,
    #[python(default = "1")]
    pub num_channels: u16,
    #[python(
        ty = "rtc.TrackPublishOptions",
        import = "livekit:rtc",
        default_factory = "lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)"
    )]
    #[node(ty = "unknown")]
    pub track_publish_options: Option<crate::HostRef>,
    /// The name of the audio track to publish. If not provided, default to "roomio_audio".
    #[python(ty = "NotGivenOr[str]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "string")]
    pub track_name: Option<String>,
}

// Deprecated but public; `RoomOptions._create_from_legacy` maps it.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    dataclass,
    name = "RoomInputOptions"
)]
#[node(interface)]
pub struct RoomInputOptions {
    /// If not given, default to True.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub text_enabled: Option<bool>,
    /// If not given, default to True.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub audio_enabled: Option<bool>,
    /// If not given, default to False.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub video_enabled: Option<bool>,
    #[python(default = "24000")]
    pub audio_sample_rate: u32,
    #[python(default = "1")]
    pub audio_num_channels: u16,
    /// The frame size in milliseconds for the audio input.
    #[python(default = "50")]
    pub audio_frame_size_ms: u32,
    #[python(
        ty = "rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None",
        import = "livekit:rtc",
        default = "None"
    )]
    #[node(ty = "unknown")]
    pub noise_cancellation: Option<crate::HostRef>,
    #[python(
        ty = "TextInputCallback",
        import = "..voice.room_io._text_input:_default_text_input_cb",
        default = "_default_text_input_cb"
    )]
    #[node(ty = "unknown")]
    pub text_input_cb: Option<crate::HostRef>,
    /// Participant kinds accepted for auto subscription. If not provided,
    /// accept `DEFAULT_PARTICIPANT_KINDS`.
    #[python(
        ty = "NotGivenOr[list[rtc.ParticipantKind.ValueType]]",
        import = "..types:NOT_GIVEN",
        default = "NOT_GIVEN"
    )]
    #[node(ty = "number[]")]
    pub participant_kinds: Option<Vec<ParticipantKind>>,
    /// The participant to link to. If not provided, link to the first participant.
    /// Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`.
    #[python(ty = "NotGivenOr[str]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "string")]
    pub participant_identity: Option<String>,
    /// Pre-connect audio enabled or not.
    #[python(default = "True")]
    pub pre_connect_audio: bool,
    /// The pre-connect audio will be ignored if it doesn't arrive within this time.
    #[python(default = "3.0")]
    pub pre_connect_audio_timeout: f64,
    /// Close the AgentSession if the linked participant disconnects with reasons in
    /// CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED.
    #[python(default = "True")]
    pub close_on_disconnect: bool,
    /// Delete the room when the AgentSession is closed, default to False
    #[python(default = "False")]
    pub delete_room_on_close: bool,
}

#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    dataclass,
    name = "RoomOutputOptions"
)]
#[node(interface)]
pub struct RoomOutputOptions {
    /// If not given, default to True.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub transcription_enabled: Option<bool>,
    /// If not given, default to True.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub audio_enabled: Option<bool>,
    #[python(default = "24000")]
    pub audio_sample_rate: u32,
    #[python(default = "1")]
    pub audio_num_channels: u16,
    #[python(
        ty = "rtc.TrackPublishOptions",
        import = "livekit:rtc",
        default_factory = "lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)"
    )]
    #[node(ty = "unknown")]
    pub audio_publish_options: Option<crate::HostRef>,
    /// The name of the audio track to publish. If not provided, default to "roomio_audio".
    #[python(ty = "NotGivenOr[str]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "string")]
    pub audio_track_name: Option<String>,
    /// False to disable transcription synchronization with audio output.
    /// Otherwise, transcription is emitted as quickly as available.
    #[python(ty = "NotGivenOr[bool]", import = "..types:NOT_GIVEN", default = "NOT_GIVEN")]
    #[node(ty = "boolean")]
    pub sync_transcription: Option<bool>,
    /// Speed factor of transcription synchronization with audio output.
    /// Only effective if `sync_transcription` is True.
    #[python(default = "1.0")]
    pub transcription_speed_factor: f64,
}

/// After `get_*_options()` resolved every `NotGivenOr`; `None` means a leg is off.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomIoConfig {
    /// `None` links to the first acceptable one.
    pub participant_identity: Option<ParticipantIdentity>,
    /// Empty means `DEFAULT_PARTICIPANT_KINDS` (`[]` is falsy in the reference). A malformed value
    /// is refused, never read as empty.
    pub participant_kinds: Vec<ParticipantKind>,
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
    pub text_input: bool,
    pub video_input: bool,
    pub audio_input: Option<AudioInputConfig>,
    pub audio_output: Option<AudioOutputConfig>,
    pub text_output: Option<TextOutputConfig>,
}

/// A malformed `participant_kinds` raises: an empty list would read as "use the defaults" and
/// widen the accept list.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for RoomIoConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{item, item_with};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "RoomIoConfigFromPy";
        let obj = &*obj;
        Ok(RoomIoConfig {
            participant_identity: item_with(obj, OWNER, "participant_identity", |v| Ok(v.extract::<Option<String>>()?.map(ParticipantIdentity)))?,
            participant_kinds: item_with(obj, OWNER, "participant_kinds", |v| {
                v.extract::<Vec<i32>>()?
                    .into_iter()
                    .map(|k| {
                        ParticipantKind::try_from(k).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "{e}: it would widen the auto-link accept list to the default rather than narrow it, \
                                 so it is refused instead"
                            ))
                        })
                    })
                    .collect()
            })?,
            close_on_disconnect: item(obj, OWNER, "close_on_disconnect")?,
            delete_room_on_close: item(obj, OWNER, "delete_room_on_close")?,
            text_input: item(obj, OWNER, "text_input")?,
            video_input: item(obj, OWNER, "video_input")?,
            audio_input: item(obj, OWNER, "audio_input")?,
            audio_output: item(obj, OWNER, "audio_output")?,
            text_output: item(obj, OWNER, "text_output")?,
        })
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "participant_linked")]
pub struct ParticipantLinked {
    pub identity: ParticipantIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "participant_unlinked")]
pub struct ParticipantUnlinked {
    pub identity: ParticipantIdentity,
    // Crosses as its name: the host only logs it.
    pub reason: DisconnectReason,
}

// Also the host's `TextInputEvent`, where the participant and `info` are host SDK objects.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(
    dataclass,
    name = "TextInputEvent",
    order = "text, info, participant"
)]
#[node(interface)]
#[binding(emit, tag = "type", tag_value = "text_input")]
pub struct TextInput {
    pub text: String,
    #[python(ty = "rtc.RemoteParticipant | None", import = "livekit:rtc", default = "None")]
    #[node(ty = "string | null")]
    pub participant: ParticipantIdentity,
    #[python(ty = "rtc.TextStreamInfo | None", import = "livekit:rtc", default = "None")]
    #[node(ty = "unknown")]
    pub info: Box<StreamInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "delete_room_requested")]
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
