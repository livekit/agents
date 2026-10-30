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

// The legs of `RoomOptions._to_core()`'s dict, read by `lk-agents-py`'s
// `conversions::room_io_config_from_py`. The public option dataclasses are hand-written in
// `voice/room_io.py`: their resolvers are behaviour.

/// `AudioInputOptions`. Every key is required: the one producer, `RoomOptions._to_core()`,
/// writes them all, and a silent default once turned AGC off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioInputConfig {
    /// 24 kHz, as in the reference; the host's stream converts once.
    pub sample_rate: u32,
    pub num_channels: u16,
    pub frame_size_ms: u32,
    /// A [`HostRef`](lk_agents_abi::HostRef) so the side-table entry lives as long as the config.
    pub noise_cancellation: Option<lk_agents_abi::HostRef>,
    /// The host's session this input feeds, told when one of its input processors fails; Rust
    /// never inspects it.
    pub session: Option<lk_agents_abi::HostRef>,
    /// Resolved by `_to_core`, applied host-side (`rtc.AudioProcessingModule`), forwarded as
    /// [`AudioStreamOptions::auto_gain_control`](crate::voice::rtc::AudioStreamOptions).
    pub auto_gain_control: bool,
    pub pre_connect_audio: bool,
    /// Seconds on the wire, clamped at zero.
    pub pre_connect_audio_timeout: std::time::Duration,
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        AudioInputConfig {
            sample_rate: 24_000,
            num_channels: 1,
            frame_size_ms: 50,
            noise_cancellation: None,
            session: None,
            auto_gain_control: true,
            pre_connect_audio: true,
            pre_connect_audio_timeout: Duration::from_secs(3),
        }
    }
}

/// The host's flat dict over `TrackPublishOptions`: `track_source` is the protobuf int of
/// `rtc.TrackSource`, an absent `track_name` keeps the default, and the queue bound is not on the
/// wire. Default: 24 kHz mono `roomio_audio` on the microphone source, with the 200 ms queue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioOutputConfig {
    pub publish: TrackPublishOptions,
}

/// `TextOutputOptions` as the core reads it; `next_in_chain` is the host's alone.
#[derive(Debug, Clone, PartialEq)]
pub struct TextOutputConfig {
    /// `None` is `NOT_GIVEN`: synchronise if there is an audio output. `Some(false)` disables it.
    pub sync_transcription: Option<bool>,
    pub transcription_speed_factor: f64,
    /// JSON per chunk, with start and end timestamps for a `TimedString`.
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
/// `lk-agents-py`'s `conversions::room_io_config_from_py`.
#[crate::host(handwritten(extract, host = "dict[str, object]", py))]
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
