//! Values exchanged through the RTC capability boundary. Opaque video references keep frame
//! resources owned by the host.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use lk_agents_abi::{AudioFrame, HostRef, MediaRef};
use lk_agents_host_derive::Host;

/// A participant's stable identity string. `room_io/` keys everything on this and never on `sid`.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[crate::host(string)]
pub struct ParticipantIdentity(pub String);

macro_rules! string_newtype {
    ($t:ident) => {
        impl $t {
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl FromStr for $t {
            type Err = std::convert::Infallible;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ok($t(s.to_owned()))
            }
        }
        impl From<&str> for $t {
            fn from(s: &str) -> Self {
                $t(s.to_owned())
            }
        }
        impl From<String> for $t {
            fn from(s: String) -> Self {
                $t(s)
            }
        }
        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}
string_newtype!(ParticipantIdentity);

/// A track publication's server-assigned id; its own type so it cannot be swapped with an identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[crate::host(string)]
pub struct TrackSid(pub String);

string_newtype!(TrackSid);

/// `wire()` is the index into `ALL`; `TryFrom<i32>` refuses a value this build has no variant for.
macro_rules! wire_i32 {
    ($t:ident, $kind:literal) => {
        impl $t {
            pub fn wire(self) -> i32 {
                Self::ALL.iter().position(|v| *v == self).expect("every variant is in ALL") as i32
            }
        }
        impl TryFrom<i32> for $t {
            type Error = UnknownWireValue;
            fn try_from(value: i32) -> Result<Self, UnknownWireValue> {
                usize::try_from(value)
                    .ok()
                    .and_then(|i| Self::ALL.get(i).copied())
                    .ok_or(UnknownWireValue { kind: $kind, value })
            }
        }
    };
}

/// Integer nanoseconds: deadlines on [`Ts`](lk_agents_abi::Ts) are compared for equality, so no
/// float round-trip. Zero when `sample_rate` is zero.
pub fn frame_duration(frame: &AudioFrame) -> Duration {
    if frame.sample_rate == 0 {
        return Duration::ZERO;
    }
    Duration::from_nanos(
        u64::from(frame.samples_per_channel)
            .saturating_mul(1_000_000_000)
            .saturating_div(u64::from(frame.sample_rate)),
    )
}

/// `_input.py` pushes half a second of it when a track ends, to flush a pending STT final.
pub fn silence_frame(samples_per_channel: u32, sample_rate: u32, num_channels: u16) -> AudioFrame {
    let bytes = samples_per_channel as usize * num_channels as usize * size_of::<i16>();
    AudioFrame::new(vec![0u8; bytes], sample_rate, num_channels, samples_per_channel)
}

/// `rtc.ConnectionState`. `room_io.py` reads it through `isconnected()`; the third state is what
/// makes that a lossy read, so both are kept. Crosses as its name (`"connected"`); the host
/// adapter folds a value this build has no variant for to `disconnected`, the safe read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Default)]
#[crate::host(wire_enum)]
pub enum ConnectionState {
    /// The honest read of a room nobody has reported on (and of a failed read).
    #[default]
    Disconnected,
    Connected,
    Reconnecting,
}

wire_i32!(ConnectionState, "ConnectionState");

impl ConnectionState {
    pub fn is_connected(self) -> bool {
        matches!(self, ConnectionState::Connected)
    }
}

/// `rtc.DisconnectReason`, verbatim. Crosses as [`as_str`](Self::as_str)'s name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
// `rtc.DisconnectReason`'s member, lower-cased, in wire order.
#[crate::host(string)]
#[crate::host(wire_strs(err = UnknownDisconnectReason, msg = UnknownDisconnectReason(s.to_owned())))]
pub enum DisconnectReason {
    Unknown,
    ClientInitiated,
    DuplicateIdentity,
    ServerShutdown,
    ParticipantRemoved,
    RoomDeleted,
    StateMismatch,
    JoinFailure,
    Migration,
    SignalClose,
    RoomClosed,
    UserUnavailable,
    UserRejected,
    SipTrunkFailure,
    ConnectionTimeout,
    MediaFailure,
    AgentError,
}

wire_i32!(DisconnectReason, "DisconnectReason");

impl DisconnectReason {
    pub fn closes_session_by_default(self) -> bool {
        matches!(
            self,
            DisconnectReason::ClientInitiated
                | DisconnectReason::RoomDeleted
                | DisconnectReason::UserRejected
        )
    }
}

/// The RTC shim folds it to a safe default; the option reader refuses it, since there it would
/// widen an accept list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownWireValue {
    pub kind: &'static str,
    pub value: i32,
}

impl fmt::Display for UnknownWireValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown rtc.{} {}", self.kind, self.value)
    }
}

impl std::error::Error for UnknownWireValue {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownDisconnectReason(pub String);

impl fmt::Display for UnknownDisconnectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown disconnect reason {:?}: one of ", self.0)?;
        for (i, r) in DisconnectReason::ALL.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(r.as_str())?;
        }
        Ok(())
    }
}

impl std::error::Error for UnknownDisconnectReason {}

/// `rtc.ParticipantKind`; must match the SDK's set, or an accepted participant is invisible. The
/// host adapter folds an unknown kind to `standard`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
#[crate::host(wire_enum)]
pub enum ParticipantKind {
    /// An unmapped kind must not make an accepted participant invisible.
    #[default]
    Standard,
    Ingress,
    Egress,
    Sip,
    Agent,
    Connector,
    Bridge,
}

wire_i32!(ParticipantKind, "ParticipantKind");

/// `rtc.TrackSource`. The host adapter folds an unknown source to `unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
#[crate::host(wire_enum)]
pub enum TrackSource {
    #[default]
    Unknown,
    Camera,
    Microphone,
    Screenshare,
    ScreenshareAudio,
}

wire_i32!(TrackSource, "TrackSource");

/// `AudioTrackFeature` (`livekit.rtc._proto.track_pb2`). `_input.py` gates the pre-connect buffer on
/// [`PreconnectBuffer`](Self::PreconnectBuffer). The host adapter drops a feature this build has
/// no variant for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[crate::host(wire_enum)]
pub enum AudioTrackFeature {
    Stereo,
    NoDtx,
    AutoGainControl,
    EchoCancellation,
    NoiseSuppression,
    EnhancedNoiseCancellation,
    PreconnectBuffer,
}

wire_i32!(AudioTrackFeature, "AudioTrackFeature");

/// Each `ALL` array's names, indexed by wire value, for the host adapter's reads of SDK ints:
/// `(connection states, disconnect reasons, participant kinds, track sources, audio features)`.
#[crate::host(export(pid = false))]
pub fn rtc_wire_names() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        ConnectionState::ALL.iter().map(|v| v.as_str()).collect(),
        DisconnectReason::ALL.iter().map(|v| v.as_str()).collect(),
        ParticipantKind::ALL.iter().map(|v| v.as_str()).collect(),
        TrackSource::ALL.iter().map(|v| v.as_str()).collect(),
        AudioTrackFeature::ALL.iter().map(|v| v.as_str()).collect(),
    )
}

/// Everything `_ParticipantAudioOutput` needs to publish its track, translated once at construction.
#[crate::class]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackPublishOptions {
    /// `track_name` on the wire; absent keeps the default (`roomio_audio`).
    pub name: String,
    /// The protobuf int, which is what the host SDK carries a `rtc.TrackSource` as.
    pub source: TrackSource,
    pub sample_rate: u32,
    pub num_channels: u16,
    /// `rtc.AudioSource(queue_size_ms=)`: the backpressure bound that paces `_forward_audio`
    /// against playout (`room_io` passes 200). Not on the wire; a zero default would mean no bound.
    pub queue_size_ms: u32,
}

/// The host's view: what `_core_rtc.LocalParticipant._core_publish_audio_track` builds its
/// `rtc.AudioSource`, `LocalAudioTrack` and `rtc.TrackPublishOptions` from.
#[crate::host(object)]
impl TrackPublishOptions {
    #[host(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
    #[host(getter)]
    pub fn source(&self) -> TrackSource {
        self.source
    }
    #[host(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    #[host(getter)]
    pub fn num_channels(&self) -> u16 {
        self.num_channels
    }
    #[host(getter)]
    pub fn queue_size_ms(&self) -> u32 {
        self.queue_size_ms
    }
}

/// `_output._ParticipantAudioOutput.__init__`'s value, not the SDK's 1000.
pub const DEFAULT_QUEUE_SIZE_MS: u32 = 200;

impl Default for TrackPublishOptions {
    fn default() -> Self {
        TrackPublishOptions {
            name: "roomio_audio".to_owned(),
            source: TrackSource::Microphone,
            sample_rate: 24_000,
            num_channels: 1,
            queue_size_ms: DEFAULT_QUEUE_SIZE_MS,
        }
    }
}

/// `rtc.AudioStream.from_track(...)`'s parameters. `Default` is hand-written: AGC defaults to on.
#[derive(Debug, Clone)]
#[crate::class]
pub struct AudioStreamOptions {
    /// `None` leaves the track's own rate. `AudioInputOptions.sample_rate` defaults to 24000.
    pub sample_rate: Option<u32>,
    pub num_channels: Option<u16>,
    /// `AudioInputOptions.frame_size_ms`, default 50.
    pub frame_size_ms: Option<u32>,
    /// The host-side `rtc.FrameProcessor` / `rtc.NoiseCancellationOptions`, an opaque host object
    /// Rust never inspects.
    pub noise_cancellation: Option<HostRef>,
    /// `AudioInputOptions.auto_gain_control`, default **true**. Performed host-side, per stream.
    pub auto_gain_control: bool,
}

/// The host's view: what `_core_rtc.TrackPublication._core_open_audio_stream` opens its
/// `rtc.AudioStream` (and, with `auto_gain_control`, its `AudioProcessingModule`) with.
#[crate::host(object)]
impl AudioStreamOptions {
    #[host(getter)]
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }
    #[host(getter)]
    pub fn num_channels(&self) -> Option<u16> {
        self.num_channels
    }
    #[host(getter)]
    pub fn frame_size_ms(&self) -> Option<u32> {
        self.frame_size_ms
    }
    /// The host's own object.
    #[host(getter)]
    pub fn noise_cancellation(&self) -> Option<HostRef> {
        self.noise_cancellation.clone()
    }
    #[host(getter)]
    pub fn auto_gain_control(&self) -> bool {
        self.auto_gain_control
    }
}

impl Default for AudioStreamOptions {
    fn default() -> Self {
        AudioStreamOptions {
            sample_rate: None,
            num_channels: None,
            frame_size_ms: None,
            noise_cancellation: None,
            auto_gain_control: true,
        }
    }
}

/// One frame off an `rtc.VideoStream`: a [`MediaRef`](lk_agents_abi::MediaRef) (the pixels never
/// cross), the size read at the seam to reject non-frames, and the frame's own capture stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::class]
pub struct VideoFrameRef {
    /// The side-table key; the host frame is released when the last clone drops.
    pub media: MediaRef,
    pub width: u32,
    pub height: u32,
    /// `VideoFrameEvent.timestamp_us`, the host's.
    pub timestamp_us: i64,
}

/// What a host `io.VideoInput` answers `_core_next_frame()` with: its frame (interned; only the key
/// crosses back) and the size it read off it.
#[crate::host(object)]
impl VideoFrameRef {
    #[host(new(defaults(timestamp_us = 0)))]
    pub fn new(frame: MediaRef, width: u32, height: u32, timestamp_us: i64) -> Self {
        VideoFrameRef { media: frame, width, height, timestamp_us }
    }
}

/// How many frames a `video_stream` queue holds before it evicts the oldest. Tiny on purpose: each
/// held frame pins a decoded host frame, and the newest frame is the one worth keeping. A host
/// video stream is opened with the same capacity, so both sides drop the oldest alike.
#[crate::host(export)]
pub const VIDEO_QUEUE_FRAMES: usize = 4;

/// `LocalParticipant.stream_text(...)`'s keyword arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::class]
pub struct StreamTextOptions {
    pub topic: String,
    /// Absent means omit the key: the writer then defaults to the local participant's identity.
    pub sender_identity: Option<ParticipantIdentity>,
    pub attributes: HashMap<String, String>,
}

/// The host's view: `_core_rtc.LocalParticipant._core_stream_text` spreads it into
/// `stream_text(..)`, leaving `sender_identity` out when it is `None` (an explicit `None` would
/// publish the user's transcript as the agent).
#[crate::host(object)]
impl StreamTextOptions {
    #[host(getter)]
    pub fn topic(&self) -> String {
        self.topic.clone()
    }
    #[host(getter)]
    pub fn sender_identity(&self) -> Option<ParticipantIdentity> {
        self.sender_identity.clone()
    }
    #[host(getter)]
    pub fn attributes(&self) -> HashMap<String, String> {
        self.attributes.clone()
    }
}

/// `rtc.TranscriptionSegment`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Host)]
#[binding(emit)]
pub struct TranscriptionSegment {
    pub id: String,
    pub text: String,
    /// Milliseconds. `room_io` always writes 0 for both; kept because the field is on the wire.
    pub start_time: u64,
    pub end_time: u64,
    pub language: String,
    #[host(rename = "final")]
    pub is_final: bool,
}

/// `rtc.Transcription`, the argument to `publish_transcription`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::class]
pub struct Transcription {
    pub participant_identity: ParticipantIdentity,
    pub track_sid: TrackSid,
    pub segments: Vec<TranscriptionSegment>,
}

/// The host's view: `_core_rtc.LocalParticipant._core_publish_transcription` builds the
/// `rtc.Transcription` and its `rtc.TranscriptionSegment`s (keyword dicts, `final` spelled plainly).
#[crate::host(object)]
impl Transcription {
    #[host(getter)]
    pub fn participant_identity(&self) -> ParticipantIdentity {
        self.participant_identity.clone()
    }
    #[host(getter)]
    pub fn track_sid(&self) -> TrackSid {
        self.track_sid.clone()
    }
    #[host(getter)]
    pub fn segments(&self) -> Vec<TranscriptionSegment> {
        self.segments.clone()
    }
}

/// What one `wait_for_playout` covered, measured since the previous one returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[crate::class]
pub struct PlaybackInfo {
    /// Audio that reached the speaker.
    pub played: Duration,
    /// Audio captured and then thrown away by `clear_queue`: non-zero means the playout was
    /// interrupted.
    pub discarded: Duration,
}

impl PlaybackInfo {
    pub fn interrupted(&self) -> bool {
        !self.discarded.is_zero()
    }
}

/// What a host audio source's `PlayoutLedger` hands back, and the core reads back unchanged.
#[crate::host(object)]
impl PlaybackInfo {
    #[host(getter)]
    pub fn played(&self) -> Duration {
        self.played
    }
    #[host(getter)]
    pub fn discarded(&self) -> Duration {
        self.discarded
    }
}

/// Header of an incoming byte or text stream: `rtc.ByteStreamInfo` / `rtc.TextStreamInfo`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Host)]
#[binding(extract, emit)]
pub struct StreamInfo {
    #[host(rename = "stream_id")]
    pub id: String,
    pub topic: String,
    pub mime_type: String,
    // A text stream has no `size`; the host adapter reads it as `None` when absent.
    #[host(rename = "size")]
    pub total_length: Option<u64>,
    // `_pre_connect_audio.py` fails on a missing `trackId`, `sampleRate` or `channels`; the host
    // adapter hands `None` over as an empty map.
    pub attributes: HashMap<String, String>,
    // Carried whole so the host can rebuild its `TextStreamInfo` for `TextInputEvent.info`.
    pub timestamp: i64,
    pub attachments: Vec<String>,
}

/// `AudioOutputCapabilities`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(deny_unknown_fields))]
#[crate::class]
pub struct AudioOutputCapabilities {
    pub pause: bool,
}

/// What a host `io.AudioOutput` answers its `_core_capabilities()` with.
#[crate::host(object)]
impl AudioOutputCapabilities {
    #[host(new)]
    pub fn new(pause: bool) -> Self {
        AudioOutputCapabilities { pause }
    }
    #[host(getter)]
    pub fn pause(&self) -> bool {
        self.pause
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `frame_duration` losing precision through `AudioFrame::duration`'s `f64`.
    #[test]
    fn frame_duration_is_exact_for_rates_that_do_not_divide_evenly() {
        let awkward = AudioFrame::new(vec![0u8; 2], 3_000, 1, 1);
        assert_eq!(frame_duration(&awkward), Duration::from_nanos(333_333));

        let normal = silence_frame(480, 24_000, 1);
        assert_eq!(frame_duration(&normal), Duration::from_millis(20));
        assert_eq!(frame_duration(&normal) * 50, Duration::from_secs(1), "50 frames make a second");
        assert_eq!(frame_duration(&AudioFrame::new(vec![0u8; 64], 0, 1, 32)), Duration::ZERO);
    }


    // Catches: `silence_frame` allocating the wrong number of bytes, or non-zero ones.
    #[test]
    fn silence_allocates_exactly_the_samples_it_claims() {
        let mono = silence_frame(480, 24_000, 1);
        assert_eq!(mono.data.len(), 480 * 2);
        assert!(mono.data.iter().all(|&b| b == 0));

        let stereo = silence_frame(480, 24_000, 2);
        assert_eq!(stereo.data.len(), 480 * 2 * 2);
        assert_eq!(stereo.samples_per_channel, 480, "per channel, not in total");
    }

    // Catches: a wire value off by one against the SDK's protobuf, or an unknown one accepted.
    #[test]
    fn the_wire_values_match_the_sdk_and_unknown_ones_are_refused() {
        assert_eq!(DisconnectReason::AgentError.wire(), 16, "the SDK's last value");
        assert_eq!(DisconnectReason::try_from(16), Ok(DisconnectReason::AgentError));
        assert_eq!(TrackSource::Microphone.wire(), 2, "what `TrackPublishOptions(source=)` takes");
        assert_eq!(
            ParticipantKind::try_from(7),
            Err(UnknownWireValue { kind: "ParticipantKind", value: 7 })
        );
        assert!(DisconnectReason::try_from(-1).is_err());
        // `Connected` is 1 (every publish guards on it); `PreconnectBuffer` is 6.
        assert_eq!(ConnectionState::Connected.wire(), 1);
        assert_eq!(ConnectionState::try_from(1), Ok(ConnectionState::Connected));
        assert!(ConnectionState::try_from(3).is_err(), "an unknown state is the caller's to fold");
        assert_eq!(AudioTrackFeature::PreconnectBuffer.wire(), 6, "what `_input.py` looks for");
        assert!(AudioTrackFeature::try_from(7).is_err());
    }

    // Catches: a name table that drifts from the enum.
    #[test]
    fn every_reason_has_a_name_that_parses_back() {
        assert_eq!("client_initiated".parse(), Ok(DisconnectReason::ClientInitiated));
        assert!("CLIENT_INITIATED".parse::<DisconnectReason>().is_err());
    }

    // Catches: the `close_on_disconnect` set drifting from `DEFAULT_CLOSE_ON_DISCONNECT_REASONS`.
    #[test]
    fn only_three_reasons_close_the_session() {
        let closing: Vec<_> = DisconnectReason::ALL
            .into_iter()
            .filter(|r| r.closes_session_by_default())
            .collect();
        assert_eq!(
            closing,
            vec![
                DisconnectReason::ClientInitiated,
                DisconnectReason::RoomDeleted,
                DisconnectReason::UserRejected
            ]
        );
    }
}
