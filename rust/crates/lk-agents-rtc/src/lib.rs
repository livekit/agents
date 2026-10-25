//! RTC capabilities supplied by the host SDK. Shared records live in `lk_agents_abi::voice::rtc`.

mod events;
mod fake;
mod playout;
mod server;
mod traits;

pub use events::{
    AudioFrameSender, ByteChunks, ByteStream, ByteStreamSender, Receiver, RoomEvent, RoomEventBus,
    TextChunks, TextStream, TextStreamSender, VideoFrameSender, assert_room_event_is_covered,
    room_event_coverage, room_event_vocabulary,
};
pub use fake::{
    ConnectCall, FakeAudioSource, FakeLocalParticipant, FakeRemoteParticipant, FakeRoom, FakeRtc,
    FakeTrackPublication, FrameSynth, RecordedTextStream, StreamRequests, subscribe_local,
    tokio_test_block,
};
/// Re-exported so the 50 fps audio legs need no copy at this boundary.
pub use lk_agents_abi::AudioFrame;
pub use lk_agents_abi::voice::rtc::{
    AudioStreamOptions, AudioTrackFeature, ConnectionState, DEFAULT_QUEUE_SIZE_MS,
    DisconnectReason, ParticipantIdentity, ParticipantKind, PlaybackInfo, StreamInfo,
    StreamTextOptions, TrackPublishOptions, TrackSid, TrackSource, Transcription,
    TranscriptionSegment, UnknownDisconnectReason, UnknownWireValue, VIDEO_QUEUE_FRAMES,
    VideoFrameRef, frame_duration, silence_frame,
};
pub use playout::PlayoutLedger;
pub use server::SipParticipantReq;
pub use traits::{
    AudioSource, LocalParticipant, RemoteParticipant, RtcHost, RtcRoom, TextStreamWriter,
    TrackPublication,
};

pub type Result<T> = std::result::Result<T, RtcError>;

/// A host sees a host failure as its own exception again, anything else as a `RuntimeError`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
pub enum RtcError {
    /// An adapter failure, preserving the original host exception when available.
    #[error("rtc host error: {0}")]
    #[host(delegate)]
    Host(#[from] lk_agents_abi::errors::HostError),

    /// The room is not connected.
    ///
    /// Its own variant because it is the one failure `room_io` *predicts* rather than catches:
    /// every publish path is guarded by `if self._room.isconnected()` and silently skips
    /// otherwise. A caller that wants that behaviour matches this; one that does not, propagates.
    #[error("room is not connected")]
    NotConnected,

    /// `register_*_stream_handler` on a topic that already has one.
    ///
    /// The SDK raises a bare `ValueError` here and both callers catch it by type and warn
    /// (`room_io.RoomIO.register_text_input`, `_pre_connect_audio.PreConnectAudioHandler.register`) -- so this is a control-flow outcome, not an
    /// error, and it must be distinguishable from a generic host failure.
    #[error("a stream handler is already registered for topic {topic:?}")]
    TopicInUse { topic: String },

    /// A publication or participant went away between being handed out and being used. §1.5's
    /// "lifetime" motivation for handles: the handle can say so, a method on a session cannot.
    #[error("track {0} is gone")]
    TrackGone(TrackSid),

    /// The source, writer or room was already closed. Idempotent teardown paths ignore it.
    #[error("{0} is closed")]
    Closed(&'static str),

    /// A method that does not exist on this side of the object.
    ///
    /// `TrackPublication` is one trait over `rtc`'s `LocalTrackPublication` and
    /// `RemoteTrackPublication` (as in the SDK, where both derive from `TrackPublication`), so
    /// `set_subscribed` has no meaning on a local one and `wait_for_subscription` none on a
    /// remote. Making that a runtime error rather than two traits keeps the shim at one class per
    /// SDK object; the alternative was eight traits for two call sites.
    #[error("{0} is not supported on this publication")]
    Unsupported(&'static str),
}

/// A host failure keeps the original exception: re-raising it raises what the host raised.
#[cfg(feature = "python")]
impl lk_agents_abi::python::FromHostErr for RtcError {
    fn from_host(py: pyo3::Python<'_>, err: &pyo3::PyErr) -> Self {
        RtcError::Host(lk_agents_abi::python::capture_host(py, err))
    }
}

impl RtcError {
    pub fn host(message: impl Into<String>) -> Self {
        RtcError::Host(lk_agents_abi::errors::HostError::new(message))
    }
}

/// A host failure keeps the host's exception; any other room failure is its message.
impl From<RtcError> for lk_agents_abi::voice::io::IoError {
    fn from(e: RtcError) -> Self {
        match e {
            RtcError::Host(host) => lk_agents_abi::voice::io::IoError::Host(host),
            other => lk_agents_abi::voice::io::IoError::Rtc(other.to_string()),
        }
    }
}
