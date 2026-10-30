//! Failures at the host RTC boundary.

use super::TrackSid;

/// A host sees a host failure as its own exception again, anything else as a `RuntimeError`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum RtcError {
    /// An adapter failure, preserving the original host exception when available.
    #[error("rtc host error: {0}")]
    #[host(delegate)]
    Host(#[from] crate::errors::HostError),

    /// The shared stream lifecycle failed while delivering RTC observations.
    #[error(transparent)]
    #[host(delegate)]
    Stream(#[from] crate::errors::ProviderError),

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

    /// Incoming data exceeded an admission bound before its payload was retained.
    #[error("RTC stream {resource} exceeds limit {limit}")]
    StreamLimit { resource: &'static str, limit: usize },

    /// An ingress writer allows one pending push, so backpressure cannot accumulate waiters.
    #[error("an RTC stream push is already pending")]
    StreamBusy,

    /// A method that does not exist on this side of the object.
    ///
    /// `TrackPublication` is one trait over `rtc`'s `LocalTrackPublication` and
    /// `RemoteTrackPublication` (as in the SDK, where both derive from `TrackPublication`), so
    /// `set_subscribed` has no meaning on a local one and `wait_for_subscription` none on a
    /// remote. Making that a runtime error rather than two traits keeps the shim at one class per
    /// SDK object; the alternative was eight traits for two call sites.
    #[error("{0} is not supported on this publication")]
    Unsupported(&'static str),

    /// A wait on the room (`wait_for_participant` and the like) ended before what it waited for:
    /// the room disconnected, or the participant it waited on left or was never there.
    #[error("{0}")]
    WaitEnded(String),
}


impl RtcError {
    pub fn host(message: impl Into<String>) -> Self {
        RtcError::Host(crate::errors::HostError::new(message))
    }
}

/// A host failure keeps the host's exception; any other room failure is its message.
impl From<RtcError> for crate::voice::io::IoError {
    fn from(e: RtcError) -> Self {
        match e {
            RtcError::Host(host) => crate::voice::io::IoError::Host(host),
            other => crate::voice::io::IoError::Rtc(other.to_string()),
        }
    }
}
