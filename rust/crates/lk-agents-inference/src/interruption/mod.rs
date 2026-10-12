//! Adaptive interruption (`inference/interruption.py`): while the agent speaks audio fills a 3 s
//! window; once the user talks over it, the window goes to the gateway every 100 ms until a
//! verdict or either side stops. [`InterruptionCore`] decides; [`transport`] drives.

mod audio;
mod cache;
mod core;
mod event;
// The session loop's `InterruptionPort`, over the detector below.
mod leg;
mod options;
pub mod proto;
#[cfg(feature = "python")]
mod python;
mod retry;
pub mod transport;

pub use audio::{AudioError, AudioFrame, AudioWindow};
pub use leg::InterruptionLeg;
pub use core::{
    InputEvent, InputOutcome, InterruptionCore, OverlapEnd, OverlapStart, RecvOutcome,
    SpanAttributes, StreamError,
};
pub use event::{
    CacheEntry, InterruptionDetectionError, OverlappingSpeechEvent,
    estimate_probability, interruption_metrics, overlapping_speech_from_cache_entry,
};
pub use options::{
    AUDIO_PREFIX_DURATION, DETECTION_INTERVAL, FRAMES_PER_SECOND, InterruptionOptions,
    MAX_AUDIO_DURATION, MIN_INTERRUPTION_DURATION, OptionsError, REMOTE_INFERENCE_TIMEOUT,
    SAMPLE_RATE, min_frames_for,
};
pub use retry::{ConnectFailure, connect_error};
pub use transport::{
    AdaptiveInterruptionDetector, ConnectionConfig, DetectorEvent, InterruptionStream, StreamClosed,
    StreamEvent,
};

/// `type(self).__module__ + "." + type(self).__name__` (`inference/interruption.py:342`), stamped
/// on every [`InterruptionDetectionError`] and read by `agent_session`'s error routing.
pub const LABEL: &str = "livekit.agents.inference.interruption.AdaptiveInterruptionDetector";

/// `AdaptiveInterruptionDetector.model`.
pub const MODEL: &str = "adaptive interruption";

/// `AdaptiveInterruptionDetector.provider`.
pub const PROVIDER: &str = "livekit";
