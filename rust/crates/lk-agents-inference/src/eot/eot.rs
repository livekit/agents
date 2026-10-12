//! The plain values `inference/eot/base.py`'s request/response protocol trades in:
//! `predict()` returns a future, `cancel_inference` supersedes, and a stale response is ignored
//! (`:282-291`).

use std::time::Duration;

use crate::eot::EotWarn;

/// `DEFAULT_SAMPLE_RATE` (`base.py:29`). The rate the models are trained at.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// `MIN_SILENCE_DURATION_MS` (`base.py:32`): the minimum VAD silence before a request is worth
/// sending. Enforced against the caller's VAD by `AudioRecognition`.
pub const MIN_SILENCE_DURATION: Duration = Duration::from_millis(200);

/// `DEFAULT_PREDICTION_TIMEOUT` (`base.py:37`), how long §2.3's `AwaitingPrediction` waits.
pub const DEFAULT_PREDICTION_TIMEOUT: Duration = Duration::from_secs(1);

pub use lk_agents_abi::turn::eot::EotRequestId;

/// Why a request is being cancelled -- `cancel_inference(timed_out=...)` (`base.py:186`). A
/// timeout on the cloud model triggers the local fallback (`base.py:203`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    /// New speech, a turn boundary, or a mode change. Just close the request.
    Superseded,
    /// §2.3's `EotPredictionTimeout` fired.
    TimedOut,
}

/// Something the detector wants said out loud; the warn-once state lives in the pure half.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EotWarning {
    /// `base.py:392`. Once per stream.
    CloudFailedFallingBack { reason: String },
    /// `base.py:378`. Terminal: no transport is left to predict with.
    CloudFailedDegraded { reason: String },
    /// `base.py:419`. Once per stream.
    LocalFailed { reason: String },
    /// `transports.py:422`, one prediction the mini model raised on. Logged every occurrence.
    LocalPredictionFailed { reason: String },
    /// `transports.py:195`, over 500 ms between the client stamping a message and the server
    /// answering it.
    TransportLatency { millis: i64 },
    /// `transports.py:262`, a `ServerMessage` whose oneof this client does not handle.
    UnexpectedServerMessage,
}

impl EotWarning {
    /// The machine-readable tag the host switches on (`action["warning"]`); `base.py` compares
    /// against every one.
    pub fn tag(&self) -> &'static str {
        match self {
            EotWarning::CloudFailedFallingBack { .. } => "cloud_failed_falling_back",
            EotWarning::CloudFailedDegraded { .. } => "cloud_failed_degraded",
            EotWarning::LocalFailed { .. } => "local_failed",
            EotWarning::LocalPredictionFailed { .. } => "local_prediction_failed",
            EotWarning::TransportLatency { .. } => "transport_latency",
            EotWarning::UnexpectedServerMessage => "unexpected_server_message",
        }
    }

    /// The values interpolated into the host's sentence (`action["reason"]`), or `None` when the
    /// tag is the whole message.
    pub fn detail(&self) -> Option<String> {
        match self {
            EotWarning::CloudFailedFallingBack { reason }
            | EotWarning::CloudFailedDegraded { reason }
            | EotWarning::LocalFailed { reason }
            | EotWarning::LocalPredictionFailed { reason } => Some(reason.clone()),
            EotWarning::UnexpectedServerMessage => None,
            EotWarning::TransportLatency { millis } => Some(format!("{millis} ms behind")),
        }
    }

    pub fn wire(&self) -> EotWarn {
        EotWarn { warning: self.tag().to_owned(), reason: self.detail() }
    }
}
