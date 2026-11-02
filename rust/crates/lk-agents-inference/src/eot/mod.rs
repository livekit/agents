//! Audio end-of-turn detection, `inference/eot/*`: caller-minted request ids, so stale answers are
//! recognised.
//!
//! `session` holds every decision; `transport`, `cloud` (gateway protocol over the shared
//! WebSocket transport) and `local` (the mini model over a [`LocalEotModel`]) carry them out, mounted by `shell`. Connection headers come from the host
//! through [`CloudOptions::headers`].

use livekit_protocol::agent as proto;
use std::time::Duration;

mod cloud;
mod detector;
mod local;
mod session;
/// The channel sink and the mounted-transport shell a binding's `run()`/`recv_event()` drive.
mod shell;
/// The stream driver a session and the host's `TurnDetector.stream()` use.
mod stream;
mod transport;

// `ClientMessage`/`ServerMessage` from the shared protocol crate, aliased as `proto`.

#[cfg(test)]
#[path = "../eot_tests.rs"]
mod tests;

pub use cloud::{CloudOptions, CloudTransport, HEADER_SESSION_ID, HeaderProvider};
pub use detector::{AudioEot, select};
pub use local::{
    AudioRingBuffer, CLIENT_BUFFER_SAMPLES, CLIENT_BUFFER_SECONDS, LocalEotModel, LocalTransport,
};
pub use session::{
    EotAction, EotDegrade, EotFlush, EotResolve, EotSession, EotStartInference, EotSwapToLocal,
    EotWarn, TransportPrediction,
};
pub use shell::{MountedTransport, PROVIDER, PredictionChannel, SinkEvent};
pub use stream::{DetectorStream, EotStream, MetricsCallback, TransportFactory};
pub use transport::{EotTransport, PredictionListener};

/// `DEFAULT_SAMPLE_RATE` (`base.py:29`). The rate the models are trained at.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

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
}
