//! Audio end-of-turn detection, `inference/eot/*`: caller-minted request ids, so stale answers are
//! recognised.
//!
//! `session` holds every decision; `transport`, `cloud` (gateway protocol over the shared
//! WebSocket transport) and `local` (the mini model over a [`LocalEotModel`]) carry them out, mounted by `shell`. Connection headers come from the host
//! through [`CloudOptions::headers`].

mod cloud;
mod detector;
mod local;
pub mod options;
mod session;
/// The channel sink and the mounted-transport shell a binding's `run()`/`recv_event()` drive.
mod shell;
/// The stream driver a session and the host's `TurnDetector.stream()` use.
mod stream;
mod transport;
#[path = "eot.rs"]
mod types;

// `ClientMessage`/`ServerMessage` from the shared protocol crate, aliased as `proto`.
use livekit_protocol::agent as proto;

#[cfg(test)]
#[path = "../eot_tests.rs"]
mod tests;

pub use cloud::{CloudOptions, CloudTransport, HEADER_SESSION_ID, HeaderProvider};
pub use detector::AudioEot;
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
pub use types::{
    CancelReason, DEFAULT_PREDICTION_TIMEOUT, DEFAULT_SAMPLE_RATE, EotRequestId, EotWarning,
};
