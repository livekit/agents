//! Audio end-of-turn detection, `inference/eot/*`: caller-minted request ids, so stale answers are
//! recognised.
//!
//! `session` holds every decision; `transport`, `cloud` (gateway protocol over a
//! [`WebSocketConnector`]), `connector` (the real socket) and `local` (the mini model over a
//! [`LocalEotModel`]) carry them out, mounted by `shell`. Connection headers come from the host
//! through [`CloudOptions::headers`].

mod cloud;
mod connector;
mod eot;
mod local;
pub mod options;
#[cfg(feature = "python")]
mod python;
mod session;
/// The channel sink and the mounted-transport shell a binding's `run()`/`recv_event()` drive.
mod shell;
mod transport;

// `ClientMessage`/`ServerMessage` from the shared protocol crate, aliased as `proto`.
use livekit_protocol::agent as proto;

#[cfg(test)]
mod tests;

pub use cloud::{
    CloudOptions, CloudTransport, HEADER_SESSION_ID, HeaderProvider, WebSocketConnector,
    WebSocketSink, WebSocketStream,
};
pub use connector::TungsteniteEotConnector;
pub use eot::{
    CancelReason, DEFAULT_PREDICTION_TIMEOUT, DEFAULT_SAMPLE_RATE, EotRequestId, EotWarning,
    MIN_SILENCE_DURATION,
};
pub use local::{
    AudioRingBuffer, CLIENT_BUFFER_SAMPLES, CLIENT_BUFFER_SECONDS, LocalEotModel, LocalTransport,
};
pub use session::{
    EotAction, EotDegrade, EotFlush, EotResolve, EotSession, EotStartInference, EotSwapToLocal,
    EotWarn, TransportPrediction,
};
pub use shell::{ChannelSink, PROVIDER, SinkEvent, TransportShell};
pub use transport::{EotTransport, PredictionSink};
