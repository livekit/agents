//! `_StreamingTurnDetectionTransport` (`base.py:41`) and the sink a transport reports into (the
//! reference's back-reference into the stream).

use std::collections::BTreeMap;

use async_trait::async_trait;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::tokenize::language::ThresholdError;
use lk_agents_abi::EotInferenceMetrics;
use lk_agents_abi::metrics::base::Metadata;

use lk_agents_abi::AudioFrame;

use crate::eot::{EotRequestId, EotWarning};
use crate::eot::session::TransportPrediction;

/// Every method is `&self` and synchronous: they are called from the receive path.
pub trait PredictionSink: Send + Sync {
    /// `stream._resolve_prediction(...)`. The sink decides whether `request_id` is still current.
    fn resolve(&self, request_id: &str, prediction: TransportPrediction);

    /// `SessionCreated`: install the server's calibrated defaults (`transports.py:229`). `Err`
    /// becomes a non-retryable failure.
    fn session_created(
        &self,
        thresholds: &BTreeMap<String, f64>,
        default: f64,
        backchannel_thresholds: Option<&BTreeMap<String, f64>>,
        backchannel_default: f64,
    ) -> Result<(), ThresholdError>;

    fn metrics(&self, metrics: EotInferenceMetrics);

    fn warn(&self, warning: EotWarning);

    /// The active model and provider, for [`EotInferenceMetrics::metadata`]; read at emit time.
    fn metadata(&self) -> Metadata;
}

/// The gateway or the local mini model. [`run`](EotTransport::run) is the whole lifetime (`Ok`:
/// audio ended; `Err`: transport died); the other methods post to a queue.
#[async_trait]
pub trait EotTransport: Send + Sync {
    /// The gateway's `X-LiveKit-Session-ID` for this connection; `None` for the local model.
    fn session_id(&self) -> Option<String>;

    /// `run_inference` (`base.py:52`): ask for a prediction under this id.
    fn run_inference(&self, request_id: &EotRequestId);

    /// One frame of 16-bit little-endian PCM, already at the detector's rate.
    fn push_frame(&self, frame: &AudioFrame);

    fn flush(&self);

    /// No more audio is coming. The cloud transport sends `session_close`; both then let
    /// [`run`](EotTransport::run) finish normally.
    fn end_input(&self);

    /// `detach` (`base.py:57`): stop, without waiting for anything in flight.
    fn detach(&self);

    async fn run(&self) -> Result<(), ApiError>;
}
