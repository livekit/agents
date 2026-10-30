//! End-of-turn actions and transport events. Session decisions live in
//! `lk_agents_inference::eot::EotSession`; Python values use the declared record classes.

use crate::EotPrediction;

/// The caller mints it, so cancelling a superseded request is a no-op rather than a teardown of the
/// live one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EotRequestId(String);

impl EotRequestId {
    /// `_utils.py:122`. The suffix is a per-stream counter, not a uuid. An empty session id takes
    /// the fallback prefix (truthiness, as in the reference).
    pub fn new(session_id: Option<&str>, seq: u64) -> Self {
        match session_id.filter(|session_id| !session_id.is_empty()) {
            Some(session_id) => EotRequestId(format!("{session_id}_eot_{seq}")),
            None => EotRequestId(format!("turn_request_{seq}")),
        }
    }

    pub fn from_string(id: String) -> Self {
        EotRequestId(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// --- the actions ------------------------------------------------------------------------------

/// A resolved request and its prediction measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct EotResolve {
    pub id: EotRequestId,
    pub end_of_turn_probability: f64,
    pub last_speaking_time: f64,
    pub detection_delay: Option<f64>,
    pub inference_duration: Option<f64>,
    pub backchannel_probability: Option<f64>,
}

impl EotResolve {
    pub fn new(id: EotRequestId, prediction: EotPrediction) -> Self {
        EotResolve {
            id,
            end_of_turn_probability: prediction.end_of_turn_probability,
            last_speaking_time: prediction.last_speaking_time,
            detection_delay: prediction.detection_delay,
            inference_duration: prediction.inference_duration,
            backchannel_probability: prediction.backchannel_probability,
        }
    }
}

/// Start inference for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct EotStartInference {
    pub id: EotRequestId,
}

/// Flush the detector's buffered audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EotFlush {}

/// Replace the cloud transport with the local mini model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EotSwapToLocal {}

/// Drop buffered audio and stop when no detector remains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EotDegrade {}

/// A detector warning and its optional cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EotWarn {
    pub warning: String,
    pub reason: Option<String>,
}

// --- the transport events ---------------------------------------------------------------------

/// The measurements a transport reports for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionEvent {
    pub id: String,
    pub probability: f64,
    pub detection_delay: Option<f64>,
    pub inference_duration: Option<f64>,
    pub backchannel_probability: Option<f64>,
}
