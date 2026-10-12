//! The EOT detector's wire. The decisions stay in `lk_agents_inference::eot::EotSession`; the
//! `tag_value` strings are the wire `inference/eot/base.py` reads.

use lk_agents_host_derive::Host;

use crate::{EotInferenceMetrics, EotPrediction};

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

#[cfg(feature = "python")]
crate::python::into_host_str!(EotRequestId, |v| v.as_str());

// --- the actions ------------------------------------------------------------------------------

/// `{"type": "resolve", "id": .., + the five `TurnDetectionEvent` fields}`.
// Flat: the host splats the tail into `TurnDetectionEvent(...)`, and emit targets cannot flatten.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "resolve")]
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

/// `{"type": "start_inference", "id": ..}` -- `transport.run_inference(request_id)` (`base.py:184`).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "start_inference")]
pub struct EotStartInference {
    pub id: EotRequestId,
}

/// `{"type": "flush"}` -- the kind is the whole message (`transport.flush()`, `base.py:205`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "flush")]
pub struct EotFlush {}

/// `{"type": "swap_to_local"}` -- mount the local mini model (`base.py:388`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "swap_to_local")]
pub struct EotSwapToLocal {}

/// `{"type": "degrade"}` -- `_degrade` (`base.py:373`): drop buffered audio and stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "degrade")]
pub struct EotDegrade {}

/// `{"type": "warn", "warning": .., "reason": ..}` -- one `EotWarning` as the host reads it.
// The sentences stay in Python. `reason` is `None`, never absent: `_warn` reads it unconditionally.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "type", tag_value = "warn")]
pub struct EotWarn {
    pub warning: String,
    pub reason: Option<String>,
}

// --- the transport events ---------------------------------------------------------------------

/// `{"type": "prediction", "id": .., + the four numbers a transport reports}`.
// `probability` here is `end_of_turn_probability` on the resolved action.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "prediction")]
pub struct PredictionEvent {
    pub id: String,
    pub probability: f64,
    pub detection_delay: Option<f64>,
    pub inference_duration: Option<f64>,
    pub backchannel_probability: Option<f64>,
}

/// `{"type": "metrics", + `EOTInferenceMetrics`' six fields, with `metadata` spread flat}`.
// Flat because the host splats it. `total_duration` and `detection_delay` come from different
// clocks.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, tag = "type", tag_value = "metrics")]
pub struct MetricsEvent {
    pub timestamp: f64,
    pub total_duration: f64,
    pub detection_delay: f64,
    pub prediction_duration: f64,
    pub num_requests: u32,
    pub model_name: Option<String>,
    pub model_provider: Option<String>,
}

impl MetricsEvent {
    /// Without metadata both keys are written as `None`, not omitted.
    pub fn of(metrics: &EotInferenceMetrics) -> Self {
        let metadata = metrics.metadata.as_ref();
        MetricsEvent {
            timestamp: metrics.timestamp,
            total_duration: metrics.total_duration,
            detection_delay: metrics.detection_delay,
            prediction_duration: metrics.prediction_duration,
            num_requests: metrics.num_requests,
            model_name: metadata.and_then(|m| m.model_name.clone()),
            model_provider: metadata.and_then(|m| m.model_provider.clone()),
        }
    }
}
