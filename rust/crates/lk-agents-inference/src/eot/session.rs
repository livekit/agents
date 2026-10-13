//! The pure half of `_BaseStreamingTurnDetectorStream` (`inference/eot/base.py:91-434`):
//! correlation, staleness, supersede, the one-way cloud->local fallback, failure defaults and
//! warn-once state. The host executes the [`EotAction`]s. `now` is a wall-clock `f64` that only
//! lands in [`EotPrediction::last_speaking_time`].

use std::sync::{Arc, Mutex};

use lk_agents_abi::tokenize::language::{ThresholdOptions, TurnDetectorModel};
use lk_agents_abi::EotPrediction;

use crate::eot::{CancelReason, EotRequestId, EotWarning};

pub use lk_agents_abi::turn::eot::{
    EotDegrade, EotFlush, EotResolve, EotStartInference, EotSwapToLocal, EotWarn,
};

/// What the shell must do next; several co-occur (a cloud failure warns, resolves and swaps, in
/// that order). Variant names and the records' `tag_value`s are the wire: `base.py`'s `_apply`
/// dispatches on `action["type"]`.
#[derive(Debug, Clone, PartialEq)]
pub enum EotAction {
    /// Hand the prediction to whoever is awaiting its id.
    Resolve(EotResolve),
    /// `transport.run_inference(request_id)` (`base.py:184`).
    StartInference(EotStartInference),
    /// `transport.flush()` (`base.py:205`).
    Flush(EotFlush),
    /// Detach the current transport and mount the local mini model (`base.py:388`).
    SwapToLocal(EotSwapToLocal),
    /// `_degrade` (`base.py:373`): nothing is left to predict with. Drop buffered audio and stop.
    Degrade(EotDegrade),
    /// Something the host should log; its dict is [`EotWarn`].
    Warn(EotWarning),
}

/// A prediction as a transport reports it. The local model reports no detection delay and no
/// backchannel probability; the gateway's protobuf always has one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransportPrediction {
    pub probability: f64,
    pub detection_delay: Option<f64>,
    pub inference_duration: Option<f64>,
    pub backchannel_probability: Option<f64>,
}

impl TransportPrediction {
    pub fn new(probability: f64) -> Self {
        TransportPrediction {
            probability,
            detection_delay: None,
            inference_duration: None,
            backchannel_probability: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EotSession {
    /// Shared with the detector, as in the reference (`base.py:137`), so the cloud->local rescale
    /// is visible to every reader.
    thresholds: Arc<Mutex<ThresholdOptions>>,
    model: TurnDetectorModel,
    local_fallback: bool,
    /// `self._request_id`. The waiter lives in the shell.
    in_flight: Option<EotRequestId>,
    is_fallback: bool,
    degraded: bool,
    /// `self._audio_ch.closed`. `predict()` on a closed stream resolves 1.0 without asking anyone
    /// (`base.py:175`).
    closed: bool,
    warned_cloud_failure: bool,
    warned_local_failure: bool,
}

impl EotSession {
    /// `local_fallback`: the caller allowed it and a local model is available.
    pub fn new(
        model: TurnDetectorModel,
        thresholds: Arc<Mutex<ThresholdOptions>>,
        local_fallback: bool,
    ) -> Self {
        EotSession {
            thresholds,
            model,
            local_fallback,
            in_flight: None,
            is_fallback: false,
            degraded: false,
            closed: false,
            warned_cloud_failure: false,
            warned_local_failure: false,
        }
    }

    pub fn model(&self) -> TurnDetectorModel {
        self.model
    }

    pub fn is_fallback(&self) -> bool {
        self.is_fallback
    }

    /// True once the cloud transport is gone with no local model to replace it (`base.py:151`).
    /// Turn detection is off for the rest of the stream and turns commit on the endpointing delay.
    pub fn is_degraded(&self) -> bool {
        self.degraded
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn in_flight(&self) -> Option<&EotRequestId> {
        self.in_flight.as_ref()
    }

    /// `predict()` (`base.py:173`): supersede whatever is in flight, then start `id`.
    pub fn start(&mut self, now: f64, id: EotRequestId) -> Vec<EotAction> {
        if self.closed {
            return vec![EotAction::Resolve(EotResolve::new(id, default_prediction(now, 1.0)))];
        }
        let mut actions = self.close_in_flight(now, 0.0);
        self.in_flight = Some(id.clone());
        actions.push(EotAction::StartInference(EotStartInference { id }));
        actions
    }

    /// `_resolve_prediction` (`base.py:282`): "Stale response is ignored".
    pub fn resolve(
        &mut self,
        now: f64,
        id: &str,
        prediction: TransportPrediction,
    ) -> Vec<EotAction> {
        if self.in_flight.as_ref().map(EotRequestId::as_str) != Some(id) {
            return Vec::new();
        }
        let id = self.in_flight.take().expect("just compared as Some");
        vec![EotAction::Resolve(EotResolve {
            id,
            end_of_turn_probability: prediction.probability,
            last_speaking_time: now,
            detection_delay: prediction.detection_delay,
            inference_duration: prediction.inference_duration,
            backchannel_probability: prediction.backchannel_probability,
        })]
    }

    /// `cancel_inference` (`base.py:186`). `Some(id)` no-ops when that request is no longer in
    /// flight; `None` cancels the current one, as the reference (the host path has no id). The
    /// timeout fallback runs either way (`base.py:200-209`).
    pub fn cancel(
        &mut self,
        now: f64,
        id: Option<&EotRequestId>,
        reason: CancelReason,
    ) -> Vec<EotAction> {
        if let Some(id) = id
            && self.in_flight.as_ref() != Some(id)
        {
            return Vec::new();
        }
        let mut actions = self.close_in_flight(now, 0.0);
        // "trigger fallback immediately" (`base.py:200`): only the cloud model, only on timeout.
        if reason == CancelReason::TimedOut && self.model == TurnDetectorModel::V1 {
            actions.extend(self.fallback_to_local(now, "eot prediction timed out").1);
        }
        actions
    }

    /// `flush` (`base.py:196`). Idempotent.
    pub fn flush(&mut self, now: f64) -> Vec<EotAction> {
        if self.closed {
            return Vec::new();
        }
        let mut actions = vec![EotAction::Flush(EotFlush {})];
        actions.extend(self.close_in_flight(now, 0.0));
        actions
    }

    /// `end_input` (`base.py:219`): flush, then close the audio stream for good.
    pub fn end_input(&mut self, now: f64) -> Vec<EotAction> {
        let actions = self.flush(now);
        self.closed = true;
        actions
    }

    /// The `except` arm of `_run` (`base.py:344`): the active transport died. The arm follows the
    /// current model, so a local failure after a fallback takes the local path.
    pub fn transport_failed(&mut self, now: f64, reason: &str) -> Vec<EotAction> {
        if self.model == TurnDetectorModel::V1 {
            let (swapped, mut actions) = self.fallback_to_local(now, reason);
            if swapped {
                return actions;
            }
            actions.extend(self.degrade(reason));
            return actions;
        }
        self.on_local_failure(now, reason)
    }

    /// `_fallback_to_local` (`base.py:388`). Returns whether the swap happened; one-way.
    fn fallback_to_local(&mut self, now: f64, reason: &str) -> (bool, Vec<EotAction>) {
        if !self.local_fallback {
            // A timeout leaves the transport mounted to retry; only `degrade` warns (`base.py:391`).
            return (false, self.emit_default_for_inflight(now));
        }

        let mut actions = Vec::new();
        if !self.warned_cloud_failure {
            self.warned_cloud_failure = true;
            actions.push(EotAction::Warn(EotWarning::CloudFailedFallingBack {
                reason: reason.to_owned(),
            }));
        }
        actions.extend(self.emit_default_for_inflight(now));
        self.thresholds.lock().expect("thresholds mutex").to_local_fallback();
        self.model = TurnDetectorModel::V1Mini;
        self.is_fallback = true;
        actions.push(EotAction::SwapToLocal(EotSwapToLocal {}));
        (true, actions)
    }

    /// `_degrade` (`base.py:373`). Terminal, and always warned.
    fn degrade(&mut self, reason: &str) -> Vec<EotAction> {
        self.degraded = true;
        // `close()` alone would retain the backlog (`base.py:384`), so the shell drains too.
        self.closed = true;
        vec![
            EotAction::Warn(EotWarning::CloudFailedDegraded { reason: reason.to_owned() }),
            EotAction::Degrade(EotDegrade {}),
        ]
    }

    /// `_on_local_failure` (`base.py:417`): warn once, default the in-flight request, stay open.
    fn on_local_failure(&mut self, now: f64, reason: &str) -> Vec<EotAction> {
        let mut actions = Vec::new();
        if !self.warned_local_failure {
            self.warned_local_failure = true;
            actions.push(EotAction::Warn(EotWarning::LocalFailed { reason: reason.to_owned() }));
        }
        actions.extend(self.emit_default_for_inflight(now));
        actions
    }

    /// `_emit_default_for_inflight` (`base.py:426`). 1.0, so waiters commit after
    /// `min_endpointing_delay`.
    fn emit_default_for_inflight(&mut self, now: f64) -> Vec<EotAction> {
        self.close_in_flight(now, 1.0)
    }

    /// Clear the in-flight request and resolve its waiter: 0.0 when cancelled (`base.py:194`),
    /// 1.0 when failed.
    fn close_in_flight(&mut self, now: f64, probability: f64) -> Vec<EotAction> {
        match self.in_flight.take() {
            Some(id) => {
                vec![EotAction::Resolve(EotResolve::new(id, default_prediction(now, probability)))]
            }
            None => Vec::new(),
        }
    }
}

/// `_default_event` (`base.py:207`): a probability and a wall stamp, no telemetry.
fn default_prediction(now: f64, probability: f64) -> EotPrediction {
    EotPrediction::new(probability, now)
}
