use std::{sync::Arc, time::Duration};

use lk_agents_abi::tokenize::language::{ThresholdOptions, ThresholdOverride, TurnDetectorModel};
use lk_agents_abi::{AudioFrame, EotDetector, EotPrediction, errors::ProviderError};
use tokio::sync::oneshot;

use super::{
    Control, Registry, Slot,
    hazard::{Site, strike},
    tags_of,
};
use crate::{
    input::{Confidence, Input},
    record::{Detail, Kind},
};

/// The detector's unlikely threshold for every language. It sits between the sampled confidences
/// 45 and 55 and above the mini model's stock table (en 0.36), so a turn judged by any other
/// table is visible.
pub const THRESHOLD: f64 = 0.5;

/// Timeout of an unanswered prediction when the driver sets none (virtual time).
const UNBOUNDED_WAIT: Duration = Duration::from_secs(30);

pub struct FakeEot {
    registry: Arc<Registry>,
    slot: Slot,
}

impl FakeEot {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        let slot = Slot { kind: Kind::Eot, nth: registry.new_eot() };
        Arc::new(FakeEot { registry, slot })
    }
}

impl EotDetector for FakeEot {
    fn push_audio(&self, frame: AudioFrame) {
        self.registry.record_forwarded(self.slot, tags_of(&frame));
    }

    fn cancel_inference(&self, _timed_out: bool) {}

    fn flush(&self, _reason: Option<lk_agents_abi::turn::detector::FlushReason>) {}

    fn thresholds(&self) -> ThresholdOptions {
        ThresholdOptions::new(
            TurnDetectorModel::V1Mini,
            Some(ThresholdOverride::All(THRESHOLD)),
            None,
        )
    }

    fn predict(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        timeout: Option<Duration>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<EotPrediction, ProviderError>> {
        Box::pin(async move {
            if let Some(fault) = self.registry.hazard(Site::Predict) {
                return strike(fault, Site::Predict).await;
            }
            let (sender, receiver) = oneshot::channel();
            let slot =
                self.registry.register(Kind::Prediction, Detail::None, Control::Prediction(sender));
            let waited = tokio::time::timeout(timeout.unwrap_or(UNBOUNDED_WAIT), receiver).await;
            self.registry.mark_closed(slot);
            match waited {
                Ok(Ok(answer)) => answer,
                _ => Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                    lk_agents_abi::errors::ErrorKind::Server,
                    "end-of-turn prediction did not arrive",
                ))),
            }
        })
    }
}

/// Answer one pending prediction. Consumed either way, like a tool answer.
pub(super) fn deliver(
    sender: oneshot::Sender<Result<EotPrediction, ProviderError>>,
    what: &Input,
) -> bool {
    let answer = match what {
        Input::EotAnswer { probability: Confidence(percent), backchannel, .. } => {
            let mut prediction = EotPrediction::new(f64::from(*percent) / 100.0, 0.0);
            prediction.backchannel_probability =
                backchannel.map(|Confidence(bc)| f64::from(bc) / 100.0);
            Ok(prediction)
        }
        Input::EotFail { .. } => Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
            lk_agents_abi::errors::ErrorKind::Server,
            "end-of-turn predictor failed",
        ))),
        _ => return false,
    };
    sender.send(answer).is_ok()
}
