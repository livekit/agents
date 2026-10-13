//! End-of-turn prediction: the third observer of the person's audio, and the only one the driver
//! *asks* rather than listens to. Each `predict` is a request the case answers, like a tool call;
//! left unanswered it times out, which is a real provider outcome the driver must handle.
use std::{sync::Arc, time::Duration};

use lk_agents_abi::{AudioFrame, ChatContext, EotDetector, EotPrediction, errors::ProviderError};
use tokio::sync::oneshot;

use super::{Control, Registry, Slot, tags_of};
use crate::{
    input::{Confidence, Input},
    record::{Detail, Kind},
};

/// What an unanswered prediction waits before failing when the driver set no timeout of its own.
/// Virtual time, so it costs nothing to reach.
const UNBOUNDED_WAIT: Duration = Duration::from_secs(30);

pub struct FakeEot {
    registry: Arc<Registry>,
    /// The leg itself, so the frames pushed into it are recorded against a nameable resource.
    slot: Slot,
}

impl FakeEot {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        let slot = Slot { kind: Kind::Eot, nth: registry.new_eot() };
        Arc::new(FakeEot { registry, slot })
    }
}

#[async_trait::async_trait]
impl EotDetector for FakeEot {
    fn push_audio(&self, frame: AudioFrame) {
        self.registry.record_forwarded(self.slot, tags_of(&frame));
    }

    fn cancel(&self, _timed_out: bool) {}

    fn flush(&self, _reason: &'static str) {}

    async fn predict(
        &self,
        _chat_ctx: &ChatContext,
        timeout: Option<Duration>,
    ) -> Result<EotPrediction, ProviderError> {
        let (sender, receiver) = oneshot::channel();
        let slot =
            self.registry.register(Kind::Prediction, Detail::None, Control::Prediction(sender));
        let waited = tokio::time::timeout(timeout.unwrap_or(UNBOUNDED_WAIT), receiver).await;
        self.registry.mark_closed(slot);
        match waited {
            Ok(Ok(answer)) => answer,
            // The case never answered, or the driver moved on: a prediction that did not arrive.
            _ => Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                "end-of-turn prediction did not arrive",
            ))),
        }
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
            "end-of-turn predictor failed",
        ))),
        _ => return false,
    };
    sender.send(answer).is_ok()
}
