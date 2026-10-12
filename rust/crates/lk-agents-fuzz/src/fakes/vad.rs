//! Voice activity: a second observer of the microphone, alongside recognition. Every `stream()`
//! the product opens is its own invocation; the audio the driver forwards is recorded, and the
//! detections come only from the case, so the relation between the two is the case's to choose.
use std::sync::Arc;

use lk_agents_abi::{
    Ts, Vad, VadCapabilities, VadEvent, VadEventType, VadSink, VadSource,
    errors::ProviderError,
    utils::stream::{Closed, StreamSource},
};
use tokio::sync::mpsc;

use super::{Control, Forward, Registry, Slot};
use crate::{
    input::{Confidence, Input, Millis},
    record::{Detail, Kind},
};

/// How often a real detector reports, in seconds; what one `VadInference` stands for.
const UPDATE_INTERVAL: f64 = 0.032;

pub struct FakeVad {
    registry: Arc<Registry>,
}

impl FakeVad {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeVad { registry })
    }
}

struct Detection {
    slot: Slot,
    registry: Arc<Registry>,
    events: mpsc::UnboundedReceiver<Result<VadEvent, ProviderError>>,
}

#[async_trait::async_trait]
impl Vad for FakeVad {
    fn label(&self) -> String {
        "fake::vad".into()
    }

    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities { update_interval: UPDATE_INTERVAL }
    }

    async fn stream(&self) -> Result<(VadSink, VadSource), ProviderError> {
        let (sender, events) = mpsc::unbounded_channel();
        let slot = self.registry.register(Kind::Vad, Detail::None, Control::Vad(sender));
        Ok((
            Box::new(Forward { slot, registry: self.registry.clone() }),
            Box::new(Detection { slot, registry: self.registry.clone(), events }),
        ))
    }
}

#[async_trait::async_trait]
impl StreamSource<VadEvent> for Detection {
    async fn next(&mut self) -> Result<Option<VadEvent>, ProviderError> {
        self.events.recv().await.transpose()
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    async fn close(self: Box<Self>, _: Ts) -> Closed {
        self.registry.mark_closed(self.slot);
        Closed::Clean
    }
}

fn secs(Millis(ms): Millis) -> f64 {
    ms as f64 / 1000.0
}

/// One detection onto a registered stream. True when the product still reads it.
pub(super) fn deliver(
    sender: &mpsc::UnboundedSender<Result<VadEvent, ProviderError>>,
    what: &Input,
) -> bool {
    use VadEventType as T;
    let ev = match what {
        Input::VadStart { speech_ms, .. } => {
            let mut ev = VadEvent::new(T::StartOfSpeech, 0, 0.0, secs(*speech_ms), 0.0);
            ev.speaking = true;
            ev.probability = 1.0;
            ev.raw_accumulated_speech = secs(*speech_ms);
            ev
        }
        // A real detector reports every interval with a confidence, and the driver places the
        // start of speech retroactively from what it accumulated. Noisy confidence is the input
        // that logic exists for, so the case chooses it.
        Input::VadInference { probability: Confidence(percent), speaking, duration, .. } => {
            let (speech, silence) =
                if *speaking { (secs(*duration), 0.0) } else { (0.0, secs(*duration)) };
            let mut ev = VadEvent::new(T::InferenceDone, 0, 0.0, speech, silence);
            ev.probability = f64::from(*percent) / 100.0;
            ev.speaking = *speaking;
            ev.inference_duration = UPDATE_INTERVAL / 4.0;
            ev.raw_accumulated_speech = speech;
            ev.raw_accumulated_silence = silence;
            ev
        }
        Input::VadEnd { silence_ms, .. } => {
            let mut ev = VadEvent::new(T::EndOfSpeech, 0, 0.0, 0.0, secs(*silence_ms));
            ev.raw_accumulated_silence = secs(*silence_ms);
            ev
        }
        _ => return false,
    };
    sender.send(Ok(ev)).is_ok()
}
