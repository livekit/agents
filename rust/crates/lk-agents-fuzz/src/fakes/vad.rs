use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{
    AudioFrame, BoxFuture, Ts, Vad, VadCapabilities, VadEvent, VadEventType, VadSink, VadSource,
    errors::ProviderError,
    utils::stream::{Closed, Entry, StreamInput, StreamOutput},
};
use lk_agents_runtime::inference::vad::plugin::PluginInput;
use lk_agents_runtime::net::clock::SimClock;
use tokio::sync::{Notify, mpsc};

use super::{
    Control, Forward, Registry, Slot,
    hazard::{Site, strike},
};
use crate::{
    input::{Confidence, Input, Millis},
    record::{Detail, Kind},
};

/// How often a real detector reports, in seconds.
const UPDATE_INTERVAL: f64 = 0.032;

pub struct FakeVad {
    registry: Arc<Registry>,
    /// Set for a plugin detector: how long each stream takes no audio after it opens.
    stall: Option<Duration>,
}

impl FakeVad {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeVad { registry, stall: None })
    }

    pub fn plugin(registry: Arc<Registry>, stall: Duration) -> Arc<Self> {
        Arc::new(FakeVad { registry, stall: Some(stall) })
    }
}

/// A plugin's input that takes nothing before `until`, then each entry at once.
struct Stalled {
    forward: Arc<Forward>,
    until: tokio::time::Instant,
}

impl StreamInput<AudioFrame> for Stalled {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.forward.push(frame)
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.forward.flush()
    }
    fn end_input(&self) {
        self.forward.end_input();
    }
    fn send(&self, entry: Entry<AudioFrame>) -> BoxFuture<'static, Result<(), ProviderError>> {
        let (forward, until) = (self.forward.clone(), self.until);
        Box::pin(async move {
            tokio::time::sleep_until(until).await;
            forward.send(entry).await
        })
    }
}

struct Detection {
    slot: Slot,
    registry: Arc<Registry>,
    events: mpsc::UnboundedReceiver<Result<VadEvent, ProviderError>>,
    ended: Arc<Notify>,
    hazard: super::hazard::StreamHazard,
}

impl Vad for FakeVad {
    fn label(&self) -> String {
        "fake::vad".into()
    }

    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities { update_interval: UPDATE_INTERVAL }
    }

    fn stream(
        &self,
        cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<VadSink, VadSource>, ProviderError>,
    > {
        let (sender, events) = mpsc::unbounded_channel();
        let ended = Arc::new(Notify::new());
        let detail = if self.stall.is_some() { Detail::PluginVad } else { Detail::None };
        let slot = self.registry.register(Kind::Vad, detail, Control::Vad(sender));
        let registry = self.registry.clone();
        let forward = Forward { slot, registry: registry.clone(), ended: Some(ended.clone()) };
        let hazard = registry.stream_hazard(Site::VadNext, Site::VadClose);
        let detection: VadSource = Box::new(Detection { slot, registry, events, ended, hazard });
        let opened = match self.stall {
            None => lk_agents_abi::DuplexStream::new(Box::new(forward) as VadSink, detection),
            Some(stall) => {
                let until = tokio::time::Instant::now() + stall;
                let input = Stalled { forward: Arc::new(forward), until };
                let stream =
                    lk_agents_abi::DuplexStream::new(Box::new(input) as VadSink, detection);
                PluginInput::wrap(stream, cx, Arc::new(SimClock::new()))
            }
        };
        Box::pin(async move { Ok(opened) })
    }
}

impl StreamOutput<VadEvent> for Detection {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::VadNext, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return strike(fault, Site::VadNext).await;
            }
            // A real stream ends once its input has; otherwise a replaced stream stays open.
            tokio::select! {
                biased;
                event = self.events.recv() => event.transpose(),
                () = self.ended.notified() => Ok(None),
            }
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::VadClose, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return strike(fault, Site::VadClose).await;
            }
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<VadEvent> for Detection {}

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
        // The case chooses the confidence: the driver places speech start retroactively from it.
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
