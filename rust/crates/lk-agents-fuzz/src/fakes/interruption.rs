//! The adaptive interruption detector: the fourth observer of the person, and the one that
//! judges overlaps. The driver tells it when the agent speaks and when an overlap starts; it
//! answers with a verdict the case chose. Its audio, like every other leg's, is recorded.
use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::{
    AudioFrame, LegEvent, ReqId, Ts,
    errors::ProviderError,
    turn::interruption::{InterruptionPort, InterruptionSignal, OverlapVerdict},
    utils::stream::StreamSink,
};
use tokio::sync::mpsc;

use super::{Control, Forward, Registry, Slot};
use crate::{
    input::Input,
    record::{Detail, Kind},
};

pub struct FakeInterruption {
    registry: Arc<Registry>,
    /// Set once registered; the port has to exist before it can be its own control.
    slot: OnceLock<Slot>,
    verdicts: mpsc::UnboundedSender<LegEvent<OverlapVerdict>>,
    pending: tokio::sync::Mutex<mpsc::UnboundedReceiver<LegEvent<OverlapVerdict>>>,
    /// The overlap the driver most recently opened. A verdict names it, so the judge can tell an
    /// answer to the current overlap from a stale one.
    overlap: Mutex<Option<ReqId>>,
}

impl FakeInterruption {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        let (verdicts, receiver) = mpsc::unbounded_channel();
        let port = Arc::new(FakeInterruption {
            registry: registry.clone(),
            slot: OnceLock::new(),
            verdicts,
            pending: tokio::sync::Mutex::new(receiver),
            overlap: Mutex::new(None),
        });
        // Lives for the session, so the case addresses it by one `nth` and every verdict goes
        // down its one channel; the port itself is the control.
        let slot = registry.register(
            Kind::Interruption,
            Detail::None,
            Control::Interruption(port.clone()),
        );
        let _ = port.slot.set(slot);
        port
    }

    fn current_overlap(&self) -> Option<ReqId> {
        *self.overlap.lock().unwrap()
    }
}

#[async_trait::async_trait]
impl InterruptionPort for FakeInterruption {
    fn audio(&self) -> Arc<dyn StreamSink<AudioFrame>> {
        let slot = *self.slot.get().expect("registered at construction");
        Arc::new(Forward { slot, registry: self.registry.clone(), ended: None })
    }

    fn signal(&self, signal: InterruptionSignal) {
        if let InterruptionSignal::OverlapEnded { .. } = signal {
            *self.overlap.lock().unwrap() = None;
        }
    }

    fn signal_overlap(&self, overlap: ReqId, signal: InterruptionSignal) {
        match signal {
            InterruptionSignal::OverlapStarted { .. } => {
                *self.overlap.lock().unwrap() = Some(overlap)
            }
            InterruptionSignal::OverlapEnded { .. } => *self.overlap.lock().unwrap() = None,
            _ => {}
        }
    }

    async fn next_event(&self) -> LegEvent<OverlapVerdict> {
        self.pending.lock().await.recv().await.unwrap_or(LegEvent::Closed)
    }
}

/// One verdict onto the port. True when the driver still reads it.
pub(super) fn deliver(port: &Arc<FakeInterruption>, what: &Input) -> bool {
    let req = port.current_overlap();
    let event = match what {
        Input::OverlapVerdict { is_interruption, agent_ended, .. } => LegEvent::Verdict {
            req,
            verdict: OverlapVerdict {
                is_interruption: *is_interruption,
                agent_ended: *agent_ended,
                ignore_user_transcript_until: Ts::ZERO,
            },
        },
        Input::OverlapFail { .. } => LegEvent::Error {
            req,
            recoverable: true,
            err: ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                "interruption detector failed",
            )),
        },
        _ => return false,
    };
    port.verdicts.send(event).is_ok()
}
