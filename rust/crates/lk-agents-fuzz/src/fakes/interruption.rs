use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::{
    AudioFrame, LegEvent, ReqId, Ts,
    errors::ProviderError,
    turn::interruption::{
        InterruptionPort, InterruptionSignal, InterruptionSource, OverlapVerdict,
    },
    utils::stream::StreamInput,
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
    /// The overlap a verdict names, so a stale answer is distinguishable from a current one.
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
        // One port per session: the case addresses it by a single `nth`.
        let slot = registry.register(
            Kind::Interruption,
            Detail::None,
            Control::Interruption(port.clone()),
        );
        assert!(port.slot.set(slot).is_ok(), "interruption port registered twice");
        port
    }

    fn current_overlap(&self) -> Option<ReqId> {
        *self.overlap.lock().unwrap()
    }
}

/// `VoiceSession(interruption_detector=)`, opened at `start()`.
pub struct FakeInterruptionSource(pub Arc<Registry>);

impl InterruptionSource for FakeInterruptionSource {
    fn open(
        &self,
        _anchor: lk_agents_abi::WallAnchor,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> Arc<dyn InterruptionPort> {
        FakeInterruption::new(self.0.clone())
    }
}

impl InterruptionPort for FakeInterruption {
    fn audio(&self) -> Arc<dyn StreamInput<AudioFrame>> {
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

    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, LegEvent<OverlapVerdict>> {
        Box::pin(async move { self.pending.lock().await.recv().await.unwrap_or(LegEvent::Closed) })
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
                reported: Some(Arc::new(lk_agents_abi::voice::events::OverlappingSpeechEvent {
                    created_at: 0.0,
                    detected_at: 0.0,
                    is_interruption: *is_interruption,
                    agent_ended: *agent_ended,
                    total_duration: 0.0,
                    prediction_duration: 0.0,
                    detection_delay: 0.0,
                    overlap_started_at: None,
                    speech_input: None,
                    probabilities: None,
                    probability: 0.0,
                    num_requests: 0,
                })),
                // The real leg reports one record per verdict.
                metrics: Some(lk_agents_abi::metrics::base::InterruptionMetrics {
                    timestamp: 0.0,
                    total_duration: 0.0,
                    prediction_duration: 0.0,
                    detection_delay: 0.0,
                    num_interruptions: u32::from(*is_interruption),
                    num_backchannels: u32::from(!*is_interruption && !*agent_ended),
                    num_requests: 1,
                    metadata: None,
                }),
            },
        },
        Input::OverlapFail { .. } => LegEvent::Error {
            req,
            recoverable: true,
            err: ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Server,
                "interruption detector failed",
            )),
        },
        _ => return false,
    };
    port.verdicts.send(event).is_ok()
}
