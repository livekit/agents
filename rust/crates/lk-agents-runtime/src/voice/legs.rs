//! The session's legs: the opening STT and VAD streams, their drains, the turn detector's
//! predictions and the tasks that forward the room's and the interruption detector's events.

use super::Runtime;
use lk_agents_abi::resources::{ResourceId, Resources};
use lk_agents_abi::turn::detector::LegEvent;
use lk_agents_abi::{BoxFuture, SpawnerExt, Stt, Vad};
use lk_agents_voice::agent_activity::false_interruption::InterruptionPort;
use lk_agents_voice::agent_session::{
    DriverError, Event, Input, OpeningLegs, QueueOwner, SessionCores, SessionExecutor, Tx,
};
use lk_agents_voice::audio::AudioChannel;
use std::sync::Arc;
use std::time::Duration;

/// The production [`SessionExecutor`]: it opens the opening streams, then runs the cores on a
/// [`Runtime`].
#[derive(Default)]
pub struct Executor {
    /// The opening streams, read once the cores exist.
    opening: Vec<(AudioChannel, ResourceId, PendingLeg)>,
}

impl SessionExecutor for Executor {
    fn stt_adapter(&self) -> lk_agents_voice::agent_session::SttAdapter {
        Arc::new(lk_agents_inference::stt::stream_adapter::StreamAdapter::adapted)
    }

    fn tts_adapter(&self) -> lk_agents_voice::agent_session::TtsAdapter {
        Arc::new(
            lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted_for::<
                crate::process::Process,
            >,
        )
    }

    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
    ) -> BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        Box::pin(async move {
            let mut legs = OpeningLegs::default();
            if let Some(stt) = stt {
                let (sink, source) =
                    stt.stream(None, None).await?.ok_or(DriverError::Missing("a streaming STT"))?;
                let sink = resources.register_audio_sink(Arc::from(sink));
                self.opening.push((AudioChannel::Stt, sink, PendingLeg::Stt(source)));
                legs.stt = Some(sink);
            }
            if let Some(vad) = vad {
                let (sink, source) = vad.stream().await?;
                let sink = resources.register_audio_sink(Arc::from(sink));
                self.opening.push((AudioChannel::Vad, sink, PendingLeg::Vad(source)));
                legs.vad = Some(sink);
            }
            Ok(legs)
        })
    }

    fn own(self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner> {
        Box::new(Runtime::new(cores, self.opening))
    }
}

/// An opened stream's source half, held while the core decides: its sink is already registered,
/// so the core names it by id. Read once activated, closed once discarded.
pub(super) enum PendingLeg {
    Stt(lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSource),
}

impl Runtime {
    /// The legs the session opens with: the opening streams' drains and the interruption
    /// detector's verdicts join the queue until `Closing`.
    pub(super) fn start_legs(&mut self, opening: Vec<(AudioChannel, ResourceId, PendingLeg)>) {
        for (channel, sink, stream) in opening {
            self.drain(channel, sink, stream);
        }
        if let Some(detector) = self.interruption.clone() {
            self.pump_verdicts(detector);
        }
    }

    /// Drains an activated stream into the queue.
    pub(super) fn drain(&mut self, channel: AudioChannel, sink: ResourceId, stream: PendingLeg) {
        let shared = &self.shared;
        let (time, closing) = (shared.time.clone(), shared.closing.subscribe());
        match stream {
            PendingLeg::Stt(mut source) => {
                source.report_retries(lk_agents_voice::audio::report_to(self.reports.clone()));
                lk_agents_voice::audio::spawn_drain(
                    &shared.spawner,
                    time,
                    closing,
                    lk_agents_voice::audio::tagged(sink, source),
                    self.reports.clone(),
                    (channel, sink),
                );
            }
            PendingLeg::Vad(source) => lk_agents_voice::audio::spawn_drain(
                &shared.spawner,
                time,
                closing,
                source,
                self.reports.clone(),
                (channel, sink),
            ),
        }
    }

    /// Asks the turn detector for a prediction off the loop; the answer joins the queue under
    /// `req`, unless the session is closing by then.
    pub(super) fn predict(&mut self, req: lk_agents_abi::ReqId, timeout: Option<Duration>) {
        let Some(detector) = self.eot.clone() else { return };
        let queue = self.reports.clone();
        let closing = self.shared.closing.subscribe();
        let task = self.shared.spawner.spawn_unboxed(async move {
            let outcome = detector.predict(timeout).await;
            if !*closing.borrow() {
                let _ = queue.send(Input::Eot(req, outcome, None));
            }
        });
        self.predictions.retain(|task| !task.is_finished());
        self.predictions.push(task);
    }

    /// Ends on the detector's `Closed` (forwarded first), when the loop is gone, or at `Closing`.
    /// None is an error: the session goes on with VAD-only interruption.
    fn pump_verdicts(&self, detector: Arc<dyn InterruptionPort>) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.subscribe();
        self.shared.spawner.spawn_unboxed(async move {
            loop {
                let ev = tokio::select! {
                    biased;
                    _ = closing.wait_for(|c| *c) => break,
                    ev = detector.next_event() => ev,
                };
                let closed = matches!(ev, LegEvent::Closed);
                if tx.send(ev).is_err() || closed {
                    break;
                }
            }
        });
    }

    /// Forwards a source's events into the queue until `Closing`.
    pub(super) fn forward<T: Into<Event> + Send + 'static>(
        &self,
        mut rx: tokio::sync::mpsc::UnboundedReceiver<T>,
    ) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.subscribe();
        self.shared.spawner.spawn_unboxed(async move {
            loop {
                let ev = tokio::select! {
                    biased;
                    ev = rx.recv() => ev,
                    _ = closing.wait_for(|c| *c) => None,
                };
                if ev.is_none_or(|ev| tx.send(ev).is_err()) {
                    break;
                }
            }
        });
    }
}
