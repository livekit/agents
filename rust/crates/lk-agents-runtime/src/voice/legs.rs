//! The session's legs: the opening STT and VAD streams, their drains, the turn detector's
//! predictions and the tasks that forward the room's and the interruption detector's events.

use super::Runtime;
use crate::voice::session::{OpeningLegs, QueueOwner, SessionCores, SessionExecutor, Timebase};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::{PushAudio, RealtimeCommand};
use lk_agents_abi::resources::{ResourceId, Resources};
use lk_agents_abi::turn::detector::LegEvent;
use lk_agents_abi::{
    AudioFrame, BoxFuture, Realtime, RealtimeSession, Spawner, SpawnerExt, StreamSink, Stt, Vad,
};
use lk_agents_voice::agent_activity::false_interruption::InterruptionPort;
use lk_agents_voice::agent_session::{DriverError, Event, Input};
use lk_agents_voice::audio::AudioChannel;
use std::sync::Arc;
use std::time::Duration;

/// The production [`SessionExecutor`]: it opens the opening streams, then runs the cores on a
/// [`Runtime`].
pub struct Executor {
    time: Timebase,
    spawner: Arc<dyn Spawner>,
    /// The opening streams, read once the cores exist.
    opening: Vec<OpeningLeg>,
}

/// Keeps an initial source and its registered sink owned until the runtime adopts them.
pub(super) struct OpeningLeg {
    channel: AudioChannel,
    sink: ResourceId,
    source: Option<PendingLeg>,
    resources: Resources,
    time: Timebase,
    spawner: Arc<dyn Spawner>,
}

impl OpeningLeg {
    fn new(
        channel: AudioChannel,
        sink: ResourceId,
        source: PendingLeg,
        resources: Resources,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        Self { channel, sink, source: Some(source), resources, time, spawner }
    }

    fn adopt(mut self) -> (AudioChannel, ResourceId, PendingLeg) {
        let source = self.source.take().expect("an opening stream is adopted once");
        (self.channel, self.sink, source)
    }
}

impl Drop for OpeningLeg {
    fn drop(&mut self) {
        let Some(source) = self.source.take() else { return };
        let resources = self.resources.clone();
        let sink = self.sink;
        let deadline = self.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        if let Some(input) = resources.audio_sink(sink) {
            input.end_input();
        }
        let channel = self.channel;
        // Detached: the leg was never adopted, so no session owns it; the close ends on its own.
        self.spawner.spawn_detached(async move {
            let outcome = match source {
                PendingLeg::Stt(source) => source.close(deadline).await,
                PendingLeg::Vad(source) => source.close(deadline).await,
            };
            if outcome != lk_agents_abi::Closed::Clean {
                log::error!(target: "livekit.agents", "opening {} stream cleanup did not settle cleanly: {outcome:?}", channel.label());
            }
            resources.release(sink);
        });
    }
}

impl Executor {
    /// Preparation and abandoned-stream cleanup use the supplied session clock and scheduler.
    pub fn new(time: Timebase, spawner: Arc<dyn Spawner>) -> Self {
        Self { time, spawner, opening: Vec::new() }
    }

    /// Hand the same queue owner to either the native loop or a deterministic caller.
    pub fn into_runtime(self, cores: SessionCores) -> Runtime {
        Runtime::new(cores, self.opening)
    }
}

impl SessionExecutor for Executor {
    fn stt_adapter(&self) -> crate::voice::session::SttAdapter {
        Arc::new(lk_agents_inference::stt::stream_adapter::StreamAdapter::adapted)
    }

    fn tts_adapter(&self) -> crate::voice::session::TtsAdapter {
        Arc::new(|tts, clock, budget| {
            lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted_for(
                tts,
                clock,
                budget,
            )
        })
    }

    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
    ) -> BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        Box::pin(async move {
            let (time, spawner) = (self.time.clone(), self.spawner.clone());
            let mut legs = OpeningLegs::default();
            let mut opening = Vec::new();
            if let Some(stt) = stt {
                let (sink, source) =
                    stt.stream(None, None).await?.ok_or(DriverError::Missing("a streaming STT"))?;
                let sink = resources.register_audio_sink(Arc::from(sink));
                opening.push(OpeningLeg::new(
                    AudioChannel::Stt,
                    sink,
                    PendingLeg::Stt(source),
                    resources.clone(),
                    time.clone(),
                    Arc::clone(&spawner),
                ));
                legs.stt = Some(sink);
            }
            if let Some(vad) = vad {
                let (sink, source) = vad.stream().await?;
                let sink = resources.register_audio_sink(Arc::from(sink));
                opening.push(OpeningLeg::new(
                    AudioChannel::Vad,
                    sink,
                    PendingLeg::Vad(source),
                    resources.clone(),
                    time.clone(),
                    Arc::clone(&spawner),
                ));
                legs.vad = Some(sink);
            }
            self.opening = opening;
            Ok(legs)
        })
    }

    fn open_realtime<'a>(
        &'a mut self,
        resources: &'a Resources,
        model: Arc<dyn Realtime>,
        disabled: bool,
    ) -> BoxFuture<'a, Result<(Arc<dyn RealtimeSession>, ResourceId), DriverError>> {
        Box::pin(async move {
            let session = model.session(disabled).await?;
            let sink = resources
                .register_audio_sink(Arc::new(RealtimeAudioSink { session: session.clone() }));
            Ok((session, sink))
        })
    }

    fn own(self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner> {
        Box::new((*self).into_runtime(cores))
    }
}

/// `flush` is a no-op, not `CommitAudio`: an input boundary means the gate substituted silence, not
/// that the user ended a turn.
struct RealtimeAudioSink {
    session: Arc<dyn RealtimeSession>,
}

impl StreamSink<AudioFrame> for RealtimeAudioSink {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.session.send(RealtimeCommand::PushAudio(PushAudio { frame }));
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        // Not `aclose`: a handoff replaces the leg, not the socket (the server's mirror survives).
    }
}

/// An opened stream's source half, held while the core decides: its sink is already registered,
/// so the core names it by id. Read once activated, closed once discarded.
pub(super) enum PendingLeg {
    Stt(lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSource),
}

impl Runtime {
    /// Preserve the provider's event order, including failures and its final end notification.
    pub(super) fn pump_realtime(&self, session: Arc<dyn lk_agents_abi::RealtimeSession>) {
        let tx = self.reports.clone();
        self.spawn("realtime events", async move {
            use lk_agents_voice::agent_activity::realtime::RealtimeWake;
            while let Some(event) = session.next_event().await {
                if tx.send(RealtimeWake::Event(event)).is_err() {
                    return;
                }
            }
            let _ = tx.send(RealtimeWake::Ended);
        });
    }

    /// The legs the session opens with: the opening streams' drains and the interruption
    /// detector's verdicts join the queue until `Closing`.
    pub(super) fn start_legs(&mut self, opening: Vec<OpeningLeg>) {
        for opening in opening {
            let (channel, sink, stream) = opening.adopt();
            self.drain(channel, sink, stream);
        }
        if let Some(detector) = self.interruption.clone() {
            self.pump_verdicts(detector);
        }
    }

    /// Drains an activated stream into the queue.
    pub(super) fn drain(&mut self, channel: AudioChannel, sink: ResourceId, stream: PendingLeg) {
        let shared = &self.shared;
        let (time, closing) =
            (shared.time.clone(), super::audio::closing(shared.closing.subscribe()));
        match stream {
            PendingLeg::Stt(mut source) => {
                source.report_retries(super::audio::report_to(self.reports.clone()));
                super::audio::spawn_drain(
                    (&self.tasks, shared.spawner.as_ref()),
                    time,
                    closing,
                    super::audio::tagged(sink, source),
                    self.reports.clone(),
                    (channel, sink),
                );
            }
            PendingLeg::Vad(source) => super::audio::spawn_drain(
                (&self.tasks, shared.spawner.as_ref()),
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
        let task = self.spawn("turn prediction", async move {
            let outcome = detector.predict(timeout).await;
            if !*closing.borrow() {
                let _ = queue.send(Input::Eot(req, outcome, None));
            }
        });
        self.predictions.retain(|task| self.tasks.contains(*task));
        self.predictions.push(task);
    }

    /// Ends on the detector's `Closed` (forwarded first), when the loop is gone, or at `Closing`.
    /// None is an error: the session goes on with VAD-only interruption.
    fn pump_verdicts(&self, detector: Arc<dyn InterruptionPort>) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.subscribe();
        self.spawn("interruption verdicts", async move {
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
        self.spawn("event forward", async move {
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
