//! The session's legs: the opening STT and VAD streams, their drains, the turn detector's
//! predictions and the tasks that forward the room's and the interruption detector's events.

use super::VoiceRuntime;
use crate::telemetry;
use crate::voice::VoiceRuntimeEvent;
use crate::voice::resources::Resources;
use crate::voice::session::{OpeningLegs, QueueOwner, SessionCores, SessionExecutor, Timebase};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::{PushAudio, RealtimeCommand};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::turn::detector::LegEvent;
use lk_agents_abi::{
    AudioFrame, BoxFuture, Realtime, RealtimeSession, Spawner, SpawnerExt, StreamInput, Stt, Vad,
};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::agent_activity::false_interruption::InterruptionPort;
use lk_agents_voice::agent_session::{DriverError, VoiceEvent};
use lk_agents_voice::audio::AudioChannel;
use std::sync::Arc;
use std::time::Duration;

/// The production [`SessionExecutor`]: it opens the opening streams, then runs the cores on a
/// [`VoiceRuntime`].
pub struct Executor {
    time: Timebase,
    spawner: Arc<dyn Spawner>,
    /// The opening streams, read once the cores exist.
    opening: Vec<OpeningLeg>,
    realtime: Option<RealtimeConnection>,
}

/// Owns a source and its registered sink until adoption or retirement.
/// AMD keeps this owner armed throughout its private STT drain.
pub struct OpeningLeg {
    channel: AudioChannel,
    sink: ResourceId,
    input: Arc<dyn StreamInput<AudioFrame>>,
    source: Option<PendingLeg>,
    resources: Resources,
    time: Timebase,
    spawner: Arc<dyn Spawner>,
}

/// An admitted close must finish even if its awaiting task is cancelled mid-poll.
/// Transfer the same pinned operation, including its original deadline, to the scheduler.
struct PendingClose<T: Send + 'static> {
    future: Option<BoxFuture<'static, T>>,
    spawner: Arc<dyn Spawner>,
}

impl<T: Send + 'static> PendingClose<T> {
    fn new(spawner: Arc<dyn Spawner>, future: BoxFuture<'static, T>) -> Self {
        Self { future: Some(future), spawner }
    }
}

impl<T: Send + 'static> std::future::Future for PendingClose<T> {
    type Output = T;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        let Some(future) = &mut self.future else { return std::task::Poll::Pending };
        match future.as_mut().poll(cx) {
            std::task::Poll::Ready(result) => {
                drop(self.future.take());
                std::task::Poll::Ready(result)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl<T: Send + 'static> Drop for PendingClose<T> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            self.spawner.spawn_detached(async move {
                let _ = future.await;
            });
        }
    }
}

/// Owns the socket from successful opening through explicit close or abandoned startup.
pub(super) struct RealtimeConnection {
    session: Option<Arc<dyn RealtimeSession>>,
    time: Timebase,
    spawner: Arc<dyn Spawner>,
}

impl RealtimeConnection {
    pub(super) fn new(
        session: Arc<dyn RealtimeSession>,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        Self { session: Some(session), time, spawner }
    }

    fn take_close(&mut self) -> Option<PendingClose<bool>> {
        let session = self.session.take()?;
        let time = self.time.clone();
        Some(PendingClose::new(
            self.spawner.clone(),
            Box::pin(async move {
                let closed = time
                    .timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, session.aclose())
                    .await
                    .is_ok();
                if !closed {
                    log::error!(target: "livekit.agents", "realtime session cleanup timed out");
                }
                closed
            }),
        ))
    }

    pub(super) fn close(&mut self, tasks: &Arc<TaskSet>, tx: crate::voice::session::Tx) {
        let Some(session) = self.session.take() else { return };
        let mut owner = Self::new(session, self.time.clone(), self.spawner.clone());
        let _ = tasks.try_spawn(self.spawner.as_ref(), "realtime close", async move {
            if let Some(close) = owner.take_close()
                && close.await
            {
                let _ = tx.send(lk_agents_voice::agent_activity::realtime::RealtimeWake::Closed);
            }
        });
    }
}

impl Drop for RealtimeConnection {
    fn drop(&mut self) {
        drop(self.take_close());
    }
}

impl OpeningLeg {
    pub(super) fn new(
        channel: AudioChannel,
        input: Arc<dyn StreamInput<AudioFrame>>,
        source: PendingLeg,
        resources: Resources,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let sink = resources.register_audio_sink(input.clone());
        Self { channel, sink, input, source: Some(source), resources, time, spawner }
    }

    pub(super) fn adopt(mut self) -> (AudioChannel, ResourceId, PendingLeg) {
        let source = self.source.take().expect("an opening stream is adopted once");
        (self.channel, self.sink, source)
    }

    pub(super) fn resource(&self) -> ResourceId {
        self.sink
    }

    pub(super) fn stt_source(&mut self) -> Option<&mut lk_agents_abi::SttSource> {
        match self.source.as_mut() {
            Some(PendingLeg::Stt(source)) => Some(source),
            _ => None,
        }
    }

    pub(super) async fn close(mut self) -> Option<lk_agents_abi::Closed> {
        match self.take_close() {
            Some(close) => Some(close.await),
            None => None,
        }
    }

    /// Explicit retirement is joined by the session's cleanup, unlike abandoned startup.
    pub(super) fn retire(self, tasks: &Arc<TaskSet>) {
        let spawner = self.spawner.clone();
        // The future retains the armed owner until it is polled. Refusal or cancellation
        // before that poll drops the owner and uses its bounded fallback cleanup.
        let _ = tasks.try_spawn(spawner.as_ref(), "unadopted leg close", async move {
            let mut opened = self;
            if let Some(close) = opened.take_close() {
                close.await;
            }
        });
    }

    fn take_close(&mut self) -> Option<PendingClose<lk_agents_abi::Closed>> {
        let source = self.source.take()?;
        let resources = self.resources.clone();
        let sink = self.sink;
        let deadline = self.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        // The owner retains the input independently of registry and async-frame drop order.
        let input = self.input.clone();
        input.end_input();
        let channel = self.channel;
        Some(PendingClose::new(
            self.spawner.clone(),
            Box::pin(async move {
                let _input = input;
                let outcome = match source {
                    PendingLeg::Stt(source) => source.close(deadline).await,
                    PendingLeg::Vad(source) => source.close(deadline).await,
                };
                if outcome != lk_agents_abi::Closed::Clean {
                    log::error!(target: "livekit.agents", "opening {} stream cleanup did not settle cleanly: {outcome:?}", channel.label());
                }
                resources.release(sink);
                outcome
            }),
        ))
    }
}

impl Drop for OpeningLeg {
    fn drop(&mut self) {
        // Startup or the queue owner disappeared: bounded cleanup survives its caller.
        drop(self.take_close());
    }
}

impl Executor {
    /// Preparation and abandoned-stream cleanup use the supplied session clock and scheduler.
    pub fn new(time: Timebase, spawner: Arc<dyn Spawner>) -> Self {
        Self { time, spawner, opening: Vec::new(), realtime: None }
    }

    /// Hand the same queue owner to either the native loop or a deterministic caller.
    pub fn into_runtime(self, cores: SessionCores) -> VoiceRuntime {
        VoiceRuntime::new(cores, self.opening, self.realtime)
    }
}

impl SessionExecutor for Executor {
    fn stt_adapter(&self) -> crate::voice::session::SttAdapter {
        Arc::new(lk_agents_inference::stt::stream_adapter::StreamAdapter::adapted)
    }

    fn tts_adapter(&self) -> crate::voice::session::TtsAdapter {
        Arc::new(|tts, clock, budget| {
            lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted_for(
                tts, clock, budget,
            )
        })
    }

    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
        cx: telemetry::Cx,
    ) -> BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        Box::pin(async move {
            let (time, spawner) = (self.time.clone(), self.spawner.clone());
            let mut legs = OpeningLegs::default();
            let mut opening = Vec::new();
            if let Some(stt) = stt {
                let (sink, source) = stt
                    .stream(&cx, None, None)
                    .await?
                    .ok_or(DriverError::Missing("a streaming STT"))?
                    .split();
                let leg = OpeningLeg::new(
                    AudioChannel::Stt,
                    Arc::from(sink),
                    PendingLeg::Stt(source),
                    resources.clone(),
                    time.clone(),
                    Arc::clone(&spawner),
                );
                legs.stt = Some(leg.sink);
                opening.push(leg);
            }
            if let Some(vad) = vad {
                let (sink, source) = vad.stream(&cx).await?.split();
                let leg = OpeningLeg::new(
                    AudioChannel::Vad,
                    Arc::from(sink),
                    PendingLeg::Vad(source),
                    resources.clone(),
                    time.clone(),
                    Arc::clone(&spawner),
                );
                legs.vad = Some(leg.sink);
                opening.push(leg);
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
            self.realtime = Some(RealtimeConnection::new(
                session.clone(),
                self.time.clone(),
                self.spawner.clone(),
            ));
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

impl StreamInput<AudioFrame> for RealtimeAudioSink {
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

impl VoiceRuntime {
    /// Preserve the provider's event order, including failures and its final end notification.
    pub(super) fn pump_realtime(&self, session: Arc<dyn lk_agents_abi::RealtimeSession>) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.clone();
        self.spawn("realtime events", async move {
            use lk_agents_voice::agent_activity::realtime::RealtimeWake;
            loop {
                let event = tokio::select! {
                    biased;
                    _ = closing.wait_for(|closed| *closed) => return,
                    () = tx.closed() => return,
                    event = session.next_event() => event,
                };
                let Some(event) = event else {
                    let _ = tx.send(RealtimeWake::Ended);
                    return;
                };
                if tx.send(RealtimeWake::Event(event)).is_err() {
                    return;
                }
            }
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
        let (time, closing) = (shared.time.clone(), super::audio::closing(shared.closing.clone()));
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
        let closing = self.shared.closing.clone();
        let task = self.spawn("turn prediction", async move {
            let outcome = detector.predict(&telemetry::Cx::NONE, timeout).await;
            if !*closing.borrow() {
                let _ = queue.send(VoiceEvent::Eot(req, outcome, None));
            }
        });
        self.predictions.retain(|task| self.tasks.contains(*task));
        self.predictions.push(task);
    }

    /// Ends on the detector's `Closed` (forwarded first), when the loop is gone, or at `Closing`.
    /// None is an error: the session goes on with VAD-only interruption.
    fn pump_verdicts(&self, detector: Arc<dyn InterruptionPort>) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.clone();
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
    pub(super) fn forward<T: Into<VoiceRuntimeEvent> + Send + 'static>(
        &self,
        mut rx: tokio::sync::mpsc::UnboundedReceiver<T>,
    ) {
        let tx = self.reports.clone();
        let mut closing = self.shared.closing.clone();
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
