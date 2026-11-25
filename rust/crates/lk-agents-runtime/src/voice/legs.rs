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
    AudioFrame, BoxFuture, Closed, Realtime, RealtimeSession, Spawner, StreamInput, Stt, Vad,
};
use lk_agents_net::owned::{Background, Closer, Owned};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::agent_session::machine::activity::false_interruption::InterruptionPort;
use lk_agents_voice::agent_session::{DriverError, VoiceEvent};
use lk_agents_voice::audio::AudioChannel;
use std::sync::Arc;
use std::time::Duration;

/// The production [`SessionExecutor`]: it opens the opening streams, then runs the cores on a
/// [`VoiceRuntime`].
pub struct Executor {
    time: Timebase,
    /// The session's tasks, created here so an abandoned opening stream's close joins them.
    background: Background,
    /// The opening streams, read once the cores exist.
    opening: Vec<OpeningLeg>,
    realtime: Option<RealtimeConnection>,
}

/// Owns a source and its registered sink until adoption; dropped unadopted, it closes.
/// AMD keeps this owner armed throughout its own STT's drain.
pub struct OpeningLeg {
    channel: AudioChannel,
    sink: ResourceId,
    source: Owned<PendingLeg>,
}

/// Owns the socket from successful opening through explicit close or abandoned startup.
pub(super) struct RealtimeConnection {
    session: Option<Owned<Arc<dyn RealtimeSession>>>,
    background: Background,
}

impl RealtimeConnection {
    pub(super) fn new(session: Arc<dyn RealtimeSession>, background: &Background) -> Self {
        let clock = Arc::clone(&background.clock);
        let closer: Closer<Arc<dyn RealtimeSession>> = Box::new(move |session, deadline| {
            Box::pin(async move {
                if lk_agents_abi::before(&*clock, Some(deadline), session.aclose()).await.is_ok() {
                    return Closed::Clean;
                }
                log::error!(target: lk_agents_abi::LOG_TARGET, "realtime session cleanup timed out");
                Closed::TimedOut
            })
        });
        let session = Some(Owned::new(background, session, closer));
        Self { session, background: background.clone() }
    }

    /// `aclose()` off the loop; a clean close comes back as the session's `RealtimeWake::Closed`.
    pub(super) fn close(&mut self, tx: crate::voice::session::Tx, id: ResourceId) {
        let Some(session) = self.session.take() else { return };
        self.background.cleanup("realtime close", async move {
            if session.close().await == Closed::Clean {
                let wake = lk_agents_voice::agent_session::machine::activity::realtime::RealtimeWake::Closed;
                let _ = tx.send(VoiceEvent::RealtimeSession { session: id, wake });
            }
        });
    }
}

/// A session a switch's model opened, owned from its open: dropped unadopted, it closes.
pub struct OpenedRealtime {
    connection: RealtimeConnection,
    session: Arc<dyn RealtimeSession>,
}

impl OpenedRealtime {
    pub(super) fn into_parts(self) -> (RealtimeConnection, Arc<dyn RealtimeSession>) {
        (self.connection, self.session)
    }
}

impl OpeningLeg {
    pub(super) fn new(
        channel: AudioChannel,
        input: Arc<dyn StreamInput<AudioFrame>>,
        source: PendingLeg,
        resources: Resources,
        background: &Background,
    ) -> Self {
        let sink = resources.register_audio_sink(input.clone());
        let closer: Closer<PendingLeg> = Box::new(move |source, deadline| {
            input.end_input();
            Box::pin(async move {
                // Held to the end, whatever the registry and the frames do meanwhile.
                let _input = input;
                let outcome = match source {
                    PendingLeg::Stt(source) => source.close(deadline).await,
                    PendingLeg::Vad(source) => source.close(deadline).await,
                };
                if outcome != Closed::Clean {
                    log::error!(target: lk_agents_abi::LOG_TARGET, "opening {} stream cleanup did not settle cleanly: {outcome:?}", channel.leg());
                }
                resources.release(sink);
                outcome
            })
        });
        Self { channel, sink, source: Owned::new(background, source, closer) }
    }

    pub(super) fn adopt(self) -> (AudioChannel, ResourceId, PendingLeg) {
        (self.channel, self.sink, self.source.into_inner())
    }

    pub(super) fn resource(&self) -> ResourceId {
        self.sink
    }

    /// The STT source, for a reader that keeps this owner armed while it reads: AMD's own STT.
    pub(super) fn stt_source(&mut self) -> Option<&mut lk_agents_abi::SttSource> {
        match &mut *self.source {
            PendingLeg::Stt(source) => Some(source),
            PendingLeg::Vad(_) => None,
        }
    }

    /// Ends the input, closes the source and releases the sink, bounded by the close's deadline.
    pub(super) async fn close(self) -> Closed {
        self.source.close().await
    }
}

impl Executor {
    /// Preparation and abandoned-stream cleanup use the supplied session clock and scheduler.
    pub fn new(time: Timebase, spawner: Arc<dyn Spawner>) -> Self {
        let background = Background::new(Arc::default(), spawner, time.clock());
        Self::owned_by(time, background)
    }

    /// Runs the session on `background`'s tasks, which its close drains.
    pub fn owned_by(time: Timebase, background: Background) -> Self {
        Self { time, background, opening: Vec::new(), realtime: None }
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
            lk_agents_inference::tts::sentence_adapter::StreamAdapter::adapted(tts, clock, budget)
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
            let (time, background) = (self.time.clone(), &self.background);
            let mut legs = OpeningLegs::default();
            let mut opening = Vec::new();
            if let Some(stt) = stt {
                let options = stt.conn_options().unwrap_or_default();
                let clock = time.clock();
                let (sink, source) =
                    lk_agents_inference::stt::stream(stt, options, clock, &cx, None, None)
                        .await?
                        .ok_or(DriverError::Missing("a streaming STT"))?
                        .split();
                let leg = OpeningLeg::new(
                    AudioChannel::Stt,
                    Arc::from(sink),
                    PendingLeg::Stt(source),
                    resources.clone(),
                    background,
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
                    background,
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
        tools: Option<lk_agents_abi::HostRef>,
        cx: telemetry::Cx,
    ) -> BoxFuture<'a, Result<(Arc<dyn RealtimeSession>, ResourceId), DriverError>> {
        Box::pin(async move {
            let session = model.session(cx, disabled, tools).await?;
            self.realtime = Some(RealtimeConnection::new(session.clone(), &self.background));
            let sink = resources
                .register_audio_sink(Arc::new(RealtimeAudioSink { session: session.clone() }));
            Ok((session, sink))
        })
    }

    fn tasks(&self) -> Arc<TaskSet> {
        Arc::clone(&self.background.tasks)
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

    fn head(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.session.head(frame);
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        // Not `aclose`: a handoff replaces the leg, not the socket.
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
    /// The socket's events, each tagged with the session `id` they came from.
    pub(super) fn pump_realtime(
        &self,
        id: ResourceId,
        session: Arc<dyn lk_agents_abi::RealtimeSession>,
    ) {
        let tx = self.reports.clone();
        let closing = self.shared.closing.clone();
        self.spawn("realtime events", async move {
            use lk_agents_voice::agent_session::machine::activity::realtime::RealtimeWake;
            loop {
                let event = tokio::select! {
                    biased;
                    () = closing.cancelled() => return,
                    () = tx.closed() => return,
                    event = session.next_event() => event,
                };
                let Some(event) = event else {
                    let wake = RealtimeWake::Ended;
                    let _ = tx.send(VoiceEvent::RealtimeSession { session: id, wake });
                    return;
                };
                let wake = RealtimeWake::Event(event);
                if tx.send(VoiceEvent::RealtimeSession { session: id, wake }).is_err() {
                    return;
                }
            }
        });
    }

    /// Opens `model`'s session off the loop under `span`, with the session's turn-detection
    /// choice and tools, and reports it as `RealtimeOpened`. A close abandons an open still in
    /// flight; a session it opened is closed.
    pub(super) fn open_realtime(
        &mut self,
        open: u64,
        model: ResourceId,
        turn_detection_disabled: bool,
        span: Option<lk_agents_abi::telemetry::record::SpanContext>,
    ) {
        let tx = self.reports.clone();
        let Some(model) = self.shared.resources.realtime_model(model) else {
            let error = ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::InvalidRequest,
                "the realtime model is no longer available",
            ));
            let provider = lk_agents_abi::ProviderId::of_label("realtime");
            let _ = tx.send(crate::voice::events::RealtimeOpened {
                open,
                provider,
                outcome: Err(error),
            });
            return;
        };
        let cx = self.shared.trace.clone().continuing(span);
        let tools = self.shared.realtime_tools.clone();
        let background = self.shared.background();
        let closing = self.shared.closing.clone();
        self.spawn("realtime open", async move {
            // Abandoning the open is safe: a host open dropped unsettled is cancelled.
            let outcome = tokio::select! {
                biased;
                () = closing.cancelled() => return,
                outcome = model.session(cx, turn_detection_disabled, tools) => outcome,
            };
            let outcome = outcome.map(|session| OpenedRealtime {
                connection: RealtimeConnection::new(session.clone(), &background),
                session,
            });
            let provider = model.id();
            // A refused send drops the opened session, which closes it, joined by the session.
            let _ = tx.send(crate::voice::events::RealtimeOpened { open, provider, outcome });
        });
    }

    /// An opened session joins the table and is read; the core decides whether it is used.
    pub(super) fn adopt_realtime(
        &mut self,
        opened: crate::voice::events::RealtimeOpened,
    ) -> VoiceEvent {
        let crate::voice::events::RealtimeOpened { open, provider, outcome } = opened;
        let outcome = outcome.map(|opened| {
            let (connection, session) = opened.into_parts();
            let resources = &self.shared.resources;
            let reference = resources.register_realtime(session.clone(), provider.clone());
            let sink = resources
                .register_audio_sink(Arc::new(RealtimeAudioSink { session: session.clone() }));
            self.realtime.insert(reference.id, connection);
            self.shared.realtime_spans.set_model(Some(provider));
            self.pump_realtime(reference.id, session);
            (reference, sink)
        });
        VoiceEvent::RealtimeOpened { open, outcome }
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
        let token = shared.closing.child_token();
        self.drains.retain(|_, abort| !abort.is_cancelled());
        self.drains.insert(sink, token.clone());
        let abort = super::audio::Abort { token, resources: shared.resources.clone() };
        let (time, ends) = (shared.time.clone(), (shared.closing.clone(), abort));
        match stream {
            PendingLeg::Stt(mut source) => {
                source.report_retries(super::audio::report_to(self.reports.clone()));
                super::audio::spawn_drain(
                    (&shared.tasks, shared.spawner.as_ref()),
                    time,
                    ends,
                    super::audio::tagged(sink, source),
                    self.reports.clone(),
                    (channel, sink),
                );
            }
            PendingLeg::Vad(source) => super::audio::spawn_drain(
                (&shared.tasks, shared.spawner.as_ref()),
                time,
                ends,
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
        let task = self.scoped("turn prediction", async move {
            let outcome = detector.predict(&telemetry::Cx::NONE, timeout).await;
            if !closing.is_cancelled() {
                let _ = queue.send(VoiceEvent::Eot(req, outcome, None));
            }
        });
        self.predictions.push(task);
    }

    /// Ends on the detector's `Closed` (forwarded first), when the loop is gone, or at `Closing`.
    /// None is an error: the session goes on with VAD-only interruption.
    fn pump_verdicts(&self, detector: Arc<dyn InterruptionPort>) {
        let tx = self.reports.clone();
        let closing = self.shared.closing.clone();
        self.spawn("interruption verdicts", async move {
            loop {
                let ev = tokio::select! {
                    biased;
                    () = closing.cancelled() => break,
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
        let closing = self.shared.closing.clone();
        self.spawn("event forward", async move {
            loop {
                let ev = tokio::select! {
                    biased;
                    ev = rx.recv() => ev,
                    () = closing.cancelled() => None,
                };
                if ev.is_none_or(|ev| tx.send(ev).is_err()) {
                    break;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the audio each way it arrives.
    #[derive(Default)]
    struct Heard(parking_lot::Mutex<Vec<&'static str>>);
    impl RealtimeSession for Heard {
        fn capabilities(&self) -> lk_agents_abi::llm::realtime::RealtimeCapabilities {
            Default::default()
        }
        fn send(&self, cmd: RealtimeCommand) {
            if let RealtimeCommand::PushAudio(_) = cmd {
                self.0.lock().push("push");
            }
        }
        fn head(&self, _: AudioFrame) {
            self.0.lock().push("head");
        }
        fn next_event(&self) -> BoxFuture<'_, Option<lk_agents_abi::llm::realtime::Event>> {
            Box::pin(std::future::pending())
        }
        fn chat_ctx(&self) -> BoxFuture<'_, lk_agents_abi::ChatContext> {
            Box::pin(async { Default::default() })
        }
        fn aclose(&self) -> BoxFuture<'_, ()> {
            Box::pin(async {})
        }
    }

    // Catches: the realtime leg pushing pre-connect audio as live audio, so the session's input
    // policy drops the start of the first turn.
    #[test]
    fn pre_connect_audio_reaches_the_session_as_head() {
        let session = Arc::new(Heard::default());
        let sink = RealtimeAudioSink { session: session.clone() };
        let frame = || AudioFrame::new(vec![0u8; 960], 48_000, 1, 480);
        sink.head(frame()).unwrap();
        sink.push(frame()).unwrap();
        assert_eq!(*session.0.lock(), ["head", "push"]);
    }
}
