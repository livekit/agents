use super::legs::{OpeningLeg, PendingLeg};
use crate::rtc::wait;
use crate::telemetry;
use crate::voice::session::PeerHandle;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::amd::{
    SIP_CALL_STATUS_ACTIVE, SIP_CALL_STATUS_ATTR, TRACK_PUBLICATION_TIMEOUT,
};
use lk_agents_abi::voice::rtc::RtcRoom;
use lk_agents_abi::{
    AudioFrame, FunctionToolCall, HostRef, Llm, SpeechEvent, StreamInput, Stt, TimerSet, Ts,
};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::agent_session::{
    AmdSpan, Command, IvrCommand, SpeechCommand, SystemTap, TurnCommand, VoiceCommandOrigin,
};
pub use lk_agents_voice::amd::AmdSetup;
use lk_agents_voice::amd::effects::AmdEffect;
use lk_agents_voice::amd::machine::classifier::{
    Amd, AmdTimer, Authorization, ClassifyRequest, In, TapPhase,
};
use lk_agents_voice::amd::{Callee, feed_input, tap_input};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{
    ChannelEndCause, CoreStep, DetectorName, DetectorStep, InputStep, Step, TimerName,
};
use std::fmt;
use std::sync::Arc;

use std::sync::atomic::{AtomicBool, Ordering};

pub use lk_agents_voice::amd::machine::classifier::{
    POSTPONE_TERMINATION_TOOL, SAVE_PREDICTION_TOOL,
};
use lk_agents_voice::amd::machine::classifier::{classification_request, classification_verdicts};

const LOG: &str = "livekit.agents";
const TAP: &str = "amd";

/// The host's end of the system: `Start`, `Verdict::Settle`, `Close`.
#[derive(Clone)]
pub struct AmdHandle {
    tx: tokio::sync::mpsc::UnboundedSender<Msg>,

    listening: Arc<AtomicBool>,
    /// The system's tasks; they end with it, and dropping the handle cancels none.
    _tasks: Arc<TaskSet>,
}

impl AmdHandle {
    pub fn input(&self, input: In) {
        let _ = self.tx.send(Msg::Input(input));
    }

    fn detached() -> Self {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        AmdHandle { tx, listening: Arc::default(), _tasks: Arc::default() }
    }
}

impl lk_agents_abi::voice::amd::AmdPort for AmdHandle {
    fn execute(&self) {
        self.input(In::Execute);
    }

    fn close(&self) {
        self.input(In::Close);
    }

    fn listening(&self) -> bool {
        self.listening.load(Ordering::Relaxed)
    }
}

/// Start one detector beside its session executor.
pub fn spawn(peer: PeerHandle, setup: AmdSetup) -> Arc<dyn lk_agents_abi::voice::amd::AmdPort> {
    Arc::new(AmdSystem::spawn(peer, setup))
}

impl fmt::Debug for AmdHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AmdHandle")
    }
}

enum Msg {
    Input(In),
    /// Classification `req` ended with every tool call answered, in order, and the request's own
    /// transcript. `failed`: no stream or it errored; the timers decide.
    Answer {
        req: u64,
        transcript: String,
        calls: Vec<FunctionToolCall>,
        failed: bool,
        /// The classification's `LLMMetrics`, as the classifier's `LLMStream` reports them.
        metrics: Option<Box<lk_agents_abi::metrics::base::LlmMetrics>>,
    },
    /// AMD's own STT stream opened -- or refused -- off the loop (`_run_stt`'s `stt.stream()`).
    SttOpened(Result<(OpeningLeg, Arc<AtomicBool>), ProviderError>),
    /// One event off the tap's drain (`_receive`, `detector._receive`).
    Transcript(SpeechEvent),
    /// What the room said of the callee ([`await_callee`]).
    Callee(Callee),
}

enum SttLeg {
    /// No STT of AMD's own: the session's finals feed the machine.
    None,
    Opening,
    /// On the forwarder's `Amd` channel as `tap`, behind `gate`: frames flow while the machine
    /// listens.
    Installed {
        gate: Arc<AtomicBool>,
        tap: lk_agents_abi::resources::ResourceId,
    },
    /// Cleared; an open that lands now is discarded.
    Done,
}

/// The tap's sink behind the listening gate: pre-answer audio is never
/// transcribed. `end_input` passes through.
struct GatedSink {
    inner: Arc<dyn StreamInput<AudioFrame>>,

    open: Arc<AtomicBool>,
    ended: AtomicBool,
}

impl StreamInput<AudioFrame> for GatedSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if self.open.load(Ordering::Relaxed) && !self.ended.load(Ordering::Relaxed) {
            self.inner.push(item)
        } else {
            Ok(())
        }
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.inner.flush()
    }

    fn end_input(&self) {
        if !self.ended.swap(true, Ordering::Relaxed) {
            self.inner.end_input();
        }
    }
}

pub struct AmdSystem {
    peer: PeerHandle,

    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
    machine: Option<Amd>,
    timers: TimerSet<AmdTimer>,
    prompt: String,
    llm: Arc<dyn Llm>,
    tools_host: Option<HostRef>,
    /// The detector's `amd` span: each classification is an `llm_request` under it.
    cx: telemetry::Cx,
    inflight: Option<(u64, tokio::sync::watch::Sender<bool>)>,
    stt: SttLeg,
    /// The session asked to close (or the machine was closed): nothing more is read.
    closed: bool,

    /// The machine's [`Amd::listening`], for the host's `AMD.started`.
    listening: Arc<AtomicBool>,
    /// Its run, the callee wait, the tap's opening and drain, and the classifications.
    tasks: Arc<TaskSet>,
}

impl AmdSystem {
    /// `AMD._run`: holding authorization from now, subscribed to the feed, the tap opening and the
    /// callee awaited off the loop. Past the close request nothing is attached and the handle
    /// drops what it is sent.
    pub fn spawn(peer: PeerHandle, setup: AmdSetup) -> AmdHandle {
        if *peer.closing.borrow() {
            peer.journal(Step::Core(CoreStep::PortFailure("amd: attached after the close")));
            return AmdHandle::detached();
        }
        let rx = peer.subscribe();
        let (tx, msg_rx) = tokio::sync::mpsc::unbounded_channel();
        // "closing it so AMD can manage the IVR lifecycle": the session navigator is closed
        // before the hold; `execute()` on a machine-ivr verdict starts AMD's own.
        let _ = peer.send(VoiceCommandOrigin::Amd, IvrCommand::Close);
        let (room, participant) = (setup.room.clone(), setup.participant.clone());
        let system = AmdSystem::new(peer, setup, tx.clone());
        let listening = system.listening.clone();
        let tasks = system.tasks.clone();
        let spawner = system.peer.spawner.clone();
        let time = system.peer.time.clone();
        let callee_tx = tx.clone();
        let _ = tasks.try_spawn(spawner.as_ref(), "amd callee", async move {
            tokio::select! {
                biased;
                () = callee_tx.closed() => {}
                callee = await_callee(room, participant, time) => {
                    let _ = callee_tx.send(Msg::Callee(callee));
                }
            }
        });
        let _ = tasks.try_spawn(spawner.as_ref(), "amd", system.run(rx, msg_rx));
        AmdHandle { tx, listening, _tasks: tasks }
    }

    fn new(peer: PeerHandle, setup: AmdSetup, tx: tokio::sync::mpsc::UnboundedSender<Msg>) -> Self {
        let stt = setup.stt.clone();
        // The machine holds the authorization from the attach.
        let (machine, out) = Amd::attached(setup.config);
        let span = telemetry::reserve::<telemetry::Amd>(&peer.session);
        let cx = peer.session.clone().under::<telemetry::Amd>(span.context);
        let mut system = AmdSystem {
            peer,
            tx,
            machine: Some(machine),
            timers: TimerSet::new(),
            prompt: setup.prompt,
            llm: setup.llm,
            tools_host: setup.tools_host,
            cx,
            inflight: None,
            stt: SttLeg::None,
            closed: false,
            listening: Arc::default(),
            tasks: Arc::default(),
        };
        let _ =
            system.peer.send(VoiceCommandOrigin::Amd, Command::AmdSpan(AmdSpan::Attached(span)));
        if let Some(stt) = stt {
            system.open_tap(stt);
            system.stt = SttLeg::Opening;
        }
        let now = system.peer.time.now();
        system.perform(now, out);
        system
    }

    /// The loop: a due deadline fires before same-instant items; ends at close or when the
    /// session's queue is gone.
    async fn run(
        mut self,

        mut rx: tokio::sync::mpsc::UnboundedReceiver<SystemTap>,

        mut msgs: tokio::sync::mpsc::UnboundedReceiver<Msg>,
    ) {
        let time = self.peer.time.clone();
        let api = self.peer.queue.clone();
        let mut closing = self.peer.closing.clone();
        // The host may drop its handle: its channel then never yields, the feed still does.
        let mut host_open = true;
        while !self.closed {
            if *closing.borrow() || api.is_closed() {
                break;
            }
            let deadline = self.timers.next_deadline();
            if deadline.is_some_and(|at| at <= time.now()) {
                self.on_timers(time.now());
                continue;
            }
            tokio::select! {
                biased;
                _ = closing.wait_for(|closed| *closed) => break,
                () = api.closed() => break,
                () = time.wait_until_opt(deadline) => self.on_timers(time.now()),
                item = rx.recv() => match item {
                    Some(item) => self.on_watched(time.now(), item),
                    None => break,
                },
                msg = async { if host_open { msgs.recv().await } else { std::future::pending().await } } => match msg {
                    Some(msg) => self.on_msg(time.now(), msg),
                    None => host_open = false,
                },
            }
        }
        if !self.closed {
            self.input(time.now(), In::Close);
        }
        // Results own their close even when a sender races this shutdown or the run is dropped.
        msgs.close();
        while let Some(msg) = msgs.recv().await {
            if let Msg::SttOpened(Ok((opened, _))) = msg {
                self.peer.journal(Step::Input(InputStep::LegOpenDiscarded { leg: TAP }));
                opened.retire(&self.tasks);
            }
        }
    }

    /// One item off the session feed.
    pub fn on_watched(&mut self, now: Ts, item: SystemTap) {
        if let Some(input) = feed_input(item) {
            self.input(now, input);
        }
    }

    /// A close (the session's or the host's) ends the loop; a verdict still owed is never
    /// released.
    fn input(&mut self, now: Ts, input: In) {
        let closing = matches!(input, In::Close);
        self.step(now, input);
        if closing {
            self.closed = true;
        }
    }

    fn on_msg(&mut self, now: Ts, msg: Msg) {
        match msg {
            Msg::Input(input) => self.input(now, input),
            Msg::Answer { req, transcript, calls, failed, metrics } => {
                // Through the session, so its usage is accounted and counted.
                if let Some(m) = metrics {
                    let metrics = Box::new(lk_agents_abi::metrics::base::AgentMetrics::Llm(*m));
                    let _ = self.peer.send(VoiceCommandOrigin::Amd, Command::HostMetrics(metrics));
                }
                self.on_answer(now, req, transcript, calls, failed)
            }
            Msg::SttOpened(outcome) => self.on_tap_opened(outcome),
            Msg::Transcript(ev) => self.on_tap_transcript(now, ev),
            Msg::Callee(callee) => {
                match &callee {
                    Callee::Answered => log::debug!(target: LOG, "AMD starts listening"),
                    Callee::Missing(error) => log::debug!(target: LOG, error = error.as_str();
                        "AMD: call ended before detection could run, settling"),
                }
                for input in callee.inputs() {
                    self.step(now, input);
                }
            }
        }
    }

    /// Fire the deadlines due at `now` in order; one cancelled by an earlier fire this pass is skipped.
    pub fn on_timers(&mut self, now: Ts) {
        while let Some(timer) = self.timers.pop_due(now) {
            self.peer.journal(Step::Core(CoreStep::TimerFired(TimerName::Amd(timer), now)));
            self.step(now, In::Timeout(timer));
        }
    }

    pub fn step(&mut self, now: Ts, input: In) {
        let Some(machine) = self.machine.take() else { return };
        let (next, out) = machine.step(now, input);
        self.machine = Some(next);
        self.perform(now, out);
        self.sync_tap();
        let listening = self.machine.as_ref().is_some_and(Amd::listening);
        self.listening.store(listening, Ordering::Relaxed);
    }

    fn perform(&mut self, now: Ts, effects: Vec<AmdEffect>) {
        for effect in effects {
            match effect {
                // Arming replaces, a cancel removes; both journalled.
                AmdEffect::SetDeadline(timer, at) => {
                    self.timers.set_deadline(timer, at);
                    self.peer.journal(Step::Core(CoreStep::TimerArmed {
                        timer: TimerName::Amd(timer),
                        at,
                    }));
                }
                AmdEffect::ClearDeadline(timer) => {
                    self.timers.clear_deadline(timer);
                    self.peer.journal(Step::Core(CoreStep::TimerCancelled(TimerName::Amd(timer))));
                }
                // The waiter is the host's (`execute()` awaits `amd_result`): nothing to wake here.
                // `AMD.emit("amd_prediction")`; `_on_amd_prediction` then cancels the classification.
                AmdEffect::Prediction(p) => {
                    let verdict = Command::AmdSpan(AmdSpan::Verdict(p.clone()));
                    let _ = self.peer.send(VoiceCommandOrigin::Amd, verdict);
                    self.peer.emit(SessionEvent::AmdResult { verdict: p });
                    self.step(now, In::EndInput);
                }
                AmdEffect::RequestClassification(req) => self.request_classification(req),
                AmdEffect::CancelClassification => self.cancel_classification(),
                // The postpone reply is discarded (`classifier._run`).
                AmdEffect::Refused(refusal) => {
                    self.peer.journal(Step::Detector(DetectorStep::AmdRefused(refusal)))
                }
                // The scheduler's hold, as the system's command.
                AmdEffect::Authorization(Authorization::Pause) => {
                    let _ =
                        self.peer.send(VoiceCommandOrigin::Amd, SpeechCommand::PauseAuthorization);
                }
                AmdEffect::Authorization(Authorization::Resume) => {
                    let _ =
                        self.peer.send(VoiceCommandOrigin::Amd, SpeechCommand::ResumeAuthorization);
                }
                AmdEffect::SkipReplies(skips) => {
                    let _ =
                        self.peer.send(VoiceCommandOrigin::Amd, TurnCommand::SkipReplies(skips));
                }
                // `interrupt(force=True)`: the greeting is being spoken over.
                AmdEffect::Interrupt { force } => {
                    let _ = self.peer.send(
                        VoiceCommandOrigin::Amd,
                        SpeechCommand::Interrupt { force, call: None },
                    );
                }
            }
        }
    }

    /// One classification with `tool_choice="required"`.
    fn request_classification(&mut self, req: ClassifyRequest) {
        let n = req.id;
        self.peer.journal(Step::Input(InputStep::LegRequested { leg: DetectorName::Amd, req: n }));
        let request = classification_request(
            &self.prompt,
            self.tools_host.clone(),
            self.peer.time.wall(),
            &req,
        );
        let (cancel, cancelled) = tokio::sync::watch::channel(false);
        self.inflight = Some((n, cancel));
        let llm = Arc::clone(&self.llm);
        let time = self.peer.time.clone();
        let tx = self.tx.clone();
        let cx = self.cx.clone();
        let _ = self.tasks.try_spawn(
            self.peer.spawner.as_ref(),
            "amd classification",
            classify(time, tx, llm, request, n, req.transcript, cancelled, cx),
        );
    }

    /// `end_input()` / `close()`: the run in flight is cancelled and reports nothing.
    fn cancel_classification(&mut self) {
        if let Some((_, cancel)) = self.inflight.take() {
            let _ = cancel.send(true);
        }
    }

    /// Perform every tool call the model answered with, in order. A bad label or a postpone with no
    /// number yields no verdict, as the reference swallows it. A superseded run still counts.
    fn on_answer(
        &mut self,
        now: Ts,
        req: u64,
        transcript: String,
        calls: Vec<FunctionToolCall>,
        failed: bool,
    ) {
        if self.inflight.as_ref().is_some_and(|(r, _)| *r == req) {
            self.inflight = None;
        }
        if failed {
            self.peer.journal(Step::Input(InputStep::LegFailed {
                leg: DetectorName::Amd,
                recoverable: true,
            }));
            return;
        }
        self.peer.journal(Step::Input(InputStep::LegVerdict { leg: DetectorName::Amd, req }));
        for verdict in classification_verdicts(calls, &transcript) {
            self.step(now, In::Verdict(verdict));
        }
    }

    /// `_run_stt`'s `stt.stream()`, off the loop; the stream comes back as `Msg::SttOpened`.
    fn open_tap(&self, stt: Arc<dyn Stt>) {
        stt.prewarm();
        let tx = self.tx.clone();
        let peer = self.peer.clone();
        let tasks = self.tasks.clone();
        let _ = self.tasks.try_spawn(self.peer.spawner.as_ref(), "amd tap open", async move {
            if *peer.closing.borrow() || tx.is_closed() {
                return;
            }
            // Let an admitted host open settle so its successful result can acquire ownership.
            let result = stt.stream(None, None, Default::default()).await;
            let outcome = match result {
                Ok(Some(pair)) => {
                    let (sink, source) = pair.split();
                    let gate = Arc::new(AtomicBool::new(false));
                    let input = Arc::new(GatedSink {
                        inner: Arc::from(sink),
                        open: gate.clone(),
                        ended: AtomicBool::new(false),
                    });
                    Ok((
                        OpeningLeg::new(
                            lk_agents_voice::audio::AudioChannel::Amd,
                            input,
                            PendingLeg::Stt(source),
                            peer.resources.clone(),
                            peer.time.clone(),
                            peer.spawner.clone(),
                        ),
                        gate,
                    ))
                }
                Err(error) => Err(error),
                Ok(None) => Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                    "the AMD STT does not stream",
                ))),
            };
            if let Err(tokio::sync::mpsc::error::SendError(Msg::SttOpened(Ok((opened, _))))) =
                tx.send(Msg::SttOpened(outcome))
            {
                opened.retire(&tasks);
            }
        });
    }

    /// Keep ownership armed through admission and the drain's first poll.
    fn on_tap_opened(&mut self, outcome: Result<(OpeningLeg, Arc<AtomicBool>), ProviderError>) {
        if !matches!(self.stt, SttLeg::Opening) {
            if let Ok((opened, _)) = outcome {
                self.peer.journal(Step::Input(InputStep::LegOpenDiscarded { leg: TAP }));
                opened.retire(&self.tasks);
            }
            return;
        }
        let (opened, gate) = match outcome {
            Ok(opened) => opened,
            Err(_) => {
                self.stt = SttLeg::Done;
                self.peer.journal(Step::Core(CoreStep::PortFailure(
                    "amd: the AMD STT could not be opened",
                )));
                return;
            }
        };
        gate.store(self.machine.as_ref().is_some_and(Amd::listening), Ordering::Relaxed);
        let tap = opened.resource();
        if self
            .peer
            .queue
            .send(lk_agents_voice::data_plane::AudioInputRouterCommand::Install {
                channel: lk_agents_voice::audio::AudioChannel::Amd,
                sink: tap,
            })
            .is_err()
        {
            self.stt = SttLeg::Done;
            opened.retire(&self.tasks);
            return;
        }
        self.stt = SttLeg::Installed { gate, tap };
        self.peer.journal(Step::Input(InputStep::StreamOpened { leg: TAP }));
        if !spawn_tap_drain(&self.tasks, self.peer.clone(), opened, tap, self.tx.clone()) {
            self.stt = SttLeg::Done;
            self.peer.remove_audio_tap(tap);
        }
    }

    fn on_tap_transcript(&mut self, now: Ts, ev: SpeechEvent) {
        if let Some(input) = tap_input(&ev) {
            self.step(now, input);
        }
    }

    fn sync_tap(&mut self) {
        let Some(phase) = self.machine.as_ref().map(Amd::tap) else { return };
        match (phase, &self.stt) {
            (TapPhase::Open, SttLeg::Installed { gate, .. }) => gate.store(true, Ordering::Relaxed),
            (TapPhase::Over, SttLeg::Installed { tap, .. }) => {
                let tap = *tap;
                self.stt = SttLeg::Done;
                self.peer.remove_audio_tap(tap);
            }
            (TapPhase::Over, SttLeg::Opening) => self.stt = SttLeg::Done,
            _ => {}
        }
    }
}

/// `AMD._setup`: the callee's audio track (subscribed), then for a SIP callee the call answered
/// (`sip.callStatus == "active"`). No track within [`TRACK_PUBLICATION_TIMEOUT`], a publisher gone
/// before it is looked up, or a call that ends unanswered is `participant_missing`, rather than
/// listening for the whole no-speech threshold. Without a room the timers start at once.
async fn await_callee(
    room: Option<Arc<dyn RtcRoom>>,
    participant: Option<String>,
    time: crate::voice::session::Timebase,
) -> Callee {
    let Some(room) = room else {
        log::warn!(target: LOG,
            "session room_io unavailable, starting amd timers immediately as fallback");
        return Callee::Answered;
    };
    let audio = Some(vec![lk_agents_abi::voice::rtc::TRACK_KIND_AUDIO]);
    let track = wait::track_publication(room.as_ref(), participant, audio, false, true);
    let publisher = match time.timeout(TRACK_PUBLICATION_TIMEOUT, track).await {
        Err(_) => return Callee::Missing("timed out waiting for participant audio track".into()),
        Ok(Err(e)) => return Callee::Missing(e.to_string()),
        Ok(Ok((publisher, _))) => publisher,
    };
    let Some(identity) = publisher else {
        return Callee::Missing("participant disappeared after track subscription".into());
    };
    let publisher = match room.participant(&identity) {
        Ok(Some(publisher)) => publisher,
        Ok(None) => {
            return Callee::Missing("participant disappeared after track subscription".into());
        }
        Err(error) => return Callee::Missing(error.to_string()),
    };
    if !Callee::awaits_sip_answer(publisher.kind()) {
        return Callee::Answered;
    }
    let (attribute, active) = (SIP_CALL_STATUS_ATTR.to_owned(), SIP_CALL_STATUS_ACTIVE.to_owned());
    let identity = publisher.identity().as_str().to_owned();
    match wait::participant_attribute(room.as_ref(), identity, attribute, active).await {
        Ok(()) => Callee::Answered,
        Err(e) => Callee::Missing(e.to_string()),
    }
}

/// The tap's drain onto the system's channel: source closed once, tap removed if still installed.
fn spawn_tap_drain(
    tasks: &Arc<TaskSet>,
    peer: PeerHandle,
    mut opened: OpeningLeg,
    tap: lk_agents_abi::resources::ResourceId,

    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
) -> bool {
    let mut closing = peer.closing.clone();
    let spawner = peer.spawner.clone();
    tasks
        .try_spawn(spawner.as_ref(), "amd tap drain", async move {
            let Some(source) = opened.stt_source() else {
                peer.remove_audio_tap(tap);
                return;
            };
            let cause = loop {
                let next = tokio::select! {
                    biased;
                    _ = closing.wait_for(|c| *c) => None,
                    () = tx.closed() => None,
                    item = source.next() => Some(item),
                };
                match next {
                    Some(Ok(Some(ev))) => {
                        if tx.send(Msg::Transcript(ev)).is_err() {
                            break ChannelEndCause::Closing;
                        }
                    }
                    Some(Ok(None)) => break ChannelEndCause::SourceEnded,
                    Some(Err(_)) => break ChannelEndCause::Error,
                    None => break ChannelEndCause::Closing,
                }
            };
            peer.remove_audio_tap(tap);
            let closed = opened.close().await;
            peer.journal(Step::Input(InputStep::ChannelEnded { leg: TAP, cause }));
            if let Some(how) = closed {
                peer.journal(Step::Input(InputStep::HostStreamClosed { label: TAP, how }));
            }
        })
        .is_some()
}

/// The classification task, an `llm_request` under the detector's `amd` span. A cancelled run
/// reports nothing. A native source may be abandoned mid-poll; a host stream may not, so the
/// cancel waits for the chunk.
#[allow(clippy::too_many_arguments)]
async fn classify(
    time: crate::voice::session::Timebase,

    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
    llm: Arc<dyn Llm>,
    mut request: lk_agents_abi::ChatRequest,
    req: u64,
    transcript: String,
    mut cancelled: tokio::sync::watch::Receiver<bool>,
    cx: telemetry::Cx,
) {
    let started_at = time.wall_secs();
    let mut trace = telemetry::LlmTrace::start(&cx, llm.as_ref(), &mut request);
    let mut attempt = Some(trace.attempt());
    let mut stream = match llm.chat(&request).await {
        Ok(s) => s,
        Err(error) => {
            trace.end_attempt(attempt.take().flatten(), Some(&error));
            trace.fail(&error);
            let (calls, metrics) = (Vec::new(), None);
            let _ = tx.send(Msg::Answer { req, transcript, calls, failed: true, metrics });
            return;
        }
    };
    let mut fold = telemetry::LlmMetricState::new(started_at);
    let mut calls: Vec<FunctionToolCall> = Vec::new();
    let mut failed = false;
    let mut aborted = false;
    loop {
        if *cancelled.borrow() {
            aborted = true;
            break;
        }
        let next = if stream.cancel_safe() {
            tokio::select! {
                biased;
                _ = cancelled.wait_for(|c| *c) => {
                    aborted = true;
                    break;
                }
                item = stream.next() => item,
            }
        } else {
            stream.next().await
        };
        match next {
            Ok(Some(chunk)) => {
                trace.observe(&chunk);
                fold.observe(&chunk, || time.wall_secs());
                if let Some(delta) = &chunk.delta {
                    calls.extend(delta.tool_calls.iter().cloned());
                }
            }
            Ok(None) => break,
            Err(error) => {
                trace.end_attempt(attempt.take().flatten(), Some(&error));
                trace.fail(&error);
                failed = true;
                break;
            }
        }
    }
    // A cancelled classification's request ends as cancelled when dropped.
    if !aborted && let Some(attempt) = attempt.take() {
        trace.end_attempt(attempt, None);
        trace.end(false, None);
    }
    // Always, on every path: dropping does not cancel a host stream, and `close` consumes it.
    let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    let _ = stream.close(deadline).await;
    if aborted {
        return;
    }
    let metrics = fold.finish(failed, || {
        let (id, now) = (llm.id(), time.wall_secs());
        let label = id.label.clone();
        let metadata = Some(id.metadata());
        let context = telemetry::Context { timestamp: now, label, cancelled: false, metadata };
        (now, context)
    });
    let metrics = metrics.map(Box::new);
    let _ = tx.send(Msg::Answer { req, transcript, calls, failed, metrics });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Default)]
    struct Input(AtomicUsize);
    impl StreamInput<AudioFrame> for Input {
        fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Source {
        input: Arc<Input>,
        closed: Arc<tokio::sync::Notify>,
    }
    impl lk_agents_abi::StreamOutput<SpeechEvent> for Source {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
            Box::pin(std::future::pending())
        }
        fn close(
            self: Box<Self>,
            _: Ts,
        ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
            Box::pin(async move {
                assert_eq!(self.input.0.load(Ordering::SeqCst), 1);
                self.closed.notify_one();
                lk_agents_abi::Closed::Clean
            })
        }
    }
    impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for Source {}

    #[tokio::test(start_paused = true)]
    async fn unread_or_refused_successful_tap_result_owns_its_close() {
        for refused in [false, true] {
            let input = Arc::new(Input::default());
            let gate = Arc::new(AtomicBool::new(false));
            let sink = Arc::new(GatedSink {
                inner: input.clone(),
                open: gate.clone(),
                ended: AtomicBool::new(false),
            });
            let closed = Arc::new(tokio::sync::Notify::new());
            let resources = crate::voice::resources::Resources::default();
            let opened = OpeningLeg::new(
                lk_agents_voice::audio::AudioChannel::Amd,
                sink.clone(),
                PendingLeg::Stt(Box::new(Source { input: input.clone(), closed: closed.clone() })),
                resources.clone(),
                crate::voice::session::Timebase::from_virtual(Arc::new(
                    lk_agents_abi::VirtualClock::new(),
                )),
                Arc::new(lk_agents_net::task::TokioSpawner::current()),
            );
            let id = opened.resource();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            if refused {
                drop(rx);
                drop(tx.send(Msg::SttOpened(Ok((opened, gate)))));
            } else {
                assert!(tx.send(Msg::SttOpened(Ok((opened, gate)))).is_ok());
                drop(rx);
            }
            // Removal by the forwarder and owner cleanup may race; the provider sees one end.
            sink.end_input();
            tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, closed.notified())
                .await
                .expect("AMD result was abandoned without closing");
            assert_eq!(input.0.load(Ordering::SeqCst), 1);
            assert!(resources.audio_sink(id).is_none());
        }
    }

    /// A model that answers with one tool call and its usage.
    struct Classifier;
    struct Answer(Option<lk_agents_abi::LlmChunk>);
    impl lk_agents_abi::StreamOutput<lk_agents_abi::LlmChunk> for Answer {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<lk_agents_abi::LlmChunk>, ProviderError>>
        {
            Box::pin(async move { Ok(self.0.take()) })
        }
        fn close(
            self: Box<Self>,
            _: Ts,
        ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
            Box::pin(async move { lk_agents_abi::Closed::Clean })
        }
    }
    impl lk_agents_abi::utils::stream::ProviderStream<lk_agents_abi::LlmChunk> for Answer {}
    impl Llm for Classifier {
        fn label(&self) -> String {
            "openai.LLM".into()
        }
        fn chat<'a>(
            &'a self,
            _: &'a lk_agents_abi::ChatRequest,
        ) -> lk_agents_abi::BoxFuture<'a, Result<lk_agents_abi::LlmStream, ProviderError>> {
            let usage = lk_agents_abi::usage::TokenUsage {
                input_tokens: 40,
                output_tokens: 3,
                ..Default::default()
            };
            let chunk = lk_agents_abi::LlmChunk {
                id: "resp_amd".into(),
                usage: Some(Box::new(usage)),
                ..Default::default()
            };
            Box::pin(async move { Ok(Box::new(Answer(Some(chunk))) as lk_agents_abi::LlmStream) })
        }
    }

    #[derive(Default)]
    struct Records(parking_lot::Mutex<Vec<telemetry::Record>>);
    impl telemetry::RecordSink for Records {
        fn record(&self, record: telemetry::Record) {
            self.0.lock().push(record);
        }
    }

    // Catches: AMD's classification recorded outside its `amd` span, or with no request span,
    // so its tokens and timing never reach the trace.
    #[tokio::test(start_paused = true)]
    async fn a_classification_is_an_llm_request_under_amd() {
        let records = Arc::new(Records::default());
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let tracer = telemetry::Tracer::new(records.clone(), clock.clone(), 3);
        let amd = telemetry::SpanRef {
            trace_id: telemetry::TraceId(1),
            span_id: telemetry::SpanId(2),
            sampled: true,
        };
        let cx = tracer.root().under::<telemetry::Amd>(amd);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (_cancel, cancelled) = tokio::sync::watch::channel(false);
        let time = crate::voice::session::Timebase::from_virtual(clock);
        let request = lk_agents_abi::ChatRequest::default();
        let llm = Arc::new(Classifier) as Arc<dyn Llm>;
        classify(time, tx, llm, request, 1, "hello?".into(), cancelled, cx).await;
        let Some(Msg::Answer { metrics, failed: false, .. }) = rx.recv().await else {
            panic!("no answer")
        };
        assert!(metrics.is_some(), "its usage reaches the session");
        let spans: Vec<_> = (records.0.lock().iter())
            .filter_map(|r| if let telemetry::Record::Span(s) = r { Some(s.clone()) } else { None })
            .collect();
        let request = spans.iter().find(|s| s.name == "llm_request").expect("{spans:?}");
        assert_eq!(request.parent, Some(telemetry::SpanId(2)));
        let tokens = request.attributes.iter().find(|(k, _)| *k == "gen_ai.usage.input_tokens");
        assert_eq!(tokens.map(|(_, v)| v.clone()), Some(telemetry::Value::Int(40)));
        let run = spans.iter().find(|s| s.name == "llm_request_run").expect("{spans:?}");
        assert_eq!(run.parent, Some(request.context.span_id));
    }
}
