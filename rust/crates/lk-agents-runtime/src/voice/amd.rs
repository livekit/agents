use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::amd::{
    SIP_CALL_STATUS_ACTIVE, SIP_CALL_STATUS_ATTR, TRACK_PUBLICATION_TIMEOUT,
};
use lk_agents_abi::voice::rtc::RtcRoom;
use lk_agents_abi::{
    AudioFrame, FunctionToolCall, HostRef, Llm, SpawnerExt, SpeechEvent, StreamSink, Stt, SttSink,
    SttSource, TimerSet, Ts,
};
use lk_agents_voice::agent_session::{
    Command, IvrCommand, Origin, PeerHandle, SpeechCommand, SystemTap, TurnCommand,
};
pub use lk_agents_voice::amd::AmdSetup;
use lk_agents_voice::amd::classifier::{
    Amd, AmdTimer, Authorization, ClassifyRequest, Effect, In, Out,
};
use lk_agents_voice::amd::{Callee, feed_input, tap_input};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{
    ChannelEndCause, CoreStep, DetectorName, DetectorStep, InputStep, Step, TimerName,
};
use std::fmt;
use std::sync::Arc;

use std::sync::atomic::{AtomicBool, Ordering};

pub use lk_agents_voice::amd::classifier::{POSTPONE_TERMINATION_TOOL, SAVE_PREDICTION_TOOL};
use lk_agents_voice::amd::classifier::{classification_request, classification_verdicts};

const LOG: &str = "livekit.agents";
const TAP: &str = "amd";

/// The host's end of the system: `Start`, `Verdict::Settle`, `Close`.
#[derive(Clone)]
pub struct AmdHandle {
    tx: tokio::sync::mpsc::UnboundedSender<Msg>,

    listening: Arc<AtomicBool>,
}

impl AmdHandle {
    pub fn input(&self, input: In) {
        let _ = self.tx.send(Msg::Input(input));
    }

    fn detached() -> Self {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        AmdHandle { tx, listening: Arc::default() }
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
    SttOpened(Result<(SttSink, SttSource), ProviderError>),
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
    inner: Arc<dyn StreamSink<AudioFrame>>,

    open: Arc<AtomicBool>,
}

impl StreamSink<AudioFrame> for GatedSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if self.open.load(Ordering::Relaxed) { self.inner.push(item) } else { Ok(()) }
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.inner.flush()
    }

    fn end_input(&self) {
        self.inner.end_input();
    }
}

pub struct AmdSystem {
    peer: PeerHandle,

    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
    machine: Option<Amd>,
    timers: TimerSet<AmdTimer>,
    requests: u64,
    prompt: String,
    llm: Arc<dyn Llm>,
    tools_host: Option<HostRef>,
    inflight: Option<(u64, tokio::sync::watch::Sender<bool>)>,
    stt: SttLeg,
    /// The session asked to close (or the machine was closed): nothing more is read.
    closed: bool,

    /// The machine's [`Amd::listening`], for the host's `AMD.started`.
    listening: Arc<AtomicBool>,
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
        let _ = peer.send(Origin::Amd, IvrCommand::Close);
        let (room, participant) = (setup.room.clone(), setup.participant.clone());
        let system = AmdSystem::new(peer, setup, tx.clone());
        let listening = system.listening.clone();
        let spawner = system.peer.spawner.clone();
        let time = system.peer.time.clone();
        let callee_tx = tx.clone();
        spawner.spawn_unboxed(async move {
            tokio::select! {
                biased;
                () = callee_tx.closed() => {}
                callee = await_callee(room, participant, time) => {
                    let _ = callee_tx.send(Msg::Callee(callee));
                }
            }
        });
        spawner.spawn_unboxed(system.run(rx, msg_rx));
        AmdHandle { tx, listening }
    }

    fn new(peer: PeerHandle, setup: AmdSetup, tx: tokio::sync::mpsc::UnboundedSender<Msg>) -> Self {
        let stt = setup.stt.clone();
        // The machine holds the authorization from the attach.
        let (machine, out) = Amd::attached(setup.config);
        let mut system = AmdSystem {
            peer,
            tx,
            machine: Some(machine),
            timers: TimerSet::new(),
            requests: 0,
            prompt: setup.prompt,
            llm: setup.llm,
            tools_host: setup.tools_host,
            inflight: None,
            stt: SttLeg::None,
            closed: false,
            listening: Arc::default(),
        };
        let _ = system.peer.send(Origin::Amd, Command::AmdSpan(None));
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
        // The host may drop its handle: its channel then never yields, the feed still does.
        let mut host_open = true;
        while !self.closed {
            let deadline = self.timers.next_deadline();
            if deadline.is_some_and(|at| at <= time.now()) {
                self.on_timers(time.now());
                continue;
            }
            tokio::select! {
                biased;
                () = time.wait_until_opt(deadline) => self.on_timers(time.now()),
                item = rx.recv() => match item {
                    Some(item) => self.on_watched(time.now(), item),
                    None => break,
                },
                msg = async { if host_open { msgs.recv().await } else { std::future::pending().await } } => match msg {
                    Some(msg) => self.on_msg(time.now(), msg),
                    None => host_open = false,
                },
                () = api.closed() => break,
            }
        }
        // A tap that opened while the system was closing is still queued here: dropped with the
        // channel it would never be closed, and a dropped host stream is never cancelled.
        msgs.close();
        while let Ok(msg) = msgs.try_recv() {
            if let Msg::SttOpened(Ok((_sink, source))) = msg {
                self.peer.journal(Step::Input(InputStep::LegOpenDiscarded { leg: TAP }));
                self.discard_tap(source);
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
                if let Some(m) = metrics {
                    let metrics = Box::new(lk_agents_abi::metrics::base::AgentMetrics::Llm(*m));
                    self.peer.emit(SessionEvent::MetricsCollected { metrics, speech: None });
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

    fn perform(&mut self, now: Ts, out: Out) {
        for effect in out.effects {
            match effect {
                // Arming replaces, a cancel removes; both journalled.
                Effect::Arm(timer, at) => {
                    self.timers.arm(timer, at);
                    self.peer.journal(Step::Core(CoreStep::TimerArmed {
                        timer: TimerName::Amd(timer),
                        at,
                    }));
                }
                Effect::Cancel(timer) => {
                    self.timers.cancel(timer);
                    self.peer.journal(Step::Core(CoreStep::TimerCancelled(TimerName::Amd(timer))));
                }
                // The waiter is the host's (`execute()` awaits `amd_result`): nothing to wake here.
                // `AMD.emit("amd_prediction")`; `_on_amd_prediction` then cancels the classification.
                Effect::Prediction(p) => {
                    let _ = self.peer.send(Origin::Amd, Command::AmdSpan(Some(p.clone())));
                    self.peer.emit(SessionEvent::AmdResult { verdict: p });
                    self.step(now, In::EndInput);
                }
                Effect::RequestClassification(req) => self.request_classification(req),
                Effect::CancelClassification => self.cancel_classification(),
                // The postpone reply is discarded (`classifier._run`).
                Effect::Refused(refusal) => {
                    self.peer.journal(Step::Detector(DetectorStep::AmdRefused(refusal)))
                }
                // The scheduler's hold, as the system's command.
                Effect::Authorization(Authorization::Pause) => {
                    let _ = self.peer.send(Origin::Amd, SpeechCommand::PauseAuthorization);
                }
                Effect::Authorization(Authorization::Resume) => {
                    let _ = self.peer.send(Origin::Amd, SpeechCommand::ResumeAuthorization);
                }
                Effect::SkipReplies(skips) => {
                    let _ = self.peer.send(Origin::Amd, TurnCommand::SkipReplies(skips));
                }
                // `interrupt(force=True)`: the greeting is being spoken over.
                Effect::Interrupt { force } => {
                    let _ =
                        self.peer.send(Origin::Amd, SpeechCommand::Interrupt { force, call: None });
                }
            }
        }
    }

    /// One classification with `tool_choice="required"`. Debounce by cancel: a concurrent run
    /// would answer from a prefix of the greeting.
    fn request_classification(&mut self, req: ClassifyRequest) {
        self.cancel_classification();
        self.requests += 1;
        let n = self.requests;
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
        self.peer.spawner.clone().spawn_unboxed(classify(
            time,
            tx,
            llm,
            request,
            n,
            req.transcript,
            cancelled,
        ));
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
        let time = self.peer.time.clone();
        self.peer.spawner.clone().spawn_unboxed(async move {
            let outcome = match stt.stream(None, None).await {
                Ok(Some(pair)) => Ok(pair),
                Err(e) => Err(e),
                Ok(None) => Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                    "the AMD STT does not stream",
                ))),
            };
            // The system stopped reading while the stream opened: closed here, since a dropped
            // host stream is never cancelled.
            if let Err(tokio::sync::mpsc::error::SendError(Msg::SttOpened(Ok((_, source))))) =
                tx.send(Msg::SttOpened(outcome))
            {
                let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
                let _ = source.close(deadline).await;
            }
        });
    }

    /// The tap's stream landed: into the gated row, drained onto this channel. Unwanted: closed
    /// unused; refused: the timers decide.
    fn on_tap_opened(&mut self, outcome: Result<(SttSink, SttSource), ProviderError>) {
        if !matches!(self.stt, SttLeg::Opening) {
            if let Ok((_sink, source)) = outcome {
                self.peer.journal(Step::Input(InputStep::LegOpenDiscarded { leg: TAP }));
                self.discard_tap(source);
            }
            return;
        }
        let (sink, source) = match outcome {
            Ok(pair) => pair,
            Err(_) => {
                self.stt = SttLeg::Done;
                self.peer.journal(Step::Core(CoreStep::PortFailure(
                    "amd: the AMD STT could not be opened",
                )));
                return;
            }
        };
        let gate = Arc::new(AtomicBool::new(self.machine.as_ref().is_some_and(Amd::listening)));
        let sink: Arc<dyn StreamSink<AudioFrame>> =
            Arc::new(GatedSink { inner: Arc::from(sink), open: Arc::clone(&gate) });
        let tap = self.peer.install_audio_tap(sink);
        self.stt = SttLeg::Installed { gate, tap };
        self.peer.journal(Step::Input(InputStep::StreamOpened { leg: TAP }));
        spawn_tap_drain(self.peer.clone(), source, tap, self.tx.clone());
    }

    fn discard_tap(&self, source: SttSource) {
        let deadline = self.peer.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        self.peer.spawner.clone().spawn_unboxed(async move {
            let _ = source.close(deadline).await;
        });
    }

    fn on_tap_transcript(&mut self, now: Ts, ev: SpeechEvent) {
        if let Some(input) = tap_input(&ev) {
            self.step(now, input);
        }
    }

    /// The tap follows the machine's phase: frames while listening, the row cleared once a verdict
    /// is out or the machine closed.
    fn sync_tap(&mut self) {
        let over = matches!(self.machine, Some(Amd::Emitted(_) | Amd::Closed(_)));
        let removed = match &self.stt {
            SttLeg::Installed { gate, .. } if self.machine.as_ref().is_some_and(Amd::listening) => {
                gate.store(true, Ordering::Relaxed);
                None
            }
            SttLeg::Installed { .. } | SttLeg::Opening if over => {
                let tap =
                    if let SttLeg::Installed { tap, .. } = self.stt { Some(tap) } else { None };
                self.stt = SttLeg::Done;
                tap
            }
            _ => None,
        };
        if let Some(tap) = removed {
            self.peer.remove_audio_tap(tap);
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
    time: lk_agents_voice::agent_session::Timebase,
) -> Callee {
    let Some(room) = room else {
        log::warn!(target: LOG,
            "session room_io unavailable, starting amd timers immediately as fallback");
        return Callee::Answered;
    };
    let audio = Some(vec![lk_agents_abi::voice::rtc::TRACK_KIND_AUDIO]);
    let track = room.wait_for_track_publication(participant, audio, false, true);
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
    match room.wait_for_participant_attribute(identity, attribute, active).await {
        Ok(()) => Callee::Answered,
        Err(e) => Callee::Missing(e.to_string()),
    }
}

/// The tap's drain onto the system's channel: source closed once, tap removed if still installed.
fn spawn_tap_drain(
    peer: PeerHandle,
    mut source: SttSource,
    tap: lk_agents_abi::resources::ResourceId,

    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
) {
    let mut closing = peer.closing.clone();
    let time = peer.time.clone();
    peer.spawner.clone().spawn_unboxed(async move {
        let cause = loop {
            let next = tokio::select! {
                biased;
                item = source.next() => Some(item),
                _ = closing.wait_for(|c| *c) => None,
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
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let closed = source.close(deadline).await;
        peer.journal(Step::Input(InputStep::ChannelEnded { leg: TAP, cause }));
        peer.journal(Step::Input(InputStep::HostStreamClosed { label: TAP, how: closed }));
    });
}

/// The classification task. A cancelled run reports nothing. A native source may be abandoned
/// mid-poll; a host stream may not, so the cancel waits for the chunk.
async fn classify(
    time: lk_agents_voice::agent_session::Timebase,

    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
    llm: Arc<dyn Llm>,
    request: lk_agents_abi::ChatRequest,
    req: u64,
    transcript: String,
    mut cancelled: tokio::sync::watch::Receiver<bool>,
) {
    let started_at = time.wall_secs();
    let mut stream = match llm.chat(&request).await {
        Ok(s) => s,
        Err(_) => {
            let (calls, metrics) = (Vec::new(), None);
            let _ = tx.send(Msg::Answer { req, transcript, calls, failed: true, metrics });
            return;
        }
    };
    let mut fold = lk_agents_abi::telemetry::provider::LlmMetricState::for_host(started_at);
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
                fold.observe(&chunk, || time.wall_secs());
                if let Some(delta) = &chunk.delta {
                    calls.extend(delta.tool_calls.iter().cloned());
                }
            }
            Ok(None) => break,
            Err(_) => {
                failed = true;
                break;
            }
        }
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
        let context = lk_agents_abi::telemetry::provider::Context {
            timestamp: now,
            label,
            cancelled: false,
            metadata,
        };
        (now, context)
    });
    let metrics = metrics.map(Box::new);
    let _ = tx.send(Msg::Answer { req, transcript, calls, failed, metrics });
}
