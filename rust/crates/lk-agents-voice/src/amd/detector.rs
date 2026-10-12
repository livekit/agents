use crate::agent_session::cycle::TurnTimer;
use crate::agent_session::{IvrCommand, Origin, Peer, SpeechCommand, SystemGone, Watched};
use crate::amd::classifier::{
    Amd, AmdConfig, AmdTimer, Authorization, ClassifyRequest, Effect, Human, In, Out,
    TranscriptSource, Verdict,
};
use crate::events::SessionEvent;
use crate::trace::{CoreStep, DetectorLeg, DetectorStep, InputStep, LegEndCause, Step};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::amd::{AMD_PROMPT, AmdCategory};
use lk_agents_abi::{
    AudioFrame, ChatContext, ChatItem, ChatMessage, FunctionToolCall, HostRef, Llm, Role,
    SpawnerExt, SpeechEvent, SpeechEventType, StreamSink, Stt, SttSink, SttSource, TimerSet, Ts,
};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// `save_prediction(label: AMDCategory)` (`classifier.save_prediction`).
pub const SAVE_PREDICTION_TOOL: &str = "save_prediction";

/// `classifier.postpone_termination`, offered while extensions remain.
pub const POSTPONE_TERMINATION_TOOL: &str = "postpone_termination";

/// The leg table's peer row and the journal's name for it (`LegOpened{"amd"}`, `LegEnded{"amd"}`).
const TAP: &str = "amd";

fn save_prediction_spec() -> lk_agents_abi::ToolSpec {
    lk_agents_abi::ToolSpec {
        name: SAVE_PREDICTION_TOOL.to_owned(),
        description: "Save the prediction to the verdict.".to_owned(),
        json_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": AmdCategory::ALL.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
                },
            },
            "required": ["label"],
        }),
        flags: lk_agents_abi::ToolFlags::default(),
    }
}

fn postpone_termination_spec() -> lk_agents_abi::ToolSpec {
    lk_agents_abi::ToolSpec {
        name: POSTPONE_TERMINATION_TOOL.to_owned(),
        description: "Postpone the termination of the classification task.\n\
            Use when the transcript is ambiguous and more audio is expected.\n\n\
            Args:\n    seconds: Additional seconds to wait (max 10)."
            .to_owned(),
        json_schema: serde_json::json!({
            "type": "object",
            "properties": { "seconds": { "type": "number" } },
            "required": ["seconds"],
        }),
        flags: lk_agents_abi::ToolFlags::default(),
    }
}

pub struct AmdSetup {
    pub config: AmdConfig,
    /// `detection_options["prompt"]`, else the reference's [`AMD_PROMPT`].
    pub prompt: String,
    /// The classifier LLM: `AMD(llm=..)`, else the session's own (`detector.AMD._resolve_classifier`).
    pub llm: Arc<dyn Llm>,
    pub tools_host: Option<HostRef>,
    /// `AMD(stt=..)`: AMD's own recogniser; `None` reads the session's finals.
    pub stt: Option<Arc<dyn Stt>>,
}

impl AmdSetup {
    pub fn new(config: AmdConfig, llm: Arc<dyn Llm>) -> Self {
        AmdSetup { config, prompt: AMD_PROMPT.to_owned(), llm, tools_host: None, stt: None }
    }
}

impl fmt::Debug for AmdSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AmdSetup")
            .field("config", &self.config)
            .field("llm", &self.llm.label())
            .field("stt", &self.stt.as_ref().map(|s| s.label()))
            .finish()
    }
}

/// The host's end of the system: `Start`, `Verdict::Settle`, `Close`.
#[derive(Clone)]
pub struct AmdHandle {
    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
}

impl AmdHandle {
    pub fn input(&self, input: In) -> Result<(), SystemGone> {
        self.tx.send(Msg::Input(input)).map_err(|_| SystemGone)
    }

    /// `AMD.push_audio(frame)`: a frame for AMD's own STT, heard only while listening (a system
    /// without an STT of its own reads the session's finals and drops it).
    pub fn push_audio(&self, frame: AudioFrame) -> Result<(), SystemGone> {
        self.tx.send(Msg::Audio(frame)).map_err(|_| SystemGone)
    }

    /// `AMD.aclose()`: timers swept, request cancelled, hold released, tap cleared. Idempotent.
    pub fn close(&self) {
        let _ = self.tx.send(Msg::Input(In::Close));
    }

    fn detached() -> Self {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        AmdHandle { tx }
    }
}

impl fmt::Debug for AmdHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AmdHandle")
    }
}

enum Msg {
    Input(In),
    /// A host frame for the tap (`AMD.push_audio`).
    Audio(AudioFrame),
    /// Classification `req` ended with every tool call answered, in order, and the request's own
    /// transcript. `failed`: no stream or it errored; the timers decide.
    Answer {
        req: u64,
        transcript: String,
        calls: Vec<FunctionToolCall>,
        failed: bool,
    },
    /// AMD's own STT stream opened -- or refused -- off the loop (`_run_stt`'s `stt.stream()`).
    SttOpened(Result<(SttSink, SttSource), ProviderError>),
    /// One event off the tap's drain (`_receive`, `detector._receive`).
    Transcript(SpeechEvent),
}

enum SttLeg {
    /// No STT of AMD's own: the session's finals feed the machine.
    None,
    Opening,
    /// In the leg table's peer row behind `gate`: frames flow while the machine listens. `sink`
    /// is that row's, for the host's own frames.
    Installed { gate: Arc<AtomicBool>, sink: Arc<dyn StreamSink<AudioFrame>> },
    /// Cleared; an open that lands now is discarded.
    Done,
}

/// The tap's sink behind the listening gate (`AMD.push_audio`): pre-answer audio is never
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
    peer: Peer,
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
}

impl AmdSystem {
    /// `AMD._run`: holding authorization from now, subscribed to the feed, the tap opening off the
    /// loop. Past the close request nothing is attached and the handle answers `SystemGone`.
    pub fn spawn(peer: Peer, setup: AmdSetup) -> AmdHandle {
        if *peer.closing.borrow() {
            peer.journal(Step::Core(CoreStep::PortFailure("amd: attached after the close")));
            return AmdHandle::detached();
        }
        let rx = peer.subscribe();
        let (tx, msg_rx) = tokio::sync::mpsc::unbounded_channel();
        // "closing it so AMD can manage the IVR lifecycle": the session navigator is closed
        // before the hold; `execute()` on a machine-ivr verdict starts AMD's own.
        let _ = peer.send(Origin::Amd, IvrCommand::Close);
        let system = AmdSystem::new(peer, setup, tx.clone());
        let spawner = system.peer.spawner.clone();
        spawner.spawn_unboxed(system.run(rx, msg_rx));
        AmdHandle { tx }
    }

    fn new(peer: Peer, setup: AmdSetup, tx: tokio::sync::mpsc::UnboundedSender<Msg>) -> Self {
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
        };
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
        mut rx: tokio::sync::mpsc::UnboundedReceiver<Watched>,
        mut msgs: tokio::sync::mpsc::UnboundedReceiver<Msg>,
    ) {
        let time = self.peer.time.clone();
        let api = self.peer.api.clone();
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
    }

    /// One item off the feed: what the reference's `AudioRecognition` hands the classifier.
    pub fn on_watched(&mut self, now: Ts, item: Watched) {
        match item {
            Watched::Vad(ev) => match ev.kind {
                lk_agents_abi::VadEventType::StartOfSpeech => {
                    self.step(now, In::Human(Human::SpeechStarted))
                }
                lk_agents_abi::VadEventType::EndOfSpeech => {
                    let silence = std::time::Duration::try_from_secs_f64(ev.silence_duration)
                        .unwrap_or_default();
                    self.step(now, In::Human(Human::SpeechEnded { silence }));
                }
                lk_agents_abi::VadEventType::InferenceDone => {}
            },
            // The session's finals once each; the empty clearing final is no words.
            Watched::Stt(ev) => {
                if ev.kind == SpeechEventType::FinalTranscript
                    && let Some(text) = ev.best_alternative().map(|a| a.text.clone())
                    && !text.trim().is_empty()
                {
                    self.step(
                        now,
                        In::Human(Human::Transcript { text, source: TranscriptSource::SessionStt }),
                    );
                }
            }
            // `amd._on_end_of_turn`, before the commit verdict.
            Watched::Session(SessionEvent::UserTurnCompleted(_)) => {
                self.step(now, In::Human(Human::EndOfTurn))
            }
            Watched::Session(_) | Watched::Ivr(_) => {}
            // `self._amd.aclose()`; a verdict still owed is never released.
            Watched::CloseRequested => {
                self.step(now, In::Close);
                self.closed = true;
            }
        }
    }

    fn on_msg(&mut self, now: Ts, msg: Msg) {
        match msg {
            Msg::Input(input) => {
                let closing = matches!(input, In::Close);
                self.step(now, input);
                if closing {
                    self.closed = true;
                }
            }
            Msg::Answer { req, transcript, calls, failed } => {
                self.on_answer(now, req, transcript, calls, failed)
            }
            Msg::Audio(frame) => {
                // `detector.py:312`: only while listening, only into AMD's own STT.
                if let SttLeg::Installed { sink, .. } = &self.stt {
                    let _ = sink.push(frame);
                }
            }
            Msg::SttOpened(outcome) => self.on_tap_opened(outcome),
            Msg::Transcript(ev) => self.on_tap_transcript(now, ev),
        }
    }

    /// Fire the deadlines due at `now` in order; one cancelled by an earlier fire this pass is skipped.
    pub fn on_timers(&mut self, now: Ts) {
        while let Some(timer) = self.timers.pop_due(now) {
            self.peer.journal(Step::Core(CoreStep::TimerFired(TurnTimer::Amd(timer), now)));
            self.step(now, In::Timeout(timer));
        }
    }

    pub fn step(&mut self, now: Ts, input: In) {
        let Some(machine) = self.machine.take() else { return };
        let (next, out) = machine.step(now, input);
        self.machine = Some(next);
        self.perform(now, out);
        self.sync_tap();
    }

    fn perform(&mut self, now: Ts, out: Out) {
        for effect in out.effects {
            match effect {
                // Arming replaces, a cancel removes; both journalled.
                Effect::Arm(timer, at) => {
                    self.timers.arm(timer, at);
                    self.peer.journal(Step::Core(CoreStep::TimerArmed {
                        timer: TurnTimer::Amd(timer),
                        at,
                    }));
                }
                Effect::Cancel(timer) => {
                    self.timers.cancel(timer);
                    self.peer.journal(Step::Core(CoreStep::TimerCancelled(TurnTimer::Amd(timer))));
                }
                // The waiter is the host's (`execute()` awaits `amd_result`): nothing to wake here.
                // `AMD.emit("amd_prediction")`; `_on_amd_prediction` then cancels the classification.
                Effect::Prediction(p) => {
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
                // `interrupt(force=True)`: the greeting is being spoken over.
                Effect::Interrupt { force } => {
                    let _ =
                        self.peer.send(Origin::Amd, SpeechCommand::Interrupt { force, done: None });
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
        self.peer.journal(Step::Input(InputStep::LegRequested { leg: DetectorLeg::Amd, req: n }));
        let wall = self.peer.time.wall();
        let chat_ctx = ChatContext::new(vec![
            ChatItem::Message(ChatMessage::text(
                "amd_system",
                Role::System,
                self.prompt.clone(),
                wall,
            )),
            ChatItem::Message(ChatMessage::text(
                "amd_user",
                Role::User,
                req.transcript.clone(),
                wall,
            )),
        ]);
        let mut tools = vec![save_prediction_spec()];
        if req.allow_postpone {
            tools.push(postpone_termination_spec());
        }
        let request = lk_agents_abi::ChatRequest {
            chat_ctx,
            tools,
            tools_host: self.tools_host.clone(),
            // Without `"required"` the model answers in prose and AMD always times out.
            tool_choice: Some(lk_agents_abi::ToolChoice::Required),
            ..Default::default()
        };
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
                leg: DetectorLeg::Amd,
                recoverable: true,
            }));
            return;
        }
        self.peer.journal(Step::Input(InputStep::LegVerdict { leg: DetectorLeg::Amd, req }));
        for call in calls {
            let args: Option<serde_json::Value> = serde_json::from_str(&call.arguments).ok();
            match call.name.as_str() {
                SAVE_PREDICTION_TOOL => {
                    let Some(category) = args
                        .as_ref()
                        .and_then(|a| a.get("label"))
                        .and_then(|l| l.as_str())
                        .and_then(AmdCategory::from_wire)
                    else {
                        continue;
                    };
                    self.step(
                        now,
                        In::Verdict(Verdict::Llm { category, transcript: transcript.clone() }),
                    );
                }
                POSTPONE_TERMINATION_TOOL => {
                    let Some(seconds) =
                        args.as_ref().and_then(|a| a.get("seconds")).and_then(|s| s.as_f64())
                    else {
                        continue;
                    };
                    self.step(now, In::Verdict(Verdict::Postpone { seconds }));
                }
                _ => {}
            }
        }
    }

    /// `_run_stt`'s `stt.stream()`, off the loop; the stream comes back as `Msg::SttOpened`.
    fn open_tap(&self, stt: Arc<dyn Stt>) {
        stt.prewarm();
        let tx = self.tx.clone();
        self.peer.spawner.clone().spawn_unboxed(async move {
            let outcome = match stt.stream(None, None).await {
                Some(Ok(pair)) => Ok(pair),
                Some(Err(e)) => Err(e),
                None => Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                    "the AMD STT does not stream",
                ))),
            };
            let _ = tx.send(Msg::SttOpened(outcome));
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
        self.stt = SttLeg::Installed { gate, sink: Arc::clone(&sink) };
        let ticket = self.peer.install_audio_tap(sink);
        self.peer.journal(Step::Input(InputStep::LegOpened { leg: TAP }));
        spawn_tap_drain(self.peer.clone(), source, ticket, self.tx.clone());
    }

    fn discard_tap(&self, source: SttSource) {
        let deadline = self.peer.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        self.peer.spawner.clone().spawn_unboxed(async move {
            let _ = source.close(deadline).await;
        });
    }

    /// `_receive`: a non-empty final reaches the machine; everything else is dropped.
    fn on_tap_transcript(&mut self, now: Ts, ev: SpeechEvent) {
        if ev.kind != SpeechEventType::FinalTranscript {
            return;
        }
        let Some(text) = ev.best_alternative().map(|a| a.text.clone()).filter(|t| !t.is_empty())
        else {
            return;
        };
        self.step(now, In::Human(Human::Transcript { text, source: TranscriptSource::AmdStt }));
    }

    /// The tap follows the machine's phase: frames while listening, the row cleared once a verdict
    /// is out or the machine closed.
    fn sync_tap(&mut self) {
        let over = matches!(self.machine, Some(Amd::Emitted(_) | Amd::Closed(_)));
        let clear = match &self.stt {
            SttLeg::Installed { gate, .. } if self.machine.as_ref().is_some_and(Amd::listening) => {
                gate.store(true, Ordering::Relaxed);
                false
            }
            SttLeg::Installed { .. } | SttLeg::Opening if over => {
                let installed = matches!(self.stt, SttLeg::Installed { .. });
                self.stt = SttLeg::Done;
                installed
            }
            _ => false,
        };
        if clear {
            self.peer.clear_audio_tap();
        }
    }
}

/// The tap's drain onto the system's channel: source closed once, row forgotten if still installed.
fn spawn_tap_drain(
    peer: Peer,
    mut source: SttSource,
    ticket: crate::audio::Ticket,
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
                        break LegEndCause::Closing;
                    }
                }
                Some(Ok(None)) => break LegEndCause::SourceEnded,
                Some(Err(_)) => break LegEndCause::Error,
                None => break LegEndCause::Closing,
            }
        };
        peer.audio_tap_ended(ticket);
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let closed = source.close(deadline).await;
        peer.journal(Step::Input(InputStep::LegEnded { leg: TAP, cause }));
        peer.journal(Step::Input(InputStep::HostStreamClosed { label: TAP, how: closed }));
    });
}

/// The classification task. A cancelled run reports nothing. A native source may be abandoned
/// mid-poll; a host stream may not, so the cancel waits for the chunk.
async fn classify(
    time: crate::agent_session::Timebase,
    tx: tokio::sync::mpsc::UnboundedSender<Msg>,
    llm: Arc<dyn Llm>,
    request: lk_agents_abi::ChatRequest,
    req: u64,
    transcript: String,
    mut cancelled: tokio::sync::watch::Receiver<bool>,
) {
    let mut stream = match llm.chat(&request).await {
        Ok(s) => s,
        Err(_) => {
            let _ = tx.send(Msg::Answer { req, transcript, calls: Vec::new(), failed: true });
            return;
        }
    };
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
    let _ = tx.send(Msg::Answer { req, transcript, calls, failed });
}
