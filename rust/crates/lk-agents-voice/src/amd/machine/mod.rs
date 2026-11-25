//! The AMD machine: one run's classification, stage and reply policy (`voice/amd/detector.py`),
//! with no I/O.
//!
//! - **States**: [`Lifecycle`] goes `Pending` (attached, not listening) → `Active` (listening)
//!   → `Finished` (completed). Inside `Active`: the [stage](stage), the latest category, the
//!   likely-human mark, counters, the turn registry, at most one classifier request, the
//!   late-final wait, the classifier history and the voicemail record.
//! - **Inputs**: [`In`], each with the time it is handled at. Observations carry the time they
//!   describe, so a commit dates its turn by when it happened, not when AMD heard of it.
//! - **Effects**: [`AmdEffect`]s: answers to the session's reply gate, classifier and menu
//!   requests, deadlines, and public events.
//! - **Terminal**: completion sends `Detach` in the same transition. `amd_completed` is emitted
//!   once the session reports `Detached` and the runtime reports the legs closed.
//!
//! Three things a turn can own are tracked apart (spec §6.2, §7.3, §7.4): the request belongs to
//! the newest turn with text, the late-final wait to the newest committed turn, and the right to
//! reply to the latest turn. An empty turn that takes the right to reply from an older turn whose
//! request is pending gets that request's result as a `reused` prediction.

pub mod stage;

mod history;
mod reply;

pub use self::history::HISTORY_LIMIT;

use self::history::{ClassifierHistory, Source, Transcripts};
use self::stage::StageEffect;
use super::effects::{AmdEffect, AmdTimer, ClassifyRequest, PreviousPrediction, ReqId};
use super::effects::{MenuRequest, TurnTranscript};
use super::events::{AmdEvent, AmdReason, Category, IvrMenuOption, MenuObserved, Outcome};
use super::events::{Prediction, Stage};
use super::options::{AmdOptions, AmdOptionsError};
use super::{AmdAnswer, AmdObservation, AmdRunId, DetachCancel, UserSpeechObservation};
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::audio_recognition::TurnId;
use lk_agents_abi::Ts;
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};

/// AMD's STT finalizes a sentence a little after the session commits it: a turn waits this long
/// for its AMD final (`_LATE_AMD_FINAL_WAIT`).
pub const LATE_AMD_FINAL_WAIT: Duration = Duration::from_millis(300);
/// Menu extraction stops before it goes stale (`_MENU_TIMEOUT`).
pub const MENU_TIMEOUT: Duration = Duration::from_secs(5);
/// At most this many options in an observed menu.
pub const MAX_MENU_OPTIONS: usize = 20;

/// What the run knows of its session and legs at attach.
#[derive(Debug, Clone, PartialEq)]
pub struct AmdSetup {
    /// The agent running at attach. A turn committed under another completes AMD with
    /// `agent_changed`.
    pub agent: String,
    /// `AMDMenuObservedEvent.session_id`, drawn by the runtime.
    pub session_id: String,
    /// When AMD was entered, on the clock tool calls are stamped with: earlier calls stay out of
    /// the classifier history.
    pub entered_at: SystemTime,
    /// The session has an STT.
    pub session_stt: bool,
    /// AMD has its own streaming STT.
    pub amd_stt: bool,
    /// The classifier is an LLM, which can extract menus; a decision model cannot.
    pub extract_menus: bool,
    /// A realtime model without user transcription and without session STT: AMD inserts each
    /// turn's transcript (spec §7.2).
    pub insert_user_items: bool,
}

/// What the machine is told.
#[derive(Debug, Clone, PartialEq)]
pub enum In {
    /// An observation from the session's reply gate.
    Observed(AmdObservation),
    /// The classifier answered `req`. Its death is `LegFailed`.
    Classified { req: ReqId, result: Result<Classification, ClassifyError> },
    /// The menu leg answered `req`; `None` when extraction failed. A failure is silent.
    MenuExtracted { req: ReqId, menu: Option<Menu> },
    /// A final from AMD's own STT.
    AmdFinal(String),
    /// AMD's STT stream failed or would not open (spec §6.3).
    SttFailed,
    /// A leg, or a prediction or menu listener in the host, failed: `internal_error`.
    LegFailed,
    /// The callee leg's progress (spec §5.2).
    Callee(Callee),
    /// A deadline armed with `SetDeadline` passed.
    Timeout(AmdTimer),
    /// The runtime closed AMD's legs after `Detach`.
    LegsClosed,
    /// `aclose()`: complete with `cancelled` if still running.
    Close,
}

/// A classifier answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    pub category: Category,
    /// Model time, as the leg measured it around the model call: Python times `classify`
    /// alone, after the late-final wait.
    pub inference_duration: Duration,
}

/// A classification that gives no category: an `inference_error` prediction (spec §8.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifyError {
    /// The provider refused or failed (`APIError`).
    Api,
    /// No single valid `record_result` call, or no decision.
    InvalidOutput,
    /// The connection options' timeout.
    Timeout,
}

/// An extracted menu.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Menu {
    pub menu: String,
    pub options: Vec<IvrMenuOption>,
}

/// The callee leg's progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Callee {
    /// The track is there and, with `wait_until_answered`, the call is answered: listen.
    Answered,
    /// No participant or audio track within the setup wait.
    Missing,
    /// The participant or the room disconnected.
    Disconnected,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lifecycle {
    /// Attached; the session discards participant audio until `Activated`.
    Pending,
    Active,
    /// Completed. The outcome cannot change after completion.
    Finished(Outcome),
}

/// Completion waits on `Detached` and `LegsClosed`, in either order, and emits once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Teardown {
    Running,
    AwaitingBoth,
    AwaitingDetached,
    AwaitingLegs,
    Done,
}

/// One committed turn.
#[derive(Debug)]
struct Turn {
    /// AMD's turn id, from 1.
    id: u64,
    committed_at: Ts,
    transcript: TurnTranscript,
    speech_duration: Duration,
    agent_changed: bool,
    /// At most one; a reused one is never revised.
    prediction: Option<Prediction>,
    /// An empty turn waiting for its AMD final before it can classify.
    awaiting_final: bool,
    /// Its reply decision was sent.
    decided: bool,
    /// Its reply shows its words in the classifier history; fixed at its decision.
    reply_words: bool,
}

/// The pending classifier request.
#[derive(Debug)]
struct Request {
    id: ReqId,
    turn: TurnId,
    /// `inference_timeout` from its start.
    deadline: Ts,
    state: RequestState,
}

#[derive(Debug)]
enum RequestState {
    /// Built at commit; sent once the turn's AMD final arrives or its wait passes.
    AwaitingFinal(ClassifyRequest),
    Issued,
}

/// The newest committed turn's claim on the next AMD final.
#[derive(Debug, Clone, Copy)]
struct LateFinalWait {
    turn: TurnId,
    until: Ts,
}

/// The menu extraction in flight.
#[derive(Debug, Clone, Copy)]
struct MenuWait {
    req: ReqId,
    turn_id: u64,
    started_at: Ts,
}

/// A speech AMD watches for the idle timer, the classifier history and the voicemail record.
#[derive(Debug, Clone, Copy)]
struct WatchedSpeech {
    /// The classifier sees its words, not `(the agent spoke)`.
    words: bool,
    add_to_chat_ctx: bool,
    /// It was playing at attach: its words started before AMD's entry.
    before_entry: bool,
    /// Its first playout, where its reply goes in the history.
    started_at: Option<Ts>,
}

/// The voicemail message (spec §10.4).
#[derive(Debug, Default)]
struct VoicemailRecord {
    /// The turn allowed the message, until its reply exists or a newer turn commits.
    reserved: Option<TurnId>,
    /// The message's speech, until it ends.
    attempt: Option<VoicemailAttempt>,
    played: bool,
}

#[derive(Debug, Clone, Copy)]
struct VoicemailAttempt {
    turn: TurnId,
    speech: SpeechId,
    played_out: bool,
    item_interrupted: bool,
}

/// Participant speech as AMD's turns measure it (spec §7.1).
#[derive(Debug, Default)]
struct SpeechClock {
    speaking: bool,
    started_at: Option<Ts>,
    /// Speech since the last commit, not counting the open stretch.
    duration: Duration,
}

/// One AMD run. See the module documentation.
#[derive(Debug)]
pub struct Amd {
    run: AmdRunId,
    options: AmdOptions,
    setup: AmdSetup,
    lifecycle: Lifecycle,
    teardown: Teardown,
    stage: Stage,
    /// For the completed category (spec §12.2).
    previous_stage: Option<Stage>,
    /// The latest accepted category; `wait` denies replies.
    category: Category,
    /// The latest classified prediction; reused predictions do not change it.
    last_prediction: Option<Prediction>,
    likely_human: bool,
    /// Set when a machine stage is accepted, never cleared.
    had_machine_stage: bool,
    turns: BTreeMap<TurnId, Turn>,
    latest: Option<TurnId>,
    last_turn_id: u64,
    request: Option<Request>,
    late_final: Option<LateFinalWait>,
    transcripts: Transcripts,
    history: ClassifierHistory,
    speeches: BTreeMap<SpeechId, WatchedSpeech>,
    voicemail: VoicemailRecord,
    menu: Option<MenuWait>,
    uncertain: u32,
    timeouts: u32,
    speech: SpeechClock,
    agent_busy: bool,
    /// The turn whose allowed reply waits for silence, until the session releases it.
    held: Option<TurnId>,
    activated_at: Option<Ts>,
    overall_deadline: Option<Ts>,
    idle_deadline: Option<Ts>,
    last_req: u64,
    effects: Vec<AmdEffect>,
}

impl Amd {
    /// A run attached to its session, not listening yet.
    pub fn new(
        run: AmdRunId,
        options: AmdOptions,
        setup: AmdSetup,
    ) -> Result<Amd, AmdOptionsError> {
        options.validate()?;
        let transcripts = Transcripts::new(setup.session_stt, setup.amd_stt);
        Ok(Amd {
            run,
            options,
            setup,
            lifecycle: Lifecycle::Pending,
            teardown: Teardown::Running,
            stage: Stage::Uncertain,
            previous_stage: None,
            category: Category::Uncertain,
            last_prediction: None,
            likely_human: false,
            had_machine_stage: false,
            turns: BTreeMap::new(),
            latest: None,
            last_turn_id: 0,
            request: None,
            late_final: None,
            transcripts,
            history: ClassifierHistory::default(),
            speeches: BTreeMap::new(),
            voicemail: VoicemailRecord::default(),
            menu: None,
            uncertain: 0,
            timeouts: 0,
            speech: SpeechClock::default(),
            agent_busy: false,
            held: None,
            activated_at: None,
            overall_deadline: None,
            idle_deadline: None,
            last_req: 0,
            effects: Vec::new(),
        })
    }

    pub fn run(&self) -> AmdRunId {
        self.run
    }

    pub fn lifecycle(&self) -> &Lifecycle {
        &self.lifecycle
    }

    /// Handles one input at `now` and returns its effects, in order.
    pub fn step(&mut self, now: Ts, input: In) -> Vec<AmdEffect> {
        // The overall limit wins over whatever arrives at or after it (`_check_hard_timeout`).
        if self.active() && self.overall_deadline.is_some_and(|at| now >= at) {
            self.finish(AmdReason::Timeout);
        }
        match input {
            In::Observed(observation) => self.observe(now, observation),
            In::Classified { req, result } => self.on_classified(now, req, result),
            In::MenuExtracted { req, menu } => self.on_menu(now, req, menu),
            In::AmdFinal(text) => self.on_amd_final(now, &text),
            In::SttFailed => self.on_stt_failed(now),
            In::LegFailed => self.finish(AmdReason::InternalError),
            In::Callee(callee) => self.on_callee(now, callee),
            In::Timeout(timer) => self.on_timeout(now, timer),
            In::LegsClosed => self.on_legs_closed(),
            In::Close => self.finish(AmdReason::Cancelled),
        }
        self.update_idle(now);
        std::mem::take(&mut self.effects)
    }

    fn active(&self) -> bool {
        self.lifecycle == Lifecycle::Active
    }

    fn finished(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Finished(_))
    }

    fn push(&mut self, effect: AmdEffect) {
        self.effects.push(effect);
    }

    fn answer(&mut self, answer: AmdAnswer) {
        self.push(AmdEffect::Answer(answer));
    }

    fn next_req(&mut self) -> ReqId {
        self.last_req += 1;
        ReqId(self.last_req)
    }

    fn observe(&mut self, now: Ts, observation: AmdObservation) {
        match observation {
            AmdObservation::TurnCommitted { turn, transcript, speaking, agent, at, .. } => {
                self.on_commit(now, turn, &transcript, speaking, &agent, at);
            }
            AmdObservation::TurnEnded { turn, .. } => {
                self.held.take_if(|held| *held == turn);
            }
            AmdObservation::ReplyCreated { turn, speech } => self.on_reply_created(turn, speech),
            AmdObservation::Speech { speech, event, at } => self.on_speech(speech, event, at),
            AmdObservation::AgentBusy { busy, .. } => self.agent_busy = busy,
            AmdObservation::SilenceHold { turn, held: false, .. } => {
                self.held.take_if(|held| *held == turn);
            }
            AmdObservation::SilenceHold { held: true, .. } => {}
            AmdObservation::UserSpeech(speech) => self.on_user_speech(speech),
            AmdObservation::ToolsExecuted { calls } => {
                if self.finished() {
                    return;
                }
                for record in calls {
                    let Some(output) = &record.output else { continue };
                    if record.call.created_at >= self.setup.entered_at {
                        self.history.add_dtmf(now, &record.call, output);
                    }
                }
            }
            AmdObservation::Detached { run } if run == self.run => self.on_detached(),
            AmdObservation::Detached { .. } => {}
        }
    }

    // Listening.

    fn on_callee(&mut self, now: Ts, callee: Callee) {
        match callee {
            Callee::Answered if self.lifecycle == Lifecycle::Pending => {
                self.lifecycle = Lifecycle::Active;
                self.activated_at = Some(now);
                let deadline = now.saturating_add(self.options.timeout);
                self.overall_deadline = Some(deadline);
                self.push(AmdEffect::SetDeadline(AmdTimer::Overall, deadline));
                let opening = self.options.opening();
                self.answer(AmdAnswer::Activated { run: self.run, opening });
            }
            Callee::Answered => {}
            Callee::Missing => self.finish(AmdReason::ParticipantMissing),
            Callee::Disconnected => self.finish(AmdReason::ParticipantDisconnected),
        }
    }

    /// The gate reports participant speech only once it is active, after this run activated. A
    /// turn's speech duration counts from activation.
    fn on_user_speech(&mut self, speech: UserSpeechObservation) {
        if !self.active() {
            return;
        }
        match speech {
            UserSpeechObservation::Started { at } => {
                self.speech.speaking = true;
                self.speech.started_at = Some(self.activated_at.map_or(at, |a| at.max(a)));
            }
            UserSpeechObservation::Ended { at } => {
                self.speech.speaking = false;
                if let Some(started) = self.speech.started_at.take() {
                    self.speech.duration += at.saturating_duration_since(started);
                }
            }
            UserSpeechObservation::Final { text, .. } => {
                self.transcripts.push(Source::Session, &text, false);
            }
        }
    }

    // Turns.

    fn on_commit(
        &mut self,
        now: Ts,
        turn: TurnId,
        committed: &str,
        speaking: bool,
        agent: &str,
        at: Ts,
    ) {
        match self.lifecycle {
            // Not bound to an AMD turn: the first row of the reply table.
            Lifecycle::Pending => {
                let decision = self.unbound_decision();
                self.answer(AmdAnswer::ReplyDecided { run: self.run, turn, decision });
                return;
            }
            // `Detach` settles it.
            Lifecycle::Finished(_) => return,
            Lifecycle::Active => {}
        }
        // The session cancelled the held reply when it committed this turn.
        self.held = None;
        let (transcript, late) = self.transcripts.end_turn(committed);
        let mut speech_duration = std::mem::take(&mut self.speech.duration);
        if speaking && let Some(started) = self.speech.started_at {
            speech_duration += at.saturating_duration_since(started);
            self.speech.started_at = Some(at);
        }
        self.last_turn_id += 1;
        let has_text = !transcript.transcript().is_empty();
        self.turns.insert(
            turn,
            Turn {
                id: self.last_turn_id,
                committed_at: at,
                transcript,
                speech_duration,
                agent_changed: agent != self.setup.agent,
                prediction: None,
                awaiting_final: !has_text && late,
                decided: false,
                reply_words: false,
            },
        );
        self.latest = Some(turn);
        // The voicemail reservation belongs to the turn that was allowed the message.
        self.voicemail.reserved = None;
        // The newest commit takes over the wait; an older turn's own wait still runs out.
        self.late_final = None;
        if late {
            let until = at.saturating_add(LATE_AMD_FINAL_WAIT);
            self.late_final = Some(LateFinalWait { turn, until });
            self.push(AmdEffect::SetDeadline(AmdTimer::LateFinal(turn), until));
        }
        if has_text {
            self.start_classification(at, turn, late);
            self.insert_user_item(turn);
        } else if !late && self.request.is_none() {
            self.record_prediction(now, turn, AmdReason::Reused, None, None);
        }
    }

    fn insert_user_item(&mut self, turn: TurnId) {
        let Some(t) = self.turns.get(&turn) else { return };
        let text = t.transcript.transcript();
        if self.setup.insert_user_items && !text.is_empty() {
            let text = text.to_owned();
            self.answer(AmdAnswer::InsertUserItem { run: self.run, turn, text });
        }
    }

    // Late AMD finals.

    fn on_amd_final(&mut self, now: Ts, text: &str) {
        let text = text.trim();
        // A blank final does not consume a turn's wait.
        if text.is_empty() || !self.active() || !self.transcripts.amd_active() {
            return;
        }
        match self.late_final.take() {
            Some(wait) if now <= wait.until => {
                self.push(AmdEffect::ClearDeadline(AmdTimer::LateFinal(wait.turn)));
                self.resolve_final(now, wait.turn, Some(text));
            }
            _ => self.transcripts.push(Source::Amd, text, false),
        }
    }

    fn on_stt_failed(&mut self, now: Ts) {
        if !self.transcripts.fail() {
            return;
        }
        // No final will come: every turn still waiting for one goes on without it, oldest first.
        self.late_final = None;
        let mut waiting: Vec<(u64, TurnId)> = (self.turns.iter())
            .filter(|(_, t)| t.awaiting_final)
            .map(|(turn, t)| (t.id, *turn))
            .collect();
        if let Some(request) = &self.request
            && matches!(request.state, RequestState::AwaitingFinal(_))
            && let Some(t) = self.turns.get(&request.turn)
        {
            waiting.push((t.id, request.turn));
        }
        waiting.sort();
        for (_, turn) in waiting {
            self.push(AmdEffect::ClearDeadline(AmdTimer::LateFinal(turn)));
            self.resolve_final(now, turn, None);
        }
    }

    /// A turn's wait ended, with its AMD final or without one.
    fn resolve_final(&mut self, now: Ts, turn: TurnId, text: Option<&str>) {
        if let Some(request) = &self.request
            && request.turn == turn
            && matches!(request.state, RequestState::AwaitingFinal(_))
        {
            return self.issue_request(text);
        }
        if self.turns.get(&turn).is_some_and(|t| t.awaiting_final) {
            self.resolve_heard(now, turn, text);
        }
    }

    /// `_classify` after the wait: the final replaces the turn's message, then the model runs.
    fn issue_request(&mut self, text: Option<&str>) {
        let Some(request) = self.request.as_mut() else { return };
        let state = std::mem::replace(&mut request.state, RequestState::Issued);
        let RequestState::AwaitingFinal(mut classify) = state else {
            request.state = state;
            return;
        };
        let turn = request.turn;
        if let (Some(text), Some(t)) = (text, self.turns.get_mut(&turn)) {
            t.transcript.amd = Some(text.to_owned());
            self.history.update_user(t.id, &t.transcript);
            history::update_user(classify.history.iter_mut(), t.id, &t.transcript);
        }
        self.push(AmdEffect::Classify(classify));
    }

    /// `_classify_when_heard`: an empty turn's wait ended.
    fn resolve_heard(&mut self, now: Ts, turn: TurnId, text: Option<&str>) {
        let latest = self.latest == Some(turn);
        let active = self.active();
        let pending = self.request.is_some();
        let Some(t) = self.turns.get_mut(&turn) else { return };
        t.awaiting_final = false;
        if !active {
            return;
        }
        let Some(text) = text else {
            // No text: the empty-turn rules, unless a newer turn owns the reply.
            if latest && t.prediction.is_none() && !pending {
                self.record_prediction(now, turn, AmdReason::Reused, None, None);
            }
            return;
        };
        t.transcript.amd = Some(text.to_owned());
        if latest && t.prediction.is_none() {
            self.start_classification(now, turn, false);
        } else {
            // A newer turn owns the request and the reply, or a reused prediction is kept: the
            // next request covers this text.
            let (id, transcript) = (t.id, t.transcript.clone());
            self.history.add_user(now, id, &transcript);
        }
        self.insert_user_item(turn);
    }

    // Classification.

    /// `_start_classification`: the turn's message joins the history and its request starts at
    /// `at` (its deadline counts from there), held back until its AMD final if `late`.
    fn start_classification(&mut self, at: Ts, turn: TurnId, late: bool) {
        let Some(t) = self.turns.get(&turn) else { return };
        let (turn_id, transcript, speech_duration) =
            (t.id, t.transcript.clone(), t.speech_duration);
        self.history.add_user(at, turn_id, &transcript);
        self.cancel_classification();
        // Only a newer turn with text cancels the menu extraction.
        self.cancel_menu();
        let req = self.next_req();
        let classify = ClassifyRequest {
            req,
            turn,
            turn_id,
            stage: self.stage,
            previous_prediction: (self.last_prediction.as_ref()).map(|p| PreviousPrediction {
                turn_id: p.turn_id,
                category: p.category,
                reason: p.reason,
            }),
            likely_human: self.likely_human,
            recommended: stage::recommended(self.stage),
            speech_duration,
            history: self.history.window(),
        };
        let deadline = at.saturating_add(self.options.inference_timeout);
        self.push(AmdEffect::SetDeadline(AmdTimer::Inference(req), deadline));
        let state = if late {
            RequestState::AwaitingFinal(classify)
        } else {
            self.push(AmdEffect::Classify(classify));
            RequestState::Issued
        };
        self.request = Some(Request { id: req, turn, deadline, state });
    }

    /// Removes the pending request and its deadlines. The caller says whether a leg still runs.
    fn take_request(&mut self) -> Option<Request> {
        let request = self.request.take()?;
        self.push(AmdEffect::ClearDeadline(AmdTimer::Inference(request.id)));
        if matches!(request.state, RequestState::AwaitingFinal(_)) {
            // Its final, if one comes, goes to the open turn.
            self.late_final.take_if(|wait| wait.turn == request.turn);
            self.push(AmdEffect::ClearDeadline(AmdTimer::LateFinal(request.turn)));
        }
        Some(request)
    }

    /// `_cancel_classification`: the pending request is stale.
    fn cancel_classification(&mut self) -> Option<Request> {
        let request = self.take_request()?;
        if matches!(request.state, RequestState::Issued) {
            self.push(AmdEffect::CancelClassify);
        }
        Some(request)
    }

    fn on_classified(
        &mut self,
        now: Ts,
        req: ReqId,
        result: Result<Classification, ClassifyError>,
    ) {
        let Some(request) = self.request.as_ref().filter(|r| r.id == req && self.active()) else {
            return;
        };
        let RequestState::Issued = request.state else { return };
        // A result after its deadline counts as a timeout.
        let late = now >= request.deadline;
        let Some(request) = self.take_request() else { return };
        if late {
            return self.timeout_inference(now, request);
        }
        match result {
            Err(_) => {
                self.forget_category();
                self.record_prediction(now, request.turn, AmdReason::InferenceError, None, None);
            }
            Ok(Classification { category, inference_duration }) => {
                self.accept_prediction(now, request.turn, category, inference_duration);
            }
        }
    }

    /// `_timeout_inference`.
    fn timeout_inference(&mut self, now: Ts, request: Request) {
        self.timeouts += 1;
        self.forget_category();
        self.record_prediction(now, request.turn, AmdReason::InferenceTimeout, None, None);
    }

    /// `_forget_category`: a turn without a prediction is unknown, not a repeat of the previous
    /// one. A carried-over `wait` would deny every reply, even to a person who picks up.
    fn forget_category(&mut self) {
        self.category = Category::Uncertain;
    }

    /// `_accept_prediction`: the likely-human downgrade, the stage machine, the counters.
    fn accept_prediction(&mut self, now: Ts, turn: TurnId, answer: Category, took: Duration) {
        let mut category = answer;
        if category == Category::Human
            && stage::is_likely_human_stage(self.stage)
            && !self.likely_human
        {
            self.likely_human = true;
            category = Category::Uncertain;
        } else if category != Category::Uncertain {
            self.likely_human = false;
        }
        // Terminal stages complete AMD, so a request never runs in one.
        let Some(transition) = stage::next(self.stage, category) else { return };
        let mut effect = transition.effect;
        if self.options.returns_early(transition.next_stage) {
            // The application handles this stage: no reply and no menu extraction.
            effect = Some(StageEffect::Complete);
        }
        if transition.next_stage != self.stage {
            self.previous_stage = Some(self.stage);
            self.clear_idle();
        }
        self.stage = transition.next_stage;
        self.category = category;
        self.had_machine_stage |= self.stage.is_machine();
        self.timeouts = 0;
        self.uncertain = if category == Category::Uncertain { self.uncertain + 1 } else { 0 };
        self.record_prediction(now, turn, AmdReason::Prediction, Some(took), effect);
    }

    /// `_record_prediction`: the turn's prediction, a `reused` copy for each later turn still
    /// waiting, the stage's effects, then the events and the latest turn's decision.
    fn record_prediction(
        &mut self,
        now: Ts,
        turn: TurnId,
        reason: AmdReason,
        inference: Option<Duration>,
        effect: Option<StageEffect>,
    ) {
        let (category, stage) = (self.category, self.stage);
        let Some(t) = self.turns.get_mut(&turn) else { return };
        let event = Prediction {
            turn_id: t.id,
            category,
            stage,
            reason,
            transcript: t.transcript.transcript().to_owned(),
            speech_duration: t.speech_duration,
            delay: now.saturating_duration_since(t.committed_at),
            inference_duration: if reason == AmdReason::Prediction { inference } else { None },
        };
        t.prediction = Some(event.clone());
        let (turn_id, transcript) = (t.id, t.transcript.transcript().to_owned());
        if reason != AmdReason::Reused {
            // Reused events repeat this one, so completion keeps the classified turn.
            self.last_prediction = Some(event.clone());
        }
        let mut events = vec![event.clone()];
        // Empty turns committed during inference reuse its result, each with its own event. A
        // turn with text would have cancelled this request.
        for later in self.turns.values_mut().filter(|t| t.id > turn_id && t.prediction.is_none()) {
            let reused = Prediction {
                turn_id: later.id,
                reason: AmdReason::Reused,
                transcript: later.transcript.transcript().to_owned(),
                speech_duration: later.speech_duration,
                delay: now.saturating_duration_since(later.committed_at),
                inference_duration: None,
                ..event.clone()
            };
            later.prediction = Some(reused.clone());
            events.push(reused);
        }
        match effect {
            Some(StageEffect::Complete) => self.finish(AmdReason::Finished),
            // A decision model only chooses a category; it cannot read a menu.
            Some(StageEffect::ExtractMenu) if self.setup.extract_menus => {
                self.start_menu(now, turn_id, transcript);
            }
            Some(StageEffect::ExtractMenu) | None => {}
        }
        if self.uncertain >= self.options.max_uncertain_turns {
            self.finish(AmdReason::MaxUncertainTurns);
        }
        for event in events {
            self.push(AmdEffect::Emit(AmdEvent::Prediction(event)));
        }
        self.decide();
    }

    // Menus.

    fn start_menu(&mut self, now: Ts, turn_id: u64, transcript: String) {
        self.cancel_menu();
        let req = self.next_req();
        self.menu = Some(MenuWait { req, turn_id, started_at: now });
        self.push(AmdEffect::ExtractMenu(MenuRequest { req, turn_id, transcript }));
        self.push(AmdEffect::SetDeadline(AmdTimer::Menu(req), now.saturating_add(MENU_TIMEOUT)));
    }

    fn cancel_menu(&mut self) {
        if let Some(menu) = self.menu.take() {
            self.push(AmdEffect::ClearDeadline(AmdTimer::Menu(menu.req)));
        }
    }

    fn on_menu(&mut self, now: Ts, req: ReqId, menu: Option<Menu>) {
        if self.finished() {
            return;
        }
        let Some(wait) = self.menu.take_if(|wait| wait.req == req) else { return };
        self.push(AmdEffect::ClearDeadline(AmdTimer::Menu(req)));
        let Some(Menu { menu, mut options }) = menu else { return };
        if menu.is_empty() && options.is_empty() {
            return;
        }
        options.truncate(MAX_MENU_OPTIONS);
        self.push(AmdEffect::Emit(AmdEvent::MenuObserved(MenuObserved {
            session_id: self.setup.session_id.clone(),
            turn_id: wait.turn_id,
            menu,
            options,
            extraction_duration: now.saturating_duration_since(wait.started_at),
        })));
    }

    // Deadlines.

    fn on_timeout(&mut self, now: Ts, timer: AmdTimer) {
        match timer {
            AmdTimer::Overall => {
                if self.active() && self.overall_deadline.is_some_and(|at| now >= at) {
                    self.finish(AmdReason::Timeout);
                }
            }
            AmdTimer::Idle => {
                if self.active() && self.idle_deadline.is_some_and(|at| now >= at) {
                    self.finish(AmdReason::IdleTimeout);
                }
            }
            AmdTimer::Inference(req) => {
                let due = (self.request.as_ref()).is_some_and(|r| r.id == req && now >= r.deadline);
                if self.active()
                    && due
                    && let Some(request) = self.cancel_classification()
                {
                    self.timeout_inference(now, request);
                }
            }
            AmdTimer::LateFinal(turn) => {
                self.late_final.take_if(|wait| wait.turn == turn);
                self.resolve_final(now, turn, None);
            }
            AmdTimer::Menu(req) => {
                // Silent: a late menu is dropped.
                self.menu.take_if(|wait| wait.req == req);
            }
        }
    }

    fn clear_idle(&mut self) {
        if self.idle_deadline.take().is_some() {
            self.push(AmdEffect::ClearDeadline(AmdTimer::Idle));
        }
    }

    // Completion.

    /// `_finish`: completes the run in this transition and detaches the gate. Only the first
    /// reason counts.
    fn finish(&mut self, reason: AmdReason) {
        if self.finished() {
            return;
        }
        let mut category = self.stage;
        if self.voicemail.played
            && category == Stage::MachineUnavailable
            && let Some(previous) = self.previous_stage
        {
            // The box's goodbye after the message ends the box, not an unreachable call.
            category = previous;
        }
        let outcome = Outcome {
            category,
            reason,
            voicemail_message_played: self.voicemail.played,
            last_prediction: self.last_prediction.clone(),
        };
        self.lifecycle = Lifecycle::Finished(outcome);
        self.teardown = Teardown::AwaitingBoth;
        self.cancel_classification();
        self.cancel_menu();
        let mut waits: BTreeSet<TurnId> = (self.turns.iter_mut())
            .filter_map(|(turn, t)| std::mem::take(&mut t.awaiting_final).then_some(*turn))
            .collect();
        waits.extend(self.late_final.take().map(|wait| wait.turn));
        for turn in waits {
            self.push(AmdEffect::ClearDeadline(AmdTimer::LateFinal(turn)));
        }
        if self.overall_deadline.take().is_some() {
            self.push(AmdEffect::ClearDeadline(AmdTimer::Overall));
        }
        self.clear_idle();
        self.voicemail.attempt = None;
        self.held = None;
        let pending = self.completion_decision(reason);
        let cancel =
            if self.forbids_reply() { DetachCancel::ForbidReplies } else { DetachCancel::Keep };
        self.answer(AmdAnswer::Detach { run: self.run, pending, cancel });
    }

    /// The session applied `Detach` or closed. A close before completion cancels the run.
    fn on_detached(&mut self) {
        self.finish(AmdReason::Cancelled);
        self.teardown = match self.teardown {
            Teardown::AwaitingBoth => Teardown::AwaitingLegs,
            Teardown::AwaitingDetached => return self.complete(),
            other => other,
        };
    }

    /// The runtime closed the legs. Legs that close before completion failed.
    fn on_legs_closed(&mut self) {
        self.finish(AmdReason::InternalError);
        self.teardown = match self.teardown {
            Teardown::AwaitingBoth => Teardown::AwaitingDetached,
            Teardown::AwaitingLegs => return self.complete(),
            other => other,
        };
    }

    fn complete(&mut self) {
        self.teardown = Teardown::Done;
        if let Lifecycle::Finished(outcome) = &self.lifecycle {
            let outcome = outcome.clone();
            self.push(AmdEffect::Emit(AmdEvent::Completed(outcome)));
        }
    }
}

#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "harness.rs"]
mod harness;

#[cfg(test)]
#[path = "turns_tests.rs"]
mod turns_tests;

#[cfg(test)]
#[path = "replies_tests.rs"]
mod replies_tests;

#[cfg(test)]
#[path = "voicemail_tests.rs"]
mod voicemail_tests;

#[cfg(test)]
#[path = "limits_tests.rs"]
mod limits_tests;

#[cfg(test)]
#[path = "invariants_tests.rs"]
mod invariants_tests;
