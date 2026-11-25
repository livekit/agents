//! The AMD machine's test driver. It steps inputs on a virtual clock, fires the deadlines the
//! machine arms, as the executor does, and records every effect. It decides nothing: tests assert
//! on what it recorded.

use super::*;
use crate::agent_session::machine::activity::scheduling::{SpeechId, SpeechKind};
pub(super) use crate::amd::effects::{AGENT_REPLY, ClassifierItem, LIKELY_HUMAN, MenuRequest};
pub(super) use crate::amd::events::MenuObserved;
pub(super) use crate::amd::options::{
    DEFAULT_HUMAN_INSTRUCTIONS, DEFAULT_IVR_INSTRUCTIONS, DEFAULT_SCREENING_INSTRUCTIONS,
    DEFAULT_VOICEMAIL_INSTRUCTIONS, VOICEMAIL_PLAYED_NOTE,
};
use crate::amd::{DetachCancel, ReplyDecision, SpeechObservation, TurnEnd, UserSpeechObservation};
use lk_agents_abi::{FunctionCall, FunctionCallOutput};

pub(super) const RUN: AmdRunId = AmdRunId(7);
/// The agent at entry.
pub(super) const AGENT: &str = "caller";
/// `AMDMenuObservedEvent.session_id` in [`setup`].
pub(super) const SESSION: &str = "amd_session";
/// The default `machine_silence_threshold`.
pub(super) const SILENCE: Duration = Duration::from_millis(1500);
/// Activation time in [`H::started`].
pub(super) const T0: Ts = Ts::from_nanos(1_000_000_000);

pub(super) fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

pub(super) fn secs(n: u64) -> Duration {
    Duration::from_secs(n)
}

/// An allowed reply under the agent's own instructions, at once.
pub(super) fn plain() -> ReplyDecision {
    ReplyDecision::allow()
}

pub(super) fn deny() -> ReplyDecision {
    ReplyDecision::deny()
}

/// An allowed machine-stage reply with stage instructions and the default silence.
pub(super) fn staged(instructions: &str) -> ReplyDecision {
    ReplyDecision {
        allow: true,
        instructions: Some(instructions.to_owned()),
        dtmf: false,
        silence: Some(SILENCE),
    }
}

/// An allowed `machine-ivr` reply: stage instructions, DTMF and the default silence.
pub(super) fn ivr(instructions: &str) -> ReplyDecision {
    ReplyDecision { dtmf: true, ..staged(instructions) }
}

/// An allowed reply in a machine stage without stage instructions: a likely human.
pub(super) fn held(dtmf: bool) -> ReplyDecision {
    ReplyDecision { allow: true, instructions: None, dtmf, silence: Some(SILENCE) }
}

/// A completion that settles the latest turn with `pending`.
pub(super) fn completed_with(pending: ReplyDecision, cancel: DetachCancel) -> AmdAnswer {
    AmdAnswer::Detach { run: RUN, pending, cancel }
}

/// A turn commit, as the session reports it.
#[derive(Debug, Clone)]
pub(super) struct Commit {
    pub text: String,
    pub speaking: bool,
    pub agent: String,
    pub eot_delay: Option<Duration>,
    pub skip_reply: bool,
    /// The gate has not applied `Activated` yet, though the run may have activated.
    pub unbound: bool,
}

impl Commit {
    pub(super) fn text(text: &str) -> Commit {
        Commit {
            text: text.to_owned(),
            speaking: false,
            agent: AGENT.to_owned(),
            eot_delay: None,
            skip_reply: false,
            unbound: false,
        }
    }
}

pub(super) struct H {
    pub amd: Amd,
    pub now: Ts,
    /// Every effect, with the time of the step that produced it.
    pub log: Vec<(Ts, AmdEffect)>,
    /// Armed deadlines, in arming order.
    armed: Vec<(AmdTimer, Ts)>,
    /// The gate applied `Activated`: it does so as soon as the run answers with it.
    gate_active: bool,
    next_turn: u64,
    next_speech: u64,
    next_call: u64,
}

impl H {
    /// A pending run with `options` and the default setup.
    pub(super) fn new(options: AmdOptions) -> H {
        H::with_setup(options, setup())
    }

    pub(super) fn with_setup(options: AmdOptions, setup: AmdSetup) -> H {
        let amd = Amd::new(RUN, options, setup).expect("valid options");
        H {
            amd,
            now: T0,
            log: Vec::new(),
            armed: Vec::new(),
            gate_active: false,
            next_turn: 1,
            next_speech: 100,
            next_call: 1,
        }
    }

    /// A run that attached with the agent idle and started listening at [`T0`].
    pub(super) fn started(options: AmdOptions) -> H {
        H::started_with(options, setup())
    }

    pub(super) fn started_with(options: AmdOptions, setup: AmdSetup) -> H {
        let mut h = H::with_setup(options, setup);
        h.observe(AmdObservation::AgentBusy { busy: false, at: h.now });
        h.step(In::Callee(Callee::Answered));
        h
    }

    /// Session turn ids start at `first` instead of 1.
    pub(super) fn turns_from(&mut self, first: u64) {
        self.next_turn = first;
    }

    pub(super) fn step(&mut self, input: In) -> Vec<AmdEffect> {
        let effects = self.amd.step(self.now, input);
        for effect in &effects {
            match effect {
                AmdEffect::SetDeadline(timer, at) => {
                    self.armed.retain(|(t, _)| t != timer);
                    self.armed.push((*timer, *at));
                }
                AmdEffect::ClearDeadline(timer) => self.armed.retain(|(t, _)| t != timer),
                AmdEffect::Answer(AmdAnswer::Activated { .. }) => self.gate_active = true,
                _ => {}
            }
            self.log.push((self.now, effect.clone()));
        }
        effects
    }

    pub(super) fn observe(&mut self, observation: AmdObservation) -> Vec<AmdEffect> {
        self.step(In::Observed(observation))
    }

    /// Moves the clock by `d`, firing each armed deadline that falls due, in time order.
    pub(super) fn advance(&mut self, d: Duration) {
        let target = self.now.saturating_add(d);
        loop {
            let due = (self.armed.iter().enumerate())
                .filter(|(_, (_, at))| *at <= target)
                .min_by_key(|(i, (_, at))| (*at, *i))
                .map(|(i, entry)| (i, *entry));
            let Some((i, (timer, at))) = due else { break };
            self.armed.remove(i);
            self.now = self.now.max(at);
            self.step(In::Timeout(timer));
        }
        self.now = target;
    }

    /// Moves the clock by `d` without firing anything: the executor runs late.
    pub(super) fn skip(&mut self, d: Duration) {
        self.now = self.now.saturating_add(d);
    }

    /// When `timer` is armed to fire, if it is.
    pub(super) fn armed(&self, timer: AmdTimer) -> Option<Ts> {
        self.armed.iter().find(|(t, _)| *t == timer).map(|(_, at)| *at)
    }

    pub(super) fn armed_timers(&self) -> Vec<(AmdTimer, Ts)> {
        self.armed.clone()
    }

    // Turns.

    pub(super) fn commit(&mut self, text: &str) -> TurnId {
        self.commit_with(Commit::text(text))
    }

    pub(super) fn commit_with(&mut self, commit: Commit) -> TurnId {
        let turn = TurnId(self.next_turn);
        self.next_turn += 1;
        let bound = self.gate_active && !commit.unbound;
        self.observe(AmdObservation::TurnCommitted {
            turn,
            transcript: commit.text,
            eot_delay: commit.eot_delay,
            skip_reply: commit.skip_reply,
            bound,
            speaking: commit.speaking,
            agent: commit.agent,
            at: self.now,
        });
        turn
    }

    pub(super) fn ended(&mut self, turn: TurnId, how: TurnEnd) {
        self.observe(AmdObservation::TurnEnded { turn, how });
    }

    // Classifier.

    pub(super) fn requests(&self) -> Vec<ClassifyRequest> {
        self.effects()
            .filter_map(|e| match e {
                AmdEffect::Classify(request) => Some(request.clone()),
                _ => None,
            })
            .collect()
    }

    /// The latest request for `turn`.
    pub(super) fn request_for(&self, turn: TurnId) -> ClassifyRequest {
        let requests = self.requests();
        let found = requests.into_iter().rev().find(|r| r.turn == turn);
        found.unwrap_or_else(|| panic!("no classifier request for {turn:?}"))
    }

    pub(super) fn requests_for(&self, turn: TurnId) -> usize {
        self.requests().iter().filter(|r| r.turn == turn).count()
    }

    /// The requests cancelled, in order. `CancelClassify` names none: it cancels the request in
    /// flight, which is the latest `Classify` before it.
    pub(super) fn cancelled(&self) -> Vec<ReqId> {
        let mut in_flight = None;
        let mut cancelled = Vec::new();
        for effect in self.effects() {
            match effect {
                AmdEffect::Classify(request) => in_flight = Some(request.req),
                AmdEffect::CancelClassify => {
                    cancelled.push(in_flight.take().expect("a request in flight"));
                }
                _ => {}
            }
        }
        cancelled
    }

    /// Answers `turn`'s latest request with `category`, after no model time.
    pub(super) fn classify(&mut self, turn: TurnId, category: Category) -> Vec<AmdEffect> {
        let req = self.request_for(turn).req;
        self.respond(req, Ok(Classification { category, inference_duration: Duration::ZERO }))
    }

    pub(super) fn respond(
        &mut self,
        req: ReqId,
        result: Result<Classification, ClassifyError>,
    ) -> Vec<AmdEffect> {
        self.step(In::Classified { req, result })
    }

    /// Commits a turn with `text` and classifies it as `category`.
    pub(super) fn turn(&mut self, text: &str, category: Category) -> TurnId {
        let turn = self.commit(text);
        self.classify(turn, category);
        turn
    }

    // Answers and events.

    pub(super) fn effects(&self) -> impl Iterator<Item = &AmdEffect> {
        self.log.iter().map(|(_, e)| e)
    }

    pub(super) fn answers(&self) -> Vec<AmdAnswer> {
        self.effects()
            .filter_map(|e| match e {
                AmdEffect::Answer(answer) => Some(answer.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every `ReplyDecided` for `turn`, in order.
    pub(super) fn decisions(&self, turn: TurnId) -> Vec<ReplyDecision> {
        self.answers()
            .into_iter()
            .filter_map(|a| match a {
                AmdAnswer::ReplyDecided { turn: t, decision, .. } if t == turn => Some(decision),
                _ => None,
            })
            .collect()
    }

    /// The one `ReplyDecided` for `turn`.
    pub(super) fn decision(&self, turn: TurnId) -> ReplyDecision {
        let decisions = self.decisions(turn);
        assert_eq!(decisions.len(), 1, "one decision for {turn:?}: {decisions:?}");
        decisions.into_iter().next().unwrap()
    }

    /// Whether a `ReplyDecided` allowed `turn` a reply.
    pub(super) fn allowed(&self, turn: TurnId) -> bool {
        self.decisions(turn).iter().any(|d| d.allow)
    }

    pub(super) fn detaches(&self) -> Vec<AmdAnswer> {
        self.answers().into_iter().filter(|a| matches!(a, AmdAnswer::Detach { .. })).collect()
    }

    /// The one `Detach`.
    pub(super) fn detach(&self) -> AmdAnswer {
        let detaches = self.detaches();
        assert_eq!(detaches.len(), 1, "one Detach: {detaches:?}");
        detaches.into_iter().next().unwrap()
    }

    pub(super) fn events(&self) -> Vec<AmdEvent> {
        self.effects()
            .filter_map(|e| match e {
                AmdEffect::Emit(event) => Some(event.clone()),
                _ => None,
            })
            .collect()
    }

    pub(super) fn predictions(&self) -> Vec<Prediction> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                AmdEvent::Prediction(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    /// `(turn_id, reason)` of every prediction, in order.
    pub(super) fn reasons(&self) -> Vec<(u64, AmdReason)> {
        self.predictions().iter().map(|p| (p.turn_id, p.reason)).collect()
    }

    pub(super) fn last_prediction(&self) -> Prediction {
        self.predictions().pop().expect("a prediction")
    }

    pub(super) fn completions(&self) -> Vec<Outcome> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                AmdEvent::Completed(o) => Some(o),
                _ => None,
            })
            .collect()
    }

    /// The session applies `Detach` and the runtime closes the legs.
    pub(super) fn finish(&mut self) {
        self.observe(AmdObservation::Detached { run: RUN });
        self.step(In::LegsClosed);
    }

    /// [`H::finish`], then the one `amd_completed`.
    pub(super) fn outcome(&mut self) -> Outcome {
        self.finish();
        let completions = self.completions();
        assert_eq!(completions.len(), 1, "one amd_completed: {completions:?}");
        completions.into_iter().next().unwrap()
    }

    pub(super) fn menu_requests(&self) -> Vec<MenuRequest> {
        self.effects()
            .filter_map(|e| match e {
                AmdEffect::ExtractMenu(request) => Some(request.clone()),
                _ => None,
            })
            .collect()
    }

    /// How many times the menu leg was told to stop.
    pub(super) fn cancelled_menus(&self) -> usize {
        self.effects().filter(|e| matches!(e, AmdEffect::CancelMenu)).count()
    }

    pub(super) fn menus(&self) -> Vec<MenuObserved> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                AmdEvent::MenuObserved(m) => Some(m),
                _ => None,
            })
            .collect()
    }

    // Participant speech.

    pub(super) fn speech_started(&mut self, at: Ts) {
        self.observe(AmdObservation::UserSpeech(UserSpeechObservation::Started { at }));
    }

    pub(super) fn speech_ended(&mut self, at: Ts) {
        self.observe(AmdObservation::UserSpeech(UserSpeechObservation::Ended { at }));
    }

    pub(super) fn session_final(&mut self, text: &str) {
        let text = text.to_owned();
        self.observe(AmdObservation::UserSpeech(UserSpeechObservation::Final {
            text,
            at: self.now,
        }));
    }

    pub(super) fn amd_final(&mut self, text: &str) {
        self.step(In::AmdFinal(text.to_owned()));
    }

    // Agent speech.

    /// The gate reports whether the agent is busy: a queued or playing speech other than the
    /// preemptive generation, thinking, or a turn waiting on its hook.
    pub(super) fn busy(&mut self, busy: bool) {
        self.observe(AmdObservation::AgentBusy { busy, at: self.now });
    }

    pub(super) fn new_speech(&mut self) -> SpeechId {
        self.next_speech += 1;
        SpeechId(self.next_speech)
    }

    pub(super) fn speech(&mut self, speech: SpeechId, event: SpeechObservation) {
        self.observe(AmdObservation::Speech { speech, event, at: self.now });
    }

    /// A speech created now, recorded in the chat context.
    pub(super) fn created(&mut self, kind: SpeechKind) -> SpeechId {
        let speech = self.new_speech();
        let event = SpeechObservation::Created { kind, add_to_chat_ctx: true, playing: false };
        self.speech(speech, event);
        speech
    }

    /// The speech plays `text` to its end, not interrupted.
    pub(super) fn play(&mut self, speech: SpeechId, text: &str) {
        self.speech(speech, SpeechObservation::PlayoutAdvanced);
        let text = Some(text.to_owned());
        self.speech(speech, SpeechObservation::ItemAdded { text, interrupted: false });
        self.speech(speech, SpeechObservation::Done { interrupted: false, failed: false });
    }

    /// A speech that replies to no turn, such as a `say()`, plays `text` now.
    pub(super) fn say(&mut self, text: &str) -> SpeechId {
        let speech = self.created(SpeechKind::Say);
        self.play(speech, text);
        speech
    }

    /// The session creates `turn`'s reply after an allowed decision.
    pub(super) fn reply(&mut self, turn: TurnId) -> SpeechId {
        let speech = self.created(SpeechKind::TurnReply);
        self.observe(AmdObservation::ReplyCreated { turn, speech });
        speech
    }

    /// The session holds `turn`'s allowed reply for silence, then releases it.
    pub(super) fn silence_held(&mut self, turn: TurnId) {
        self.observe(AmdObservation::SilenceHold { turn, held: true, at: self.now });
    }

    pub(super) fn silence_released(&mut self, turn: TurnId) {
        self.observe(AmdObservation::SilenceHold { turn, held: false, at: self.now });
    }

    /// The session applies `turn`'s allowed decision as the gate does: the reply exists, the
    /// turn ends with it, then the silence hold starts if the decision asks for one.
    pub(super) fn replied(&mut self, turn: TurnId) -> SpeechId {
        let speech = self.reply(turn);
        self.ended(turn, TurnEnd::Replied);
        speech
    }

    /// `turn`'s allowed machine-stage reply waits for silence, then plays `text` to its end.
    pub(super) fn reply_after_silence(&mut self, turn: TurnId, text: &str) -> SpeechId {
        let speech = self.replied(turn);
        self.silence_held(turn);
        self.silence_released(turn);
        self.play(speech, text);
        speech
    }

    /// `turn`'s allowed reply plays `text` to its end, after the silence hold its decision asked
    /// for, if any.
    pub(super) fn reply_played(&mut self, turn: TurnId, text: &str) -> SpeechId {
        let silence = self.decisions(turn).last().is_some_and(|d| d.silence.is_some());
        if silence {
            return self.reply_after_silence(turn, text);
        }
        let speech = self.replied(turn);
        self.play(speech, text);
        speech
    }

    // Tools.

    /// A `send_dtmf_events` call created now and its output.
    pub(super) fn dtmf(&mut self, digits: &str) -> (FunctionCall, FunctionCallOutput) {
        self.tool("send_dtmf_events", digits, false, self.wall())
    }

    pub(super) fn tool(
        &mut self,
        name: &str,
        digits: &str,
        is_error: bool,
        created_at: SystemTime,
    ) -> (FunctionCall, FunctionCallOutput) {
        let n = self.next_call;
        self.next_call += 1;
        let call = FunctionCall {
            id: format!("item_{n}"),
            call_id: format!("dtmf_{n}"),
            arguments: serde_json::json!({ "events": digits.chars().map(String::from).collect::<Vec<_>>() })
                .to_string(),
            name: name.to_owned(),
            created_at,
            extra: Default::default(),
            group_id: None,
        };
        let text = if is_error {
            "Send failed".to_owned()
        } else {
            format!("Successfully sent DTMF events: {digits}")
        };
        let mut output = FunctionCallOutput::new(
            format!("item_{n}_out"),
            call.call_id.clone(),
            text,
            is_error,
            created_at,
        );
        output.name = name.to_owned();
        (call, output)
    }

    pub(super) fn executed(&mut self, calls: &[(FunctionCall, FunctionCallOutput)]) {
        let calls = (calls.iter())
            .map(|(call, output)| crate::amd::ToolCallRecord {
                call: call.clone(),
                output: Some(output.clone()),
            })
            .collect();
        self.observe(AmdObservation::ToolsExecuted { calls });
    }

    /// The wall clock: entry plus the virtual time since [`T0`].
    pub(super) fn wall(&self) -> SystemTime {
        setup().entered_at + self.now.saturating_duration_since(T0)
    }
}

pub(super) const CATEGORIES: [Category; 7] = [
    Category::Uncertain,
    Category::Wait,
    Category::MachineScreening,
    Category::MachineVm,
    Category::MachineIvr,
    Category::Human,
    Category::MachineUnavailable,
];

/// The recommended list of spec §9.1.
pub(super) fn spec_recommended(stage: Stage) -> std::collections::BTreeSet<Category> {
    use Category::*;
    match stage {
        Stage::Uncertain => CATEGORIES.into_iter().collect(),
        Stage::MachineScreening => {
            [MachineScreening, Human, MachineVm, MachineUnavailable, Uncertain, Wait].into()
        }
        Stage::MachineVm => {
            [MachineVm, Human, MachineIvr, MachineUnavailable, Uncertain, Wait].into()
        }
        Stage::MachineIvr => {
            [MachineIvr, Human, MachineVm, MachineUnavailable, Uncertain, Wait].into()
        }
        Stage::Human | Stage::MachineUnavailable => Default::default(),
    }
}

/// The recommended list a request carries, as a set.
pub(super) fn listed(request: &ClassifyRequest) -> std::collections::BTreeSet<Category> {
    request.recommended.iter().copied().collect()
}

/// The default setup: session STT, no AMD STT, an LLM classifier, the agent [`AGENT`].
pub(super) fn setup() -> AmdSetup {
    AmdSetup {
        agent: AGENT.to_owned(),
        session_id: SESSION.to_owned(),
        entered_at: lk_agents_abi::utils::clock::simulated_epoch(),
        session_stt: true,
        amd_stt: false,
        extract_menus: true,
        insert_user_items: false,
    }
}

/// A request's system message, parsed.
pub(super) fn state(request: &ClassifyRequest) -> serde_json::Value {
    serde_json::from_str(&request.state_json()).expect("the state is JSON")
}

/// The user messages of a request's history, as `(turn_id, text)`.
pub(super) fn user_turns(request: &ClassifyRequest) -> Vec<(u64, String)> {
    (request.history.iter())
        .filter_map(|item| match item {
            ClassifierItem::User { turn_id, transcript } => {
                Some((*turn_id, transcript.for_classifier()))
            }
            _ => None,
        })
        .collect()
}

/// The history's messages as `("user" | "assistant", text)`, tool items left out.
pub(super) fn messages(request: &ClassifyRequest) -> Vec<(&'static str, String)> {
    (request.history.iter())
        .filter_map(|item| match item {
            ClassifierItem::User { transcript, .. } => Some(("user", transcript.for_classifier())),
            ClassifierItem::Agent { text } => Some(("assistant", text.clone())),
            _ => None,
        })
        .collect()
}

/// The DTMF calls of a request's history.
pub(super) fn dtmf_calls(request: &ClassifyRequest) -> Vec<FunctionCall> {
    (request.history.iter())
        .filter_map(|item| match item {
            ClassifierItem::DtmfCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}
