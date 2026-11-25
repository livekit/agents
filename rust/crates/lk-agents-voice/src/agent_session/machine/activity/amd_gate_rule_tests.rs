//! The session side of the AMD reply gate, at the decision core: one test per transition and
//! local rule of the AMD proposal ("The session side"). Python's `tests/test_amd.py` on
//! `main` is the reference for intent; a test that ports one names it.
//!
//! Every test reads what the session does, not what it says it did: whether a reply was created
//! (`StartSpeech` and the request it carries), whether its playback was authorized (a speech
//! view with a generation), whether it was cancelled (an interrupted view), the items the
//! conversation saved, and the observations AMD receives.

use crate::agent::TurnEdit;
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::agent_session::machine::coordination::VoiceCallback;
use crate::agent_session::{
    Command, HookId, HookOutcome, Now, Sent, VoiceCommandOrigin, VoiceDriver, VoiceEvent,
    VoiceHostEvent,
};
use crate::amd::{
    AmdAnswer, AmdCommand, AmdObservation, AmdRunId, DetachCancel, Opening, ReplyDecision,
    SpeechObservation, TurnEnd, UserSpeechObservation,
};
use crate::audio_recognition::{EndOfTurnInfo, TurnDetectionMode, TurnId};
use crate::events::SessionEvent;
use crate::generation::{GenEvent, ReplyContext};
use crate::speech_handle::SpeechUpdate;
use crate::trace::{GateStep, Step};
use lk_agents_abi::{ChatItem, Role, Ts, VadEvent, VadEventType};
use std::time::Duration;

const RUN: AmdRunId = AmdRunId(1);
/// `machine_silence_threshold`'s default (spec §4.1).
const SILENCE: Duration = Duration::from_millis(1500);
const STAGE: &str = "You reached a call screener. Say who you are and why you are calling.";

/// The name the host gives AMD's built-in DTMF tool.
const SEND_DTMF_TOOL: &str = "send_dtmf_events";

/// AMD's built-in DTMF tool, as a host declares it at attach.
pub(super) fn dtmf_tool() -> lk_agents_abi::ToolSpec {
    lk_agents_abi::ToolSpec {
        name: SEND_DTMF_TOOL.into(),
        description: "Send a list of DTMF events to the telephony provider.".into(),
        json_schema: serde_json::json!({
            "type": "object",
            "properties": { "events": { "type": "array", "items": { "type": "string" } } },
            "required": ["events"],
        }),
        flags: Default::default(),
        provider_tool: None,
    }
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn ts(n: u64) -> Ts {
    Ts::ZERO.saturating_add(ms(n))
}

fn info(turn: u64, transcript: &str) -> EndOfTurnInfo {
    EndOfTurnInfo {
        turn_id: TurnId(turn),
        owner: TurnDetectionMode::Vad,
        stt_enabled: true,
        skip_reply: false,
        new_transcript: transcript.into(),
        transcript_confidence: 0.9,
        metrics: Default::default(),
        backchannel_over_agent: false,
        late: false,
        snapshot: Default::default(),
        promoted: None,
    }
}

fn allow() -> ReplyDecision {
    ReplyDecision::allow()
}

fn deny() -> ReplyDecision {
    ReplyDecision::deny()
}

fn staged() -> ReplyDecision {
    ReplyDecision { instructions: Some(STAGE.into()), ..ReplyDecision::allow() }
}

fn held(silence: Duration) -> ReplyDecision {
    ReplyDecision { silence: Some(silence), ..staged() }
}

/// One session's decision core, driven only through its inputs, with every effect kept.
struct Gate {
    d: VoiceDriver,
    at: u64,
    out: Vec<VoiceEffect>,
}

impl Gate {
    fn new() -> Self {
        Gate::with(|_| {})
    }

    /// With `hooked`, `on_user_turn_completed` is hosted, so a turn parks on its hook.
    fn hooked() -> Self {
        Gate::with(|d| d.activity.hook_bound = Some(Duration::from_secs(10)))
    }

    fn with(setup: impl FnOnce(&mut VoiceDriver)) -> Self {
        let mut d = crate::agent_session::tests::driver();
        d.journalled = true;
        setup(&mut d);
        let mut gate = Gate { d, at: 0, out: Vec::new() };
        gate.step(None);
        gate
    }

    fn now(&self) -> Now {
        Now { at: ts(self.at), wall: std::time::UNIX_EPOCH + ms(self.at) }
    }

    /// Moves virtual time to `at` ms and lets every timer due by then fire.
    fn advance_to(&mut self, at: u64) {
        assert!(at >= self.at, "time runs forward");
        self.at = at;
        self.step(None);
    }

    fn step(&mut self, input: Option<VoiceEvent>) -> usize {
        let from = self.out.len();
        let now = self.now();
        let effects = self.d.handle(input, now);
        self.out.extend(effects);
        from
    }

    fn send(&mut self, command: Command) -> usize {
        self.step(Some(VoiceEvent::Host(Sent { origin: VoiceCommandOrigin::Host, command })))
    }

    fn attach(&mut self, run: AmdRunId) -> usize {
        self.send(Command::Amd(AmdCommand::Attach { run, dtmf_tool: dtmf_tool(), call: None }))
    }

    /// AMD's answers are inputs of their own, not host commands.
    fn answer(&mut self, answer: AmdAnswer) -> usize {
        self.step(Some(VoiceEvent::Amd(answer)))
    }

    fn activate(&mut self, opening: Option<Opening>) -> usize {
        self.answer(AmdAnswer::Activated { run: RUN, opening })
    }

    /// Attached and listening, with no opening.
    fn listening() -> Self {
        let mut gate = Gate::new();
        gate.attach(RUN);
        gate.activate(None);
        gate
    }

    fn listening_hooked() -> Self {
        let mut gate = Gate::hooked();
        gate.attach(RUN);
        gate.activate(None);
        gate
    }

    fn decide(&mut self, turn: u64, decision: ReplyDecision) -> usize {
        self.answer(AmdAnswer::ReplyDecided { run: RUN, turn: TurnId(turn), decision })
    }

    fn detach(&mut self, pending: ReplyDecision, cancel: DetachCancel) -> usize {
        self.answer(AmdAnswer::Detach { run: RUN, pending, cancel })
    }

    fn commit_info(&mut self, info: EndOfTurnInfo) -> usize {
        self.d.defer_callback(VoiceCallback::EndOfTurn(info));
        self.step(None)
    }

    fn commit(&mut self, turn: u64, text: &str) -> usize {
        self.commit_info(info(turn, text))
    }

    /// The VAD reports speech that started `speech_ms` before now.
    fn vad_start(&mut self, speech_ms: u64) -> usize {
        let secs = speech_ms as f64 / 1000.0;
        let ev = VadEvent::new(VadEventType::StartOfSpeech, 0, self.at as f64 / 1000.0, secs, 0.0);
        self.step(Some(VoiceEvent::Vad(ev)))
    }

    /// The VAD reports speech that ended `silence_ms` before now.
    fn vad_end(&mut self, silence_ms: u64) -> usize {
        let secs = silence_ms as f64 / 1000.0;
        let ev = VadEvent::new(VadEventType::EndOfSpeech, 0, self.at as f64 / 1000.0, 0.3, secs);
        self.step(Some(VoiceEvent::Vad(ev)))
    }

    /// The sink played the speech's first frame.
    fn first_frame(&mut self, speech: SpeechId) -> usize {
        self.step(Some(VoiceEvent::Gen(GenEvent::FirstFrame { speech })))
    }

    fn hook_done(&mut self, hook: HookId, outcome: HookOutcome) -> usize {
        self.send(Command::HookDone { id: hook, outcome })
    }

    /// `on_user_turn_completed` calls the host made, oldest first.
    fn hooks(&self) -> Vec<HookId> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::Host(VoiceHostEvent::Hook(call)) => Some(call.id()),
                _ => None,
            })
            .collect()
    }

    fn last_hook(&self) -> HookId {
        *self.hooks().last().expect("a turn hook was issued")
    }

    /// Every generation the session started, in order: the speech and its request.
    fn generations(&self) -> Vec<(SpeechId, &ReplyContext)> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::StartSpeech { speech, context, .. } => Some((*speech, &**context)),
                _ => None,
            })
            .collect()
    }

    fn generations_since(&self, from: usize) -> Vec<SpeechId> {
        self.out[from..]
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::StartSpeech { speech, .. } => Some(*speech),
                _ => None,
            })
            .collect()
    }

    /// The generation that answers the user words `text`: its request ends with them, or with
    /// stage instructions after them.
    // The newest user message, not the last two items: a superseded or denied turn's words are
    // committed just before the next turn's ("Apply a decision": a denied turn commits its user
    // item), so they sit second to last in the newer turn's request.
    fn reply_to(&self, text: &str) -> Option<(SpeechId, &ReplyContext)> {
        self.generations().into_iter().find(|(_, ctx)| {
            let last_user = ctx
                .request
                .chat_ctx
                .items
                .iter()
                .rev()
                .find_map(|item| item.as_message().filter(|m| m.role == Role::User));
            last_user.is_some_and(|m| m.raw_text_content().as_deref() == Some(text))
        })
    }

    fn only_reply_to(&self, text: &str) -> SpeechId {
        let replies: Vec<_> = self
            .generations()
            .into_iter()
            .filter(|(_, ctx)| {
                ctx.request.chat_ctx.items.iter().any(|item| {
                    item.as_message().is_some_and(|m| {
                        m.role == Role::User && m.raw_text_content().as_deref() == Some(text)
                    })
                })
            })
            .map(|(id, _)| id)
            .collect();
        assert_eq!(replies.len(), 1, "one reply to {text:?}, got {replies:?}");
        replies[0]
    }

    /// The scheduler let the speech play: its view carries an authorized generation.
    fn authorized(&self, speech: SpeechId) -> bool {
        self.out.iter().any(|e| {
            matches!(e, VoiceEffect::Speech(id, SpeechUpdate::View(view))
                if *id == speech && !view.generations.is_empty())
        })
    }

    fn authorized_since(&self, from: usize, speech: SpeechId) -> bool {
        self.out[from..].iter().any(|e| {
            matches!(e, VoiceEffect::Speech(id, SpeechUpdate::View(view))
                if *id == speech && !view.generations.is_empty())
        })
    }

    /// The speech was interrupted or cancelled before or while it played.
    fn cancelled(&self, speech: SpeechId) -> bool {
        self.out.iter().any(|e| match e {
            VoiceEffect::Speech(id, SpeechUpdate::View(view)) => {
                *id == speech && view.interrupted()
            }
            VoiceEffect::CancelSpeech(id) => *id == speech,
            _ => false,
        })
    }

    /// Speeches the session announced, by kind.
    fn created(
        &self,
        kind: crate::agent_session::machine::activity::scheduling::SpeechKind,
    ) -> Vec<SpeechId> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::Emit(SessionEvent::SpeechCreated { speech, kind: k, .. })
                    if *k == kind =>
                {
                    Some(*speech)
                }
                _ => None,
            })
            .collect()
    }

    /// The opening, a `say` made at activation.
    fn one_opening(&self) -> SpeechId {
        use crate::agent_session::machine::activity::scheduling::SpeechKind;
        let said = self.created(SpeechKind::Say);
        assert_eq!(said.len(), 1, "one opening, created at activation");
        said[0]
    }

    /// Everything the conversation recorded.
    fn saved(&self) -> Vec<&ChatItem> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::Emit(SessionEvent::ConversationItemAdded { item, .. }) => {
                    Some(&**item)
                }
                _ => None,
            })
            .collect()
    }

    fn saved_user_words(&self) -> Vec<String> {
        self.saved()
            .into_iter()
            .filter_map(|item| item.as_message())
            .filter(|m| m.role == Role::User)
            .filter_map(|m| m.raw_text_content())
            .collect()
    }

    /// Whether any record the session keeps or publishes holds `text`.
    fn persisted_anywhere(&self, text: &str) -> bool {
        let holds = |item: &ChatItem| {
            item.as_message().and_then(|m| m.raw_text_content()).is_some_and(|t| t.contains(text))
        };
        self.out.iter().any(|e| match e {
            VoiceEffect::Emit(SessionEvent::ConversationItemAdded { item, .. }) => holds(item),
            VoiceEffect::ChatContext(ctx) => ctx.items.iter().any(holds),
            VoiceEffect::ConversationHistory { history, .. } => history.iter().any(holds),
            VoiceEffect::StorageCommit { history, .. }
            | VoiceEffect::StorageCheckpoint { history, .. } => history.items.iter().any(holds),
            _ => false,
        })
    }

    /// What `run` observed, in order.
    fn observed(&self, run: AmdRunId) -> Vec<&AmdObservation> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::Amd { run: r, observation: o } if *r == run => Some(o),
                _ => None,
            })
            .collect()
    }

    fn observed_since(&self, from: usize) -> Vec<&AmdObservation> {
        self.out[from..]
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::Amd { observation: o, .. } => Some(o),
                _ => None,
            })
            .collect()
    }

    fn ended(&self, turn: u64) -> Vec<TurnEnd> {
        self.observed(RUN)
            .into_iter()
            .filter_map(|o| match o {
                AmdObservation::TurnEnded { turn: t, how } if *t == TurnId(turn) => Some(*how),
                _ => None,
            })
            .collect()
    }

    fn reply_created_for(&self, turn: u64) -> Vec<SpeechId> {
        self.observed(RUN)
            .into_iter()
            .filter_map(|o| match o {
                AmdObservation::ReplyCreated { turn: t, speech } if *t == TurnId(turn) => {
                    Some(*speech)
                }
                _ => None,
            })
            .collect()
    }

    fn silence_holds(&self, turn: u64) -> Vec<(bool, Ts)> {
        self.observed(RUN)
            .into_iter()
            .filter_map(|o| match o {
                AmdObservation::SilenceHold { turn: t, held, at } if *t == TurnId(turn) => {
                    Some((*held, *at))
                }
                _ => None,
            })
            .collect()
    }

    fn gate_steps(&self) -> Vec<&GateStep> {
        self.out
            .iter()
            .filter_map(|e| match e {
                VoiceEffect::Journal(step) => match &**step {
                    Step::Gate(g) => Some(g),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    fn tools_of(&self, speech: SpeechId) -> Vec<String> {
        let (_, ctx) = self.generations().into_iter().find(|(id, _)| *id == speech).unwrap();
        ctx.request.tools.iter().map(|t| t.name.clone()).collect()
    }

    fn request_texts(&self, speech: SpeechId) -> Vec<String> {
        let (_, ctx) = self.generations().into_iter().find(|(id, _)| *id == speech).unwrap();
        ctx.request
            .chat_ctx
            .items
            .iter()
            .filter_map(|i| i.as_message())
            .filter_map(|m| m.raw_text_content())
            .collect()
    }
}

// ---- Commit and decision (proposal "Transitions": Commit, Decision; rules 1 and 2) ----

/// Rule 1: the session holds at commit in the same transition; nothing replies before the
/// turn's decision. Python: `test_reply_waits_for_classification_and_silence`.
#[test]
fn a_committed_turn_waits_for_its_decision() {
    let mut g = Gate::listening();
    g.commit(1, "hello who is this");
    assert!(g.reply_to("hello who is this").is_none(), "no reply before the decision");

    g.decide(1, allow());
    let speech = g.only_reply_to("hello who is this");
    assert!(g.authorized(speech), "an allowed reply with no silence plays at once");
    assert_eq!(g.reply_created_for(1), [speech]);
    assert_eq!(g.ended(1), [TurnEnd::Replied]);
}

/// Each `TurnCommitted` carries the fields the proposal lists ("Session to AMD"), with the
/// time it describes.
#[test]
fn turn_committed_carries_the_turn_and_its_facts() {
    let mut g = Gate::listening();
    g.advance_to(1_000);
    let mut turn = info(1, "please leave a message");
    turn.metrics.end_of_turn_delay = Some(ms(400));
    g.commit_info(turn);
    let committed: Vec<_> = g
        .observed(RUN)
        .into_iter()
        .filter(|o| matches!(o, AmdObservation::TurnCommitted { .. }))
        .cloned()
        .collect();
    assert_eq!(
        committed,
        [AmdObservation::TurnCommitted {
            turn: TurnId(1),
            transcript: "please leave a message".into(),
            eot_delay: Some(ms(400)),
            skip_reply: false,
            speaking: false,
            agent: "agent".into(),
            at: ts(1_000),
        }]
    );
}

/// A denied turn commits only its user item (proposal "Apply a decision"). Python's
/// `wait` row: `test_wait_skips_reply_and_allows_a_fresh_prediction`.
#[test]
fn a_denied_turn_saves_its_words_and_never_replies() {
    let mut g = Gate::listening();
    g.commit(1, "please hold");
    g.decide(1, deny());
    assert!(g.generations().is_empty(), "a denied turn has no reply");
    assert_eq!(g.saved_user_words(), ["please hold"]);
    assert_eq!(g.ended(1), [TurnEnd::Denied]);

    // `Hold::Denied` exits at the next commit: the next turn is decided on its own.
    g.commit(2, "thank you for holding");
    assert!(g.reply_to("thank you for holding").is_none());
    g.decide(2, allow());
    let speech = g.only_reply_to("thank you for holding");
    assert!(g.authorized(speech));
}

/// Rule 2: a decision naming a turn other than the latest committed one is dropped, as
/// `on_hook_done` drops a late hook answer. Python: `turn_id != self._turn_id` denies.
#[test]
fn a_decision_for_an_older_turn_is_dropped() {
    let mut g = Gate::listening();
    g.commit(1, "hello");
    g.commit(2, "is anyone there");
    g.decide(1, allow());
    assert!(g.generations().is_empty(), "a stale decision replies to neither turn");
    assert!(
        g.gate_steps()
            .iter()
            .any(|s| matches!(s, GateStep::AnswerDropped { turn: Some(TurnId(1)), .. })),
        "the stale answer is journalled as dropped: {:?}",
        g.gate_steps()
    );

    g.decide(2, allow());
    let speech = g.only_reply_to("is anyone there");
    assert!(g.authorized(speech));
    assert!(g.reply_to("hello").is_none(), "the superseded turn never replies");
}

/// A decision for a turn not committed yet is not stored for it.
#[test]
fn a_decision_for_a_future_turn_is_dropped() {
    let mut g = Gate::listening();
    g.commit(1, "hello");
    g.decide(2, allow());
    g.commit(2, "hello again");
    assert!(g.generations().is_empty(), "turn 2 is undecided after its own commit");
    g.decide(2, allow());
    g.only_reply_to("hello again");
}

/// Answers name their run: another run's answer is dropped.
#[test]
fn an_answer_from_another_run_is_dropped() {
    let mut g = Gate::listening();
    g.commit(1, "hello");
    g.answer(AmdAnswer::ReplyDecided { run: AmdRunId(9), turn: TurnId(1), decision: allow() });
    assert!(g.generations().is_empty());
    assert!(
        g.gate_steps()
            .iter()
            .any(|s| matches!(s, GateStep::AnswerDropped { run: AmdRunId(9), .. })),
        "{:?}",
        g.gate_steps()
    );
    g.answer(AmdAnswer::Detach { run: AmdRunId(9), pending: allow(), cancel: DetachCancel::Keep });
    assert!(g.generations().is_empty(), "another run's detach does not settle this run's turn");
    assert!(
        !g.observed(RUN).iter().any(|o| matches!(o, AmdObservation::Detached { .. })),
        "the gate stays"
    );
}

// ---- The customer hook (proposal "Customer hook returns") ----

/// A decision may arrive before the hook returns; `finish_turn` consumes it.
#[test]
fn a_decision_before_the_hook_returns_applies_when_it_returns() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    g.decide(1, staged());
    assert!(g.generations().is_empty(), "the turn is still in its hook");
    g.hook_done(hook, HookOutcome::Returned);
    let speech = g.only_reply_to("hello");
    assert!(g.request_texts(speech).iter().any(|t| t == STAGE));
    assert!(g.authorized(speech));
}

/// A hook that returns first parks the turn until its decision.
#[test]
fn a_turn_whose_hook_returned_first_waits_parked() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    g.hook_done(hook, HookOutcome::Returned);
    assert!(g.generations().is_empty(), "parked: no reply without a decision");
    g.decide(1, allow());
    let speech = g.only_reply_to("hello");
    assert!(g.authorized(speech));
}

/// A parked turn's decision applies the hook's edits too.
#[test]
fn a_parked_turn_keeps_its_hook_edits() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let hook = g.last_hook();
    g.hook_done(hook, HookOutcome::Turn(vec![TurnEdit::UpdateUserMessage("hello there".into())]));
    assert!(g.generations().is_empty(), "parked until its decision");
    g.decide(1, allow());
    g.only_reply_to("hello there");
}

/// Rule 3: a newer commit denies a parked turn locally, with no answer from AMD. A denied turn
/// commits only its user item.
#[test]
fn a_newer_commit_supersedes_a_parked_turn() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    g.commit(2, "hello is anyone there");
    assert_eq!(g.ended(1), [TurnEnd::Superseded]);
    assert!(g.saved_user_words().contains(&"hello".to_string()));

    g.decide(1, allow());
    assert!(g.generations().is_empty(), "the superseded turn's late decision is dropped");
    g.hook_done(g.last_hook(), HookOutcome::Returned);
    g.decide(2, allow());
    g.only_reply_to("hello is anyone there");
    assert!(g.reply_to("hello").is_none());
}

/// A turn whose hook answers `StopResponse` ends without a decision applied and holds until the
/// next commit (proposal "Turn exits without a decision applied").
#[test]
fn stop_response_ends_the_turn_and_a_later_decision_does_nothing() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    g.hook_done(g.last_hook(), HookOutcome::Turn(vec![TurnEdit::StopResponse]));
    assert_eq!(g.ended(1), [TurnEnd::Stopped]);
    g.decide(1, allow());
    assert!(g.generations().is_empty());
}

/// `skip_reply` ends the turn with its item and no reply.
#[test]
fn a_skip_reply_turn_ends_with_its_item() {
    let mut g = Gate::listening();
    let mut turn = info(1, "one moment");
    turn.skip_reply = true;
    g.commit_info(turn);
    assert!(g.observed(RUN).iter().any(|o| matches!(
        o,
        AmdObservation::TurnCommitted { turn: TurnId(1), skip_reply: true, .. }
    )));
    assert_eq!(g.ended(1), [TurnEnd::SkipReply]);
    g.decide(1, allow());
    assert!(g.generations().is_empty());
    assert_eq!(g.saved_user_words(), ["one moment"]);
}

/// A newer commit while the older turn is still in its hook supersedes it through
/// `supersede_turn`, and the observation says so.
#[test]
fn a_turn_superseded_in_its_hook_ends_as_superseded() {
    let mut g = Gate::listening_hooked();
    g.commit(1, "hello");
    let first = g.last_hook();
    g.commit(2, "are you there");
    g.hook_done(first, HookOutcome::Returned);
    assert_eq!(g.ended(1), [TurnEnd::Superseded]);
    assert!(g.reply_to("hello").is_none());
}

// ---- Local supersession of held replies (rule 3; "Local rules": commit interrupt) ----

/// Python: `test_newer_turn_cancels_a_held_reply`. The newer commit cancels the reply held for
/// silence before it plays, without waiting for AMD.
#[test]
fn a_newer_commit_cancels_a_reply_held_for_silence() {
    let mut g = Gate::listening();
    g.commit(1, "you have reached the screener");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("you have reached the screener");
    assert!(!g.authorized(speech), "held for silence");
    g.advance_to(200);
    g.commit(2, "state your name");
    assert!(g.cancelled(speech), "the held reply is cancelled at the newer commit");
    g.advance_to(5_000);
    assert!(!g.authorized(speech), "the cancelled reply never plays");
    assert!(
        g.gate_steps().iter().any(|s| matches!(s, GateStep::Superseded { turn: TurnId(1), .. }))
    );
}

/// Python: `test_newer_turn_without_a_reply_cancels_a_held_reply`.
#[test]
fn a_skip_reply_commit_also_cancels_a_held_reply() {
    let mut g = Gate::listening();
    g.commit(1, "you have reached the screener");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("you have reached the screener");
    let mut turn = info(2, "mm");
    turn.skip_reply = true;
    g.commit_info(turn);
    assert!(g.cancelled(speech));
    g.advance_to(5_000);
    assert!(!g.authorized(speech));
}

// ---- Silence hold (proposal "Silence"; spec §10.3) ----

/// The hold is timed from the VAD's boundary time, backdated by the silence the VAD measured,
/// not from the event's arrival (`detector.py:668`). Python:
/// `test_machine_reply_generates_during_the_hold_and_plays_after_it`: generation starts at
/// once, playback waits.
#[test]
fn a_silence_hold_releases_at_the_backdated_boundary_plus_the_threshold() {
    let mut g = Gate::listening();
    g.advance_to(100);
    g.vad_start(100);
    g.advance_to(1_000);
    // Speech ended at 1000 - 500 = 500 ms.
    g.vad_end(500);
    g.commit(1, "you have reached the screener");
    g.advance_to(1_100);
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("you have reached the screener");
    assert!(!g.authorized(speech), "the reply is generated during the hold, not played");
    assert_eq!(g.silence_holds(1), [(true, ts(1_100))]);

    g.advance_to(1_999);
    assert!(!g.authorized(speech), "500 + 1500 is not reached at 1999 ms");
    g.advance_to(2_000);
    assert!(g.authorized(speech), "released at 2000 ms, not at the event's 1000 + 1500");
    assert_eq!(g.silence_holds(1), [(true, ts(1_100)), (false, ts(2_000))]);
}

/// With no speech boundary since the previous commit, the boundary is the speech end estimated
/// from `eot_delay` (`detector.py:708`).
#[test]
fn without_a_boundary_the_hold_uses_the_eot_delay() {
    let mut g = Gate::listening();
    g.advance_to(1_000);
    let mut turn = info(1, "please leave a message after the tone");
    turn.metrics.end_of_turn_delay = Some(ms(400));
    g.commit_info(turn);
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("please leave a message after the tone");
    g.advance_to(2_099);
    assert!(!g.authorized(speech));
    g.advance_to(2_100);
    assert!(g.authorized(speech), "600 + 1500");
}

/// With no boundary and no `eot_delay`, the boundary is the commit.
#[test]
fn without_a_boundary_or_eot_delay_the_hold_starts_at_the_commit() {
    let mut g = Gate::listening();
    g.advance_to(1_000);
    g.commit(1, "please leave a message");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("please leave a message");
    g.advance_to(2_499);
    assert!(!g.authorized(speech));
    g.advance_to(2_500);
    assert!(g.authorized(speech));
}

/// A boundary before the previous commit does not count for the next turn: the next turn falls
/// back to its own commit.
#[test]
fn a_boundary_before_the_previous_commit_is_not_reused() {
    let mut g = Gate::listening();
    g.advance_to(100);
    g.vad_start(100);
    g.advance_to(500);
    g.vad_end(0);
    g.commit(1, "hello");
    g.decide(1, deny());
    g.advance_to(3_000);
    g.commit(2, "please leave a message");
    g.decide(2, held(SILENCE));
    let speech = g.only_reply_to("please leave a message");
    g.advance_to(4_499);
    assert!(!g.authorized(speech), "timed from the 3000 ms commit, not the 500 ms boundary");
    g.advance_to(4_500);
    assert!(g.authorized(speech));
}

/// Python: `test_resumed_speech_extends_the_hold_for_a_generated_reply`. Start of speech
/// disarms the hold; end of speech re-arms it with a new deadline; the stale expiry does not
/// release it (proposal: "a timer for any other deadline is stale").
#[test]
fn speech_that_restarts_during_the_hold_rearms_it() {
    let mut g = Gate::listening();
    g.advance_to(1_000);
    g.commit(1, "you have reached the screener");
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("you have reached the screener");
    g.advance_to(1_500);
    g.vad_start(0);
    g.advance_to(2_500);
    assert!(!g.authorized(speech), "the 2500 ms expiry is stale while the participant speaks");
    g.advance_to(3_000);
    g.vad_end(0);
    g.advance_to(4_499);
    assert!(!g.authorized(speech), "re-armed for 3000 + 1500");
    g.advance_to(4_500);
    assert!(g.authorized(speech));
    let holds: Vec<bool> = g.silence_holds(1).into_iter().map(|(held, _)| held).collect();
    assert_eq!(holds.first(), Some(&true));
    assert_eq!(holds.last(), Some(&false));
}

/// Silence that already elapsed at the decision releases it at once. Python:
/// `test_elapsed_silence_and_slow_inference_do_not_add_another_wait`.
#[test]
fn an_elapsed_silence_releases_at_once() {
    let mut g = Gate::listening();
    g.advance_to(100);
    g.vad_start(100);
    g.advance_to(500);
    g.vad_end(0);
    g.commit(1, "you have reached the screener");
    assert!(g.generations().is_empty(), "held for its decision");
    g.advance_to(2_500);
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("you have reached the screener");
    assert!(g.authorized(speech));
}

/// The participant still speaking at the decision holds the reply with no deadline.
#[test]
fn speech_in_progress_holds_without_a_deadline() {
    let mut g = Gate::listening();
    g.advance_to(100);
    g.vad_start(100);
    g.advance_to(800);
    g.commit(1, "you have reached");
    assert!(g.observed(RUN).iter().any(|o| matches!(
        o,
        AmdObservation::TurnCommitted { turn: TurnId(1), speaking: true, .. }
    )));
    g.decide(1, held(SILENCE));
    let speech = g.only_reply_to("you have reached");
    g.advance_to(10_000);
    assert!(!g.authorized(speech), "never released while the participant speaks");
    g.vad_end(0);
    g.advance_to(11_500);
    assert!(g.authorized(speech));
}

// Activation, the opening, `Detach`, the session's rules while attached, the reply payload and
// the observations.
#[path = "amd_gate_reply_tests.rs"]
mod reply;

// A second run, activation while a turn is in its hook, the session's options, deferred tool
// replies and the close.
#[path = "amd_gate_lifecycle_tests.rs"]
mod lifecycle;
