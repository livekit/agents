//! Machine checks the Python ports do not make: exact wire text, option errors and completion
//! edges, driven through `step` and judged by the effects.

use super::*;
use crate::agent_session::machine::activity::scheduling::SpeechKind;
use crate::amd::effects::ClassifierItem;
use crate::amd::{Opening, ReplyDecision, SpeechObservation};
use std::time::UNIX_EPOCH;

const AGENT: &str = "agent";
const THRESHOLD: Duration = Duration::from_millis(1500);

fn setup() -> AmdSetup {
    AmdSetup {
        agent: AGENT.into(),
        session_id: "session".into(),
        entered_at: UNIX_EPOCH,
        session_stt: true,
        amd_stt: false,
        extract_menus: true,
        insert_user_items: false,
    }
}

struct Run {
    amd: Amd,
    now: Ts,
    last_turn: u64,
}

impl Run {
    fn new() -> Run {
        Run::with(AmdOptions::default(), setup())
    }

    fn with(options: AmdOptions, setup: AmdSetup) -> Run {
        let amd = Amd::new(AmdRunId(7), options, setup).unwrap();
        Run { amd, now: Ts::ZERO, last_turn: 0 }
    }

    fn active(options: AmdOptions, setup: AmdSetup) -> Run {
        let mut run = Run::with(options, setup);
        run.step(In::Callee(Callee::Answered));
        run
    }

    fn step(&mut self, input: In) -> Vec<AmdEffect> {
        self.amd.step(self.now, input)
    }

    fn advance(&mut self, ms: u64) {
        self.now = self.now.saturating_add(Duration::from_millis(ms));
    }

    fn observe(&mut self, observation: AmdObservation) -> Vec<AmdEffect> {
        self.step(In::Observed(observation))
    }

    /// Commits the next turn with the session's transcript.
    fn commit(&mut self, text: &str) -> (TurnId, Vec<AmdEffect>) {
        self.last_turn += 1;
        let turn = TurnId(self.last_turn);
        let effects = self.observe(AmdObservation::TurnCommitted {
            turn,
            transcript: text.into(),
            eot_delay: None,
            skip_reply: false,
            bound: true,
            speaking: false,
            agent: AGENT.into(),
            at: self.now,
        });
        (turn, effects)
    }

    /// Commits a turn with text and answers its request with `category`.
    fn turn(&mut self, text: &str, category: Category) -> (TurnId, Vec<AmdEffect>) {
        let (turn, effects) = self.commit(text);
        let req = request(&effects).expect("a request").req;
        self.advance(100);
        (turn, self.step(In::Classified { req, result: ok(category, 100) }))
    }
}

/// A classifier answer that took `ms` of model time.
fn ok(category: Category, ms: u64) -> Result<Classification, ClassifyError> {
    Ok(Classification { category, inference_duration: Duration::from_millis(ms) })
}

fn request(effects: &[AmdEffect]) -> Option<ClassifyRequest> {
    effects.iter().find_map(|e| match e {
        AmdEffect::Classify(request) => Some(request.clone()),
        _ => None,
    })
}

fn decision(effects: &[AmdEffect]) -> Option<(TurnId, ReplyDecision)> {
    effects.iter().find_map(|e| match e {
        AmdEffect::Answer(AmdAnswer::ReplyDecided { turn, decision, .. }) => {
            Some((*turn, decision.clone()))
        }
        _ => None,
    })
}

fn predictions(effects: &[AmdEffect]) -> Vec<Prediction> {
    effects
        .iter()
        .filter_map(|e| match e {
            AmdEffect::Emit(AmdEvent::Prediction(p)) => Some(p.clone()),
            _ => None,
        })
        .collect()
}

fn detach(effects: &[AmdEffect]) -> Option<(ReplyDecision, DetachCancel)> {
    effects.iter().find_map(|e| match e {
        AmdEffect::Answer(AmdAnswer::Detach { pending, cancel, .. }) => {
            Some((pending.clone(), *cancel))
        }
        _ => None,
    })
}

fn completed(effects: &[AmdEffect]) -> Option<Outcome> {
    effects.iter().find_map(|e| match e {
        AmdEffect::Emit(AmdEvent::Completed(outcome)) => Some(outcome.clone()),
        _ => None,
    })
}

fn reason(run: &Run) -> Option<AmdReason> {
    match run.amd.lifecycle() {
        Lifecycle::Finished(outcome) => Some(outcome.reason),
        _ => None,
    }
}

#[test]
fn options_refuse_what_the_constructor_refuses() {
    let refused = |change: fn(&mut AmdOptions)| {
        let mut options = AmdOptions::default();
        change(&mut options);
        options.validate().unwrap_err()
    };
    assert_eq!(refused(|o| o.idle_timeout = Duration::ZERO), AmdOptionsError::NotPositive);
    assert_eq!(refused(|o| o.max_uncertain_turns = 0), AmdOptionsError::NotPositive);
    assert_eq!(refused(|o| o.max_inference_timeouts = 0), AmdOptionsError::MaxInferenceTimeouts);
    let both = |o: &mut AmdOptions| {
        o.opening_text = Some("hi".into());
        o.opening_instructions = Some("greet".into());
    };
    assert_eq!(refused(both), AmdOptionsError::BothOpenings);
    assert_eq!(refused(|o| o.opening_text = Some(" ".into())), AmdOptionsError::EmptyOpeningText);
    assert!(AmdOptions::default().validate().is_ok());
    let defaults = AmdOptions::default();
    assert_eq!(defaults.machine_silence_threshold, THRESHOLD);
    assert_eq!((defaults.max_uncertain_turns, defaults.max_inference_timeouts), (3, 3));
}

#[test]
fn activation_answers_with_the_opening_and_arms_the_overall_deadline() {
    let options = AmdOptions { opening_text: Some("Hi there".into()), ..AmdOptions::default() };
    let mut run = Run::with(options, setup());
    // Before activation a turn is not bound: it replies as usual.
    let (turn, effects) = run.commit("hello");
    assert_eq!(decision(&effects), Some((turn, ReplyDecision::allow())));
    assert!(request(&effects).is_none());
    run.advance(1000);
    let effects = run.step(In::Callee(Callee::Answered));
    let deadline = run.now.saturating_add(Duration::from_secs(60));
    assert!(effects.contains(&AmdEffect::SetDeadline(AmdTimer::Overall, deadline)));
    let opening = Some(Opening::Text("Hi there".into()));
    assert!(
        effects.contains(&AmdEffect::Answer(AmdAnswer::Activated { run: AmdRunId(7), opening }))
    );
}

#[test]
fn voicemail_plays_once_and_an_unavailable_goodbye_reports_the_previous_stage() {
    let mut run = Run::active(AmdOptions::default(), setup());
    let (t1, effects) = run.turn("Leave a message.", Category::MachineVm);
    assert!(decision(&effects).unwrap().1.allow);
    let speech = SpeechId(1);
    let created = SpeechObservation::Created {
        kind: SpeechKind::TurnReply,
        add_to_chat_ctx: true,
        playing: false,
    };
    run.observe(AmdObservation::Speech { speech, event: created, at: run.now });
    run.observe(AmdObservation::ReplyCreated { turn: t1, speech });
    run.observe(AmdObservation::Speech {
        speech,
        event: SpeechObservation::PlayoutAdvanced,
        at: run.now,
    });
    run.observe(AmdObservation::Speech {
        speech,
        event: SpeechObservation::Done { interrupted: false, failed: false },
        at: run.now,
    });
    // Another voicemail turn gets no second message.
    let (t2, effects) = run.turn("Leave a message.", Category::MachineVm);
    assert_eq!(decision(&effects), Some((t2, ReplyDecision::deny())));
    let (_, effects) = run.turn("Goodbye.", Category::MachineUnavailable);
    assert_eq!(detach(&effects).unwrap().1, DetachCancel::ForbidReplies);
    let Lifecycle::Finished(outcome) = run.amd.lifecycle() else { panic!("finished") };
    assert!(outcome.voicemail_message_played);
    assert_eq!(outcome.category, Stage::MachineVm);
}

#[test]
fn empty_turns_reuse_the_pending_result_without_a_request() {
    let mut run = Run::active(AmdOptions::default(), setup());
    let (_, effects) = run.commit("Hello, you've reached");
    let req = request(&effects).unwrap();
    let (t2, effects) = run.commit("");
    assert!(request(&effects).is_none() && predictions(&effects).is_empty());
    let effects = run.step(In::Classified { req: req.req, result: ok(Category::MachineVm, 0) });
    let p = predictions(&effects);
    assert_eq!(
        p.iter().map(|p| (p.turn_id, p.reason)).collect::<Vec<_>>(),
        [(1, AmdReason::Prediction), (2, AmdReason::Reused)]
    );
    // The empty turn holds the right to reply.
    assert_eq!(decision(&effects).unwrap().0, t2);
    // With nothing pending, an empty turn reuses at once; the latest prediction stays.
    let (_, effects) = run.commit("");
    assert_eq!(predictions(&effects)[0].reason, AmdReason::Reused);
    let (_, effects) = run.commit("more");
    assert_eq!(request(&effects).unwrap().previous_prediction.unwrap().turn_id, 1);
}

#[test]
fn an_inference_error_is_uncertain_and_keeps_the_likely_human_mark() {
    let mut run = Run::active(AmdOptions::default(), setup());
    run.turn("Press 1.", Category::MachineIvr);
    run.turn("Hello?", Category::Human);
    let (_, effects) = run.commit("Hello? Anyone?");
    let req = request(&effects).unwrap();
    let effects =
        run.step(In::Classified { req: req.req, result: Err(ClassifyError::InvalidOutput) });
    let p = &predictions(&effects)[0];
    assert_eq!(
        (p.category, p.reason, p.inference_duration),
        (Category::Uncertain, AmdReason::InferenceError, None)
    );
    let (_, effects) = run.commit("Hi?");
    assert!(request(&effects).unwrap().likely_human);
}

#[test]
fn a_turn_waits_for_its_late_amd_final_before_it_classifies() {
    let both = AmdSetup { amd_stt: true, ..setup() };
    let mut run = Run::active(AmdOptions::default(), both);
    let (turn, effects) = run.commit("Hello");
    assert!(request(&effects).is_none());
    assert!(effects.contains(&AmdEffect::SetDeadline(
        AmdTimer::LateFinal(turn),
        run.now.saturating_add(LATE_AMD_FINAL_WAIT)
    )));
    run.advance(100);
    // A blank final does not end the wait.
    assert!(request(&run.step(In::AmdFinal(" ".into()))).is_none());
    let req = request(&run.step(In::AmdFinal("Hello there".into()))).unwrap();
    let ClassifierItem::User { transcript, .. } = &req.history[0] else { panic!("user item") };
    assert_eq!(
        transcript.for_classifier(),
        r#"{"transcript_1": "Hello", "transcript_2": "Hello there"}"#
    );
    // Without a final, the wait's end sends the request with the session text alone.
    let (turn, _) = run.commit("Second");
    run.advance(300);
    let req = request(&run.step(In::Timeout(AmdTimer::LateFinal(turn)))).unwrap();
    let ClassifierItem::User { transcript, .. } = req.history.last().unwrap() else { panic!() };
    assert_eq!(transcript.for_classifier(), r#"{"transcript_1": "Second", "transcript_2": ""}"#);
}

#[test]
fn idle_arms_when_nothing_is_busy_and_completes() {
    let mut run = Run::active(AmdOptions::default(), setup());
    run.observe(AmdObservation::AgentBusy { busy: true, at: run.now });
    run.advance(5000);
    let effects = run.observe(AmdObservation::AgentBusy { busy: false, at: run.now });
    let armed = run.now.saturating_add(Duration::from_secs(10));
    assert!(effects.contains(&AmdEffect::SetDeadline(AmdTimer::Idle, armed)));
    // A stale firing does nothing; the armed one completes.
    run.step(In::Timeout(AmdTimer::Idle));
    assert_eq!(reason(&run), None);
    run.advance(10_000);
    run.step(In::Timeout(AmdTimer::Idle));
    assert_eq!(reason(&run), Some(AmdReason::IdleTimeout));
}

#[test]
fn the_overall_timeout_releases_a_held_reply() {
    let mut run = Run::active(AmdOptions::default(), setup());
    let (t1, effects) = run.turn("Please state your name.", Category::MachineScreening);
    assert_eq!(decision(&effects).unwrap().0, t1);
    run.advance(60_000);
    // Anything at or after the deadline completes first.
    let effects = run.step(In::AmdFinal("late".into()));
    assert_eq!(reason(&run), Some(AmdReason::Timeout));
    assert_eq!(detach(&effects), Some((ReplyDecision::allow(), DetachCancel::Keep)));
}

#[test]
fn request_state_is_rendered_as_python_does() {
    let request = ClassifyRequest {
        req: ReqId(1),
        turn: TurnId(2),
        turn_id: 2,
        stage: Stage::MachineVm,
        previous_prediction: Some(PreviousPrediction {
            turn_id: 1,
            category: Category::MachineVm,
            reason: AmdReason::Prediction,
        }),
        likely_human: false,
        recommended: vec![Category::Human, Category::Wait],
        speech_duration: Duration::from_millis(1500),
        history: Vec::new(),
    };
    assert_eq!(
        request.state_json(),
        r#"{"stage": "machine-vm", "previous_prediction": {"category": "machine-vm", "reason": "prediction", "turn_id": 1}, "recommended_next_categories": ["human", "wait"], "speech_duration": 1.5}"#
    );
    // Non-ASCII is escaped, as `json.dumps` does.
    let both = TurnTranscript { session: Some("h\u{e9}llo".into()), amd: Some(String::new()) };
    assert_eq!(both.for_classifier(), r#"{"transcript_1": "h\u00e9llo", "transcript_2": ""}"#);
}

#[test]
fn close_and_an_early_detach_cancel_the_run() {
    let mut run = Run::new();
    let effects = run.step(In::Close);
    assert_eq!(reason(&run), Some(AmdReason::Cancelled));
    assert!(detach(&effects).is_some());
    let mut run = Run::active(AmdOptions::default(), setup());
    run.observe(AmdObservation::Detached { run: AmdRunId(7) });
    assert_eq!(reason(&run), Some(AmdReason::Cancelled));
    let outcome = completed(&run.step(In::LegsClosed)).unwrap();
    assert!(outcome.is_error());
}
