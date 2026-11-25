//! A real AMD run: the production executor, machine and legs over fake models. These stories
//! check the plumbing a caller relies on: the run classifies, answers the gate, completes after
//! the session detached it, and journals its own steps.

use super::{Call, listening};
use lk_agents_fuzz::fakes::amd::STAGE;
use lk_agents_fuzz::fakes::amd_models::AmdModelCall;
use lk_agents_fuzz::input::{AmdModel, AmdPersona, Count, Input};
use lk_agents_voice::amd::events::{AmdReason, Category, Outcome, Stage};
use lk_agents_voice::trace::{AmdStep, GateStep, Step};

const LINE: &str = "Hi, this is the clinic calling.";

impl Call {
    /// A real run starts, its classifier answering as `persona` would.
    async fn enter(&mut self, persona: AmdPersona, model: AmdModel, own_stt: bool, seed: u32) {
        self.send(Input::AmdEnter {
            persona,
            model,
            opening: None,
            own_stt,
            stt_fails: false,
            seed: Count(seed),
        })
        .await;
    }

    fn steps(&self) -> Vec<AmdStep> {
        (self.run.record.journal().iter())
            .filter_map(|(_, step)| match step {
                Step::Amd(step) => Some(step.clone()),
                _ => None,
            })
            .collect()
    }

    fn outcome(&self) -> Option<Outcome> {
        self.steps().into_iter().find_map(|step| match step {
            AmdStep::Completed { outcome } => Some(outcome),
            _ => None,
        })
    }

    fn calls(&self) -> Vec<AmdModelCall> {
        self.run.record.amd_models().iter().map(|(_, call)| call.clone()).collect()
    }
}

/// A seed whose first classification answers in time and is not refused.
const PROMPT: u32 = 7;

// Catches: an executor that never steps the machine on the gate's observations, never delivers
// its answers, or returns before the session's `Detached`.
#[test]
fn a_human_answer_completes_the_run_and_its_reply_plays() {
    let call = Call::run(listening(), async |call| {
        call.enter(AmdPersona::Human, AmdModel::Llm, false, PROMPT).await;
        call.advance(100).await;
        call.says("Hello?").await;
        call.advance(3_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let outcome = call.outcome().expect("the run completed");
    assert_eq!((outcome.category, outcome.reason), (Stage::Human, AmdReason::Finished));
    assert_eq!(call.heard_requests("Hello?").len(), 1, "{:?}", call.asked("Hello?"));
    let steps = call.steps();
    let bound = steps.iter().position(|s| matches!(s, AmdStep::TurnBound { has_text: true, .. }));
    let issued = steps.iter().position(|s| matches!(s, AmdStep::RequestIssued { .. }));
    let predicted = steps.iter().position(|s| {
        matches!(s, AmdStep::Predicted { category: Category::Human, source_req: Some(_), .. })
    });
    let completed = steps.iter().position(|s| matches!(s, AmdStep::Completed { .. }));
    assert!(
        bound < issued && issued < predicted && predicted < completed && bound.is_some(),
        "{steps:?}"
    );
    let detached = call.detached_at().expect("the session detached the run");
    let completed_at = (call.run.record.journal().iter())
        .find_map(|(at, s)| matches!(s, Step::Amd(AmdStep::Completed { .. })).then_some(at.at_ms))
        .expect("completed");
    assert!(detached <= completed_at, "amd_completed before the session's Detached");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

// Catches: a machine-stage decision losing its stage instructions or its silence on the way to
// the gate, or a close that never completes the run.
#[test]
fn a_voicemail_reply_carries_its_stage_and_the_close_completes_the_run() {
    let call = Call::run(listening(), async |call| {
        call.enter(AmdPersona::Voicemail, AmdModel::Llm, false, PROMPT).await;
        call.advance(100).await;
        call.says("Please leave a message after the tone.").await;
        call.advance(3_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.send(Input::AmdClose {}).await;
        call.advance(1_000).await;
    });
    let asked = call.made("Please leave a message after the tone.");
    assert!(asked.iter().any(|q| q.carries(STAGE)), "{asked:?}");
    let outcome = call.outcome().expect("the close completed the run");
    assert_eq!((outcome.category, outcome.reason), (Stage::MachineVm, AmdReason::Cancelled));
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

// Catches: a decision model asked for a menu, or a `wait` turn answered.
#[test]
fn a_decision_model_holds_a_wait_turn_and_reads_no_menu() {
    let call = Call::run(listening(), async |call| {
        call.enter(AmdPersona::Hold, AmdModel::Decision, false, PROMPT).await;
        call.advance(100).await;
        call.says("Please hold while we connect you.").await;
        call.advance(3_000).await;
        call.model_answers(LINE).await;
        call.advance(2_000).await;
    });
    assert!(call.unanswered("Please hold while we connect you."));
    let calls = call.calls();
    assert!(
        matches!(
            calls.as_slice(),
            [AmdModelCall::Choice { category: Some(Category::Wait), options: 7, .. }]
        ),
        "{calls:?}"
    );
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

// Catches: AMD's own STT never opened, never fed, or its finals not reaching the classifier next
// to the session's transcript.
#[test]
fn amd_stt_finals_reach_the_classifier_beside_the_session_transcript() {
    let call = Call::run(listening(), async |call| {
        call.enter(AmdPersona::Human, AmdModel::Llm, true, PROMPT).await;
        call.advance(100).await;
        call.says("Hello, who is calling?").await;
        call.advance(3_000).await;
        call.model_answers(LINE).await;
        call.advance(2_000).await;
    });
    let calls = call.calls();
    let Some(AmdModelCall::Classify { last_user, .. }) = calls.first() else { panic!("{calls:?}") };
    assert!(last_user.starts_with(r#"{"transcript_1": "Hello, who is calling?""#), "{last_user}");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

// Catches: a second run attached over a live one instead of refused at entry.
#[test]
fn a_second_run_is_refused_while_one_is_attached() {
    let call = Call::run(listening(), async |call| {
        call.enter(AmdPersona::Hold, AmdModel::Llm, false, PROMPT).await;
        call.advance(100).await;
        call.enter(AmdPersona::Human, AmdModel::Llm, false, PROMPT).await;
        call.advance(100).await;
        call.send(Input::AmdClose {}).await;
        call.advance(1_000).await;
    });
    let refused =
        call.run.record.journal().iter().any(
            |(_, s)| matches!(s, Step::Gate(GateStep::AttachRefused { run, .. }) if run.0 == 2),
        );
    assert!(refused);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}
