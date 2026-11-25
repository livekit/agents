//! Holds, keyed decisions, supersession and the customer hook (proposal "Transitions").

use super::{Call, listening};
use lk_agents_fuzz::drive::Setup;
use lk_agents_fuzz::input::{AmdVerdict, Input, Ref};
use lk_agents_voice::amd::{AmdObservation, TurnEnd};

const LINE: &str = "Hi, this is the clinic calling.";

/// Nothing answers a turn before its decision; once allowed, it is answered and heard once.
/// Python: `test_reply_waits_for_classification_and_silence`.
#[test]
fn a_turn_is_answered_only_after_its_decision() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("hello who is this").await;
        call.advance(2_000).await;
        assert!(call.unanswered("hello who is this"), "nothing answers before the decision");
        call.decide(AmdVerdict::Allow).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let heard = call.heard_requests("hello who is this");
    assert_eq!(heard.len(), 1, "one reply heard: {:?}", call.asked("hello who is this"));
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A denied turn (`wait`) keeps its words and is never answered; the next turn is decided on its
/// own. Python: `test_wait_skips_reply_and_allows_a_fresh_prediction`.
#[test]
fn a_denied_turn_keeps_its_words_and_is_never_answered() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("please hold").await;
        call.decide(AmdVerdict::Wait).await;
        call.advance(3_000).await;
        call.says("thank you for holding").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    assert!(call.unanswered("please hold"));
    assert!(call.saved_user_words().contains(&"please hold".to_string()));
    assert_eq!(call.heard_requests("thank you for holding").len(), 1);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A decision for a turn a newer one superseded answers nothing (protocol rules 2 and 3).
#[test]
fn a_stale_decision_answers_nothing() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("hello").await;
        call.says("is anyone there").await;
        call.send(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow }).await;
        call.advance(2_000).await;
        assert!(call.unanswered("hello") && call.unanswered("is anyone there"));
        call.decide(AmdVerdict::Allow).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    assert!(call.unanswered("hello"), "the superseded turn is never answered");
    assert_eq!(call.heard_requests("is anyone there").len(), 1);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A newer turn cancels the reply held for silence, without waiting for AMD. Python:
/// `test_newer_turn_cancels_a_held_reply`.
#[test]
fn a_newer_turn_cancels_a_held_reply() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("you have reached the screener").await;
        call.decide(AmdVerdict::Silence).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(300).await;
        call.says("state your name").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(5_000).await;
    });
    let held = call.made("you have reached the screener");
    assert_eq!(held.len(), 1, "the held reply was generated during the hold");
    assert_eq!(call.heard_reply("you have reached the screener"), None, "and never heard");
    assert!(call.heard_reply("state your name").is_some());
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

fn hooked() -> Setup {
    Setup { hook_bound_ms: Some(30_000), ..listening() }
}

/// Proposal story: a decision that arrives before the customer hook returns applies when it
/// returns.
#[test]
fn a_decision_before_the_hook_returns() {
    let call = Call::run(hooked(), async |call| {
        call.attach(None).await;
        call.says("hello who is this").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(2_000).await;
        assert!(call.unanswered("hello who is this"), "the hook has not returned");
        call.hook_returns().await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    assert_eq!(call.heard_requests("hello who is this").len(), 1);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Proposal story: completion while a turn is in its customer hook, for a human (the "AMD
/// completed" row allows) and for a stage the application handles (forbids). `Detached` does not
/// wait for the hook. Python: `test_terminal_result_completes_once_and_releases_turn_hooks`.
#[test]
fn completion_while_a_turn_is_in_its_hook() {
    for forbid in [false, true] {
        let call = Call::run(hooked(), async |call| {
            call.attach(None).await;
            call.says("hello this is Sam").await;
            call.detach(AmdVerdict::Allow, forbid).await;
            call.advance(100).await;
            assert!(call.detached_at().is_some(), "forbid={forbid}: Detached before the hook");
            call.hook_returns().await;
            call.advance(100).await;
            call.model_answers(LINE).await;
            call.advance(3_000).await;
        });
        let heard = call.heard_requests("hello this is Sam");
        assert_eq!(heard.len(), usize::from(!forbid), "forbid={forbid}: {heard:?}");
        assert!(call.failures().is_empty(), "forbid={forbid}: {:?}", call.failures());
    }
}

/// Proposal story: a second run attaches while the first run's turn is still in its hook. The old
/// turn keeps the decision its run settled; the new run gates only its own turns.
#[test]
fn a_second_attach_while_the_old_turn_is_in_its_hook() {
    let call = Call::run(hooked(), async |call| {
        call.attach(None).await;
        call.says("hello").await;
        call.detach(AmdVerdict::Allow, false).await;
        call.attach(None).await;
        call.hook_returns().await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.says("are you there").await;
        call.hook_returns().await;
        call.advance(2_000).await;
        assert!(call.unanswered("are you there"), "the second run holds its own turn");
        call.decide(AmdVerdict::Allow).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    // Whether the second run's entry hold also holds this reply is left open by the proposal
    // ("apply it without a gate"): only its creation is asserted.
    assert!(!call.asked("hello").is_empty(), "the old turn replies, as its run settled it");
    assert_eq!(call.heard_requests("are you there").len(), 1);
    let second_saw_the_old_turn = call.run.record.amd().iter().any(|(_, run, o)| {
        run.0 == 2 && matches!(o, AmdObservation::TurnCommitted { transcript, .. } if transcript == "hello")
    });
    assert!(!second_saw_the_old_turn);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Every exit reports `TurnEnded`: here a reply, a denial and a supersession (proposal "Turn
/// exits").
#[test]
fn every_turn_reports_how_it_ended() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("hello").await;
        call.says("hello again").await;
        call.decide(AmdVerdict::Wait).await;
        call.advance(1_000).await;
        call.says("thanks").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let ended: Vec<TurnEnd> = call
        .observed()
        .into_iter()
        .filter_map(|(_, o)| match o {
            AmdObservation::TurnEnded { how, .. } => Some(how),
            _ => None,
        })
        .collect();
    assert_eq!(ended, [TurnEnd::Superseded, TurnEnd::Denied, TurnEnd::Replied]);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}
