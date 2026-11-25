//! The silence hold (proposal "Silence"; spec §10.3; invariant 7). Times are the fake VAD's
//! ground truth: the person stops speaking when the case reports the detector's end of speech.

use super::{Call, listening};
use lk_agents_fuzz::input::{AmdVerdict, Input, Millis, Ref};

const LINE: &str = "Hello, this is the clinic. Please call us back.";
/// `machine_silence_threshold` (spec §4.1).
const THRESHOLD_MS: u64 = 1_500;

/// The reply is generated during the hold and heard only after the threshold of silence.
/// Python: `test_machine_reply_generates_during_the_hold_and_plays_after_it`.
#[test]
fn a_machine_reply_is_generated_during_the_hold_and_heard_after_it() {
    let mut ended = 0;
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        ended = call.says("you have reached the screener").await;
        call.decide(AmdVerdict::Silence).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(4_000).await;
    });
    let made = call.made("you have reached the screener");
    assert_eq!(made.len(), 1);
    assert!(made[0].at_ms < ended + THRESHOLD_MS, "generation starts during the hold");
    let heard = call.heard_reply("you have reached the screener").expect("heard after the hold");
    assert!(heard >= ended + THRESHOLD_MS, "heard at {heard} ms, silence began at {ended} ms");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Proposal stories: participant speech that restarts during the hold without a commit, and
/// silence that expires, speech that restarts and ends, then the stale expiry. Python:
/// `test_resumed_speech_extends_the_hold_for_a_generated_reply`.
#[test]
fn speech_that_restarts_during_the_hold_rearms_it() {
    let mut ended_again = 0;
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        let ended = call.says("you have reached the screener").await;
        call.decide(AmdVerdict::Silence).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        // The person speaks again before the first deadline, and stops before it too: the
        // first deadline fires stale.
        let restart = (ended + 1_000).saturating_sub(call.now_ms());
        call.advance(restart).await;
        call.send(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).await;
        call.advance(200).await;
        call.send(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) }).await;
        ended_again = call.now_ms();
        call.advance(4_000).await;
    });
    let heard = call.heard_reply("you have reached the screener").expect("heard in the end");
    assert!(
        heard >= ended_again + THRESHOLD_MS,
        "heard at {heard} ms, the person last stopped at {ended_again} ms"
    );
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Proposal story: completion during a silence hold releases the held reply at once, for every
/// reason. Python: `test_completion_cancels_silence_wait_and_releases_the_held_reply`.
#[test]
fn completion_during_a_silence_hold_releases_it() {
    let mut ended = 0;
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        ended = call.says("please leave a message after the tone").await;
        call.decide(AmdVerdict::Silence).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(100).await;
        call.detach(AmdVerdict::Silence, false).await;
        call.advance(3_000).await;
    });
    let heard = call.heard_reply("please leave a message after the tone").expect("released");
    assert!(heard < ended + THRESHOLD_MS, "heard at {heard} ms: released by the completion");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A decision that asks for no silence is heard at once. Python:
/// `test_human_and_initial_uncertain_do_not_wait`.
#[test]
fn a_decision_without_silence_is_heard_at_once() {
    let mut ended = 0;
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        ended = call.says("hello this is Sam").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let heard = call.heard_reply("hello this is Sam").expect("heard");
    assert!(heard < ended + THRESHOLD_MS, "a human reply does not wait: heard at {heard} ms");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}
