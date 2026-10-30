//! The microphone gate an uninterruptible speech closed reopens before the next frame.
use lk_agents_fuzz::input::Input;
use lk_agents_fuzz::search;

// Catches: the gate reopening through the owner's queue, behind frames already queued, so frames
// pushed after the speech finished were silenced (a45633b37). The gate's effects settle in their
// batch, before another frame is forwarded. Seed 27243 witnesses the gate window: with the effect
// sent through the queue again, it fails `forwarded_while_enabled` (re-picked from the campaign
// with the bug restored once the DST held a call's handle from its return; 30097 was). Seed
// 5390 fails it with the effect sent through the queue since the fakes' delays are hand-set.
#[test]
fn microphone_is_not_silenced_after_uninterruptible_speech_finishes() {
    let seed = 5_390;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    assert!(setup.stt, "the case must gate a recognition leg");
    let outcome = search::explore(setup, seed, 3_000);
    assert!(
        outcome
            .case
            .inputs
            .iter()
            .any(|input| matches!(input, Input::Say { allow: Some(false), .. })),
        "the case must play an uninterruptible speech"
    );
    assert!(outcome.expectations.get("forwarded_while_enabled").is_some_and(|n| *n > 0));
    assert!(
        !outcome.failures.iter().any(|failure| failure.contract == "forwarded_while_enabled"),
        "{:#?}",
        outcome.failures
    );
}
