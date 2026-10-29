//! The microphone gate an uninterruptible speech closed reopens before the next frame.
use lk_agents_fuzz::search;

// Catches: the gate reopening through the owner's queue, behind frames already queued, so frames
// pushed after the speech finished were silenced (a45633b37). The gate's effects settle in their
// batch, before another frame is forwarded. Seed 30097 is the gate window's only witness: with
// the effect sent through the queue again, it fails `forwarded_while_enabled`.
#[test]
fn microphone_is_not_silenced_after_uninterruptible_speech_finishes() {
    let seed = 30_097;
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let outcome = search::explore(setup, seed, 3_000);
    assert!(outcome.expectations.get("forwarded_while_enabled").is_some_and(|n| *n > 0));
    assert!(
        !outcome.failures.iter().any(|failure| failure.contract == "forwarded_while_enabled"),
        "{:#?}",
        outcome.failures
    );
}
