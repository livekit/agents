//! A voice lost after its audio played ends the segment's audio leg once.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::search;

// Catches: a synthesis failing after its audio played reported its segment as heard with no audio,
// so the segment ended its audio leg a second time as an unvoiced one; the synchroniser paired the
// next speech's words with that empty audio and showed them all before any of their audio played
// (`transcript_follows_audio`, campaign seed 20655).
#[test]
fn the_next_speech_words_follow_its_own_audio() {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(20_655));
    let case = search::explore(setup, 20_655, 3_000).case;
    let outcome = simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await
    });
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
