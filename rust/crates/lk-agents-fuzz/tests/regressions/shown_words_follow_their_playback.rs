//! A shown word is the speech's whose playback released it, and counts against that playback.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::search;
use lk_agents_voice::trace::{PlayoutStep, Step};

// Catches: a speech whose playback report was lost gave up on it; the report arriving later still
// released its words, which the next speech's playout, the one listening by then, journalled as
// its own with nothing of its audio played (`transcript_follows_audio`, campaign seed 15283).
#[test]
fn a_late_report_shows_no_words_for_the_next_speech() {
    let (outcome, journal) = explored(15_283);
    assert!(
        journal.iter().any(|step| matches!(step, Step::Playout(PlayoutStep::FinishLost { .. }))),
        "the scenario: a playback report lost, then delivered late"
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}

/// The campaign's case for `seed`, run to its end: its outcome, and the journal of its inputs.
fn explored(seed: u64) -> (Outcome, Vec<Step>) {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(seed));
    let case = search::explore(setup, seed, 3_000).case;
    simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        let journal = run.record.journal().iter().map(|(_, step)| step.clone()).collect();
        let outcome = finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await;
        (outcome, journal)
    })
}
