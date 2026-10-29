//! A speech cut after its segment completed unvoiced does not end that segment's audio again.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::search;
use lk_agents_voice::trace::{InterruptStep, PlayoutStep, SinkCall, Step};

// Catches: a segment whose voice ended with no audio was cut by its speech's interruption (the
// cut closed its words), then ended its audio leg as an unvoiced one anyway; the synchroniser
// opened a segment for that audio end and showed the next speech's words whole before any of
// their audio played (`transcript_follows_audio`, campaign seed 19354).
#[test]
fn the_next_speech_words_wait_for_their_audio() {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(19_354));
    let case = search::explore(setup, 19_354, 3_000).case;
    let (outcome, journal) = simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        let journal: Vec<Step> = run.record.journal().iter().map(|(_, s)| s.clone()).collect();
        (finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await, journal)
    });
    // The scenario: a speech's words forwarded with none of its audio pushed, then the speech
    // interrupted, then another speech's words forwarded.
    let cut_unvoiced = journal.iter().enumerate().find_map(|(i, step)| {
        let Step::Playout(PlayoutStep::TranscriptForwarded { speech, .. }) = step else {
            return None;
        };
        let rest = &journal[i..];
        let pushed = rest.iter().any(|s| matches!(s, Step::Playout(PlayoutStep::SinkCalled {
            speech: Some(p), call: SinkCall::Push { .. } }) if p == speech));
        let cut = rest.iter().position(|s| matches!(s,
            Step::Interrupt(InterruptStep::Interrupted { speech: c, .. }) if c == speech));
        cut.filter(|_| !pushed).map(|at| (*speech, i + at))
    });
    let (cut, at) = cut_unvoiced.expect("the scenario: a speech cut with its words but no audio");
    assert!(
        journal[at..].iter().any(|s| matches!(s,
            Step::Playout(PlayoutStep::TranscriptForwarded { speech, .. }) if *speech != cut)),
        "the scenario: another speech's words forwarded after the cut"
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
