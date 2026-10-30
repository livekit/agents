//! A voice lost after its audio played ends the segment's audio leg once.
use lk_agents_fuzz::drive::{Outcome, Run, finish, simulate_with};
use lk_agents_fuzz::search;
use lk_agents_voice::events::{ErrorSource, SessionEvent};
use lk_agents_voice::trace::{PlayoutStep, Step};

// Catches: a synthesis failing after its audio played reported its segment as heard with no audio,
// so the segment ended its audio leg a second time as an unvoiced one; the synchroniser paired the
// next speech's words with that empty audio and showed them all before any of their audio played
// (`transcript_follows_audio`, campaign seed 20655; seed 6034 draws the scenario since the fakes'
// delays are hand-set).
#[test]
fn the_next_speech_words_follow_its_own_audio() {
    let setup = search::sample_setup(&mut fastrand::Rng::with_seed(6_034));
    let case = search::explore(setup, 6_034, 3_000).case;
    let (outcome, journal) = simulate_with(case.schedule, async {
        let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
        for input in &case.inputs {
            run.apply(input).await;
        }
        let journal: Vec<Step> = run.record.journal().iter().map(|(_, s)| s.clone()).collect();
        (finish(run, Outcome { case: case.clone(), ..Outcome::default() }).await, journal)
    });
    // The scenario: a speech's audio plays out whole, then its voice fails, then another speech's
    // words are shown.
    let played_then_lost = journal.iter().enumerate().find_map(|(i, step)| {
        let Step::Playout(PlayoutStep::PlaybackFinished { speech, interrupted: false, .. }) = step
        else {
            return None;
        };
        journal[i..]
            .iter()
            .position(|s| {
                matches!(s, Step::Emitted(SessionEvent::Error(e))
                if e.source == ErrorSource::Tts && e.speech == Some(*speech))
            })
            .map(|lost| (*speech, i + lost))
    });
    let (lost, at) = played_then_lost.expect("the scenario: a voice lost after its audio played");
    assert!(
        journal[at..].iter().any(|s| matches!(s,
            Step::Playout(PlayoutStep::TranscriptShown { speech, .. }) if *speech != lost)),
        "the scenario: another speech's words shown after the loss"
    );
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
